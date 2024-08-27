import fnmatch
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        if len(filtered_paths) == 0:
            print("Warning: glob_to_path_list didn't find any paths")
        path_list += filtered_paths
    return path_list

class BridgeDataset:
    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 25000,
        cache: bool = False,
        train: bool = True,
        **kwargs,
    ):
        logging.warning("I'm the real BridgeDataset")
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.cache = cache
        self.is_train = train

        # construct a dataset for each sub-list of paths
        datasets = []
        for i, sub_data_paths in enumerate(data_paths):
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        # We need to create a shuffle buffer whether we are training or evaluating
        for i in range(len(datasets)):
            datasets[i] = (
                datasets[i]
                .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
            )

        if train:
            # repeat the datasets
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard, and for validation as 
        # well it ensures the number of batches we sample is less than the validation size
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=True
        )

        # Filter out trajectories without language instructions
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language"] != "")
        )

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._transform_images, num_parallel_calls=tf.data.AUTOTUNE)

        # cache if desired
        if self.cache:
            dataset = dataset.cache()

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "steps/observation/image_0": tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),  # Encoded images as a sequence
        "steps/observation/visual_trajectory": tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),  # Encoded images as a sequence
        "steps/language_instruction": tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),   # Language as a sequence
    }

    def _decode_example(self, serialized_example):
        # Parse the serialized example
        parsed_features = tf.io.parse_single_example(serialized_example, self.PROTO_TYPE_SPEC)

        # Decode images
        images_decoded = tf.map_fn(
            fn=lambda x: tf.io.decode_jpeg(x, channels=3),
            elems=parsed_features["steps/observation/image_0"],
            fn_output_signature=tf.TensorSpec(shape=[256, 256, 3], dtype=tf.uint8)
        )
        annotated_images_decoded = tf.map_fn(
            fn=lambda x: tf.io.decode_jpeg(x, channels=3),
            elems=parsed_features["steps/observation/visual_trajectory"],
            fn_output_signature=tf.TensorSpec(shape=[256, 256, 3], dtype=tf.uint8)
        )

        # restructure the dictionary into the downstream format
        return {
            "pixel_values": annotated_images_decoded,
            "conditioning_pixel_values": images_decoded,
            "language": parsed_features["steps/language_instruction"],
        }
    
    def _transform_images(self, traj):
        """
        As per the original flax controlnet training code, we resize to 512x512 and normalize to [-1, 1]
        """
        traj["pixel_values"] = tf.image.resize(traj["pixel_values"], [512, 512], tf.image.ResizeMethod.BICUBIC)
        traj["conditioning_pixel_values"] = tf.image.resize(traj["conditioning_pixel_values"], [512, 512], tf.image.ResizeMethod.BICUBIC)

        traj["pixel_values"] = (traj["pixel_values"] / 255.0 - 0.5) / 0.5
        traj["conditioning_pixel_values"] = (traj["conditioning_pixel_values"] / 255.0 - 0.5) / 0.5

        return traj

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()