export WANDB_API_KEY=4145da060fe685cc3be8b5b886a7a1c14da76b7f
python examples/controlnet/train_controlnet_bridge.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --from_pt --per_worker_batch_size=35 --snr_gamma=5.0 --validation_image "./validation_images/image_0.jpg" "./validation_images/image_1.jpg" "./validation_images/image_2.jpg" "./validation_images/image_3.jpg" "./validation_images/image_4.jpg" --validation_prompt "move the blue cloth and place it on the right side of the white basket" "turn lever vertical to front" "Pick up the spatula and move it to the lower right corner of the table" "move the plastic ice cream scoop to the lower center of the countertop" "move the plastic ice cream scoop to the lower center of the countertop"
