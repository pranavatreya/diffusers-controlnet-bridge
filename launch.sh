cd /nfs/nfs3/users/pranav/diffusers-controlnet-bridge
source ~/.zshrc
conda activate jaxrl
export WANDB_API_KEY=4145da060fe685cc3be8b5b886a7a1c14da76b7f

python examples/controlnet/train_controlnet_bridge.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --from_pt --per_worker_batch_size=8 --snr_gamma=5.0 
