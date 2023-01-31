export PYTHONPATH=$(pwd)

model_path="training/31_01_2023_18_02_47_dummy/ema_0.9999.pt"
method_name="diffusion_fmnist_ddpm"

SAMPLE_FLAGS="--batch_size 10 --num_samples 100"

WANDB_NAME=${comments} CUDA_VISIBLE_DEVICES=0 python  scripts/fmnist_sample.py \
	--gpu_id 0 \
	--method_name ${method_name} \
	--image_size 32 \
	--class_cond False \
	--learn_sigma True \
	--num_channels 256 \
	--num_res_blocks 2 \
	--num_head_channels 64 \
	--attention_resolutions 32,16,8 \
	--dropout 0.0 \
	--diffusion_steps 1000 \
	--noise_schedule linear \
	--use_scale_shift_norm True \
	--resblock_updown True \
	--use_fp16 False \
	--use_new_attention_order True \
	--use_checkpoint True \
	--resume_checkpoint ${model_path} \
	${SAMPLE_FLAGS} 



