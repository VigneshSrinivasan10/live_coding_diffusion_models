export PYTHONPATH=$(pwd)

model_path="training/31_01_2023_19_58_33_exp1/ema_0.9999_054000.pt"
method_name="diffusion_fmnist_ddpm_54k"

export OPENAI_LOGDIR=$(pwd)/testing/${method_name}

SAMPLE_FLAGS="--batch_size 2 --num_samples 20" # --timestep_respacing ddim27 --use_ddim True"

#WANDB_NAME=${method_name}
CUDA_VISIBLE_DEVICES=0 python  scripts/fmnist_sample.py \
	--gpu_id 0 \
	--method_name ${method_name} \
	--image_size 32 \
	--class_cond True \
	--learn_sigma False \
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



