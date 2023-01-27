date_time=$(date '+%d_%m_%Y_%H_%M_%S');
comments=$1
training_folder=${date_time}_${comments}

export PYTHONPATH=$(pwd)

TRAIN_FLAGS="--batch_size 64 --lr 3e-6 --save_interval 10000 --weight_decay 0.05" # --iterations 300000  --anneal_lr True"
WANDB_NAME=${comments} mpiexec -n 1 python  scripts/image_train.py \
	--data_dir "./" \
	--image_size 32 \
	--num_classes 10 \
	--class_cond True \
	--learn_sigma False \
	--num_channels 256 \
	--num_res_blocks 2 \
	--num_head_channels 64 \
	--attention_resolutions 32,16,8 \
	--dropout 0.1 \
	--diffusion_steps 1000 \
	--noise_schedule linear \
	--use_scale_shift_norm True \
	--resblock_updown True \
	--use_fp16 True \
	--use_new_attention_order True \
	${reload_ckpt} \
	${TRAIN_FLAGS}
