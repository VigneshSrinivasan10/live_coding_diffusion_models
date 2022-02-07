date_time=$(date '+%d_%m_%Y_%H_%M_%S');
comments=$1
training_folder=${date_time}_${comments}
data_folder="/home/teamshare/Zalando_Research/fullbody_models_Jan2021/m/"

reload="False"
from_imagenet="False"

if [ "${reload}" = "True" ]; then
    if [ "${from_imagenet}" = "True" ]; then
	if [ "${image_size}" = "512" ]; then
	    model_path="models/512x512_diffusion.pt"
	else
	    model_path="models/256x256_diffusion_uncond.pt"
	fi
    else
	model_path="training/04_02_2022_16_58_12_256_from_scratch_test/model030000.pt" # add the model here 
    fi
    reload_ckpt="--use_checkpoint True --resume_checkpoint ${model_path}"
else
    reload_ckpt="--use_checkpoint False"
fi

export PYTHONPATH=$(pwd)
export OPENAI_LOGDIR=/home/vsrinivasan/Projects/guided-diffusion/training/${training_folder}

# CLASSIFIER_FLAGS="--image_size 512 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
TRAIN_FLAGS="--batch_size 4 --lr 3e-5 --save_interval 20000 --weight_decay 0.05"
mpiexec -n 8 python  scripts/image_train.py \
	--data_dir ${data_folder} \
	--image_size 256 \
	--class_cond False \
	--learn_sigma True \
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
