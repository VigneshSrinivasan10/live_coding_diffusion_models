export PYTHONPATH=$(pwd)
export OPENAI_LOGDIR=/home/vsrinivasan/Projects/guided-diffusion/test_results/256_from_scratch/

data_folder="/home/teamshare/Zalando_Research/fullbody_models_Jan2021/m/"

model_path="training/14_02_2022_15_52_35_training_from_scratch_multigpu/model100000.pt"
model_path="training/15_02_2022_10_05_27_finetuning_from_multigpu_training/model.pt"

# CLASSIFIER_FLAGS="--image_size 512 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 10 --timestep_respacing ddim25 --use_ddim True"
# --timestep_respacing 1000"
#" # --iterations 300000  --anneal_lr True"
python  scripts/stl_sample.py \
	--data_dir ${data_folder} \
	--image_size 256 \
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
	--save_all_steps True ${SAMPLE_FLAGS} 

# python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path training/07_02_2022_13_46_05_256_from_scratch/ema_0.9999.pt $SAMPLE_FLAGS

# python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path training/11_02_2022_15_29_56_training_from_scratch_multigpu/ema_0.9999.pt $SAMPLE_FLAGS --data_dir ${data_folder}
#python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path $1 $SAMPLE_FLAGS --data_dir ${data_folder}


