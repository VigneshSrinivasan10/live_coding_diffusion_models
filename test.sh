export PYTHONPATH=$(pwd)
export OPENAI_LOGDIR=/home/vsrinivasan/Projects/guided-diffusion/test_results/256_from_scratch/

data_folder="/home/teamshare/Zalando_Research/fullbody_models_Jan2021/m/"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 8 --num_samples 8 --timestep_respacing 250"
#ddim25 --use_ddim True"
# --timestep_respacing 1000"

# python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path training/07_02_2022_13_46_05_256_from_scratch/ema_0.9999.pt $SAMPLE_FLAGS

# python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path training/11_02_2022_15_29_56_training_from_scratch_multigpu/ema_0.9999.pt $SAMPLE_FLAGS --data_dir ${data_folder}
python scripts/image_sample.py $MODEL_FLAGS --use_checkpoint True --model_path $1 $SAMPLE_FLAGS --data_dir ${data_folder}


