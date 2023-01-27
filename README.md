### Guided-diffusion

This is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

# Create the virtual env
conda env create -f virtualenv.yml

# Data directory to for logging

Please change `${data_dir}` to the directory of your choice. 
```
${datadir}=/home/vsrinivasan/Projects/live_coding_diffusion_models/training/
export OPENAI_LOGDIR=${data_dir}/${training_folder}
```

# Download pre-trained models


# Testing models

```
sh test.sh 
```

```
SAMPLE_FLAGS="--batch_size 10 --num_samples 100"
```

# DDIM
For sampling with 25 step DDIM:

```
use `--timestep_respacing ddim25` and `--use_ddim True`
```


# Training models

# Single GPU training 
```
mpiexec -n 1 python scripts/image_train.py $TRAIN_FLAGS 
```

# Multi-GPU training
```
mpiexec -n N python scripts/image_train.py $TRAIN_FLAGS
```

# Training flags
```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
```

