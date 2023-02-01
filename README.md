### Guided-diffusion

This is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

# Create the virtual env

```
conda create --name diffusion python=3.9
conda activate diffusion
pip install -r requirements.txt
conda install mpi4py
```

Finally
```
pip install -e .
```

# Download pre-trained models



# Testing models

```
sh test.sh 
```

To modify parameters in the test script
```
SAMPLE_FLAGS="--batch_size 2 --num_samples 20"
```

# DDIM
For sampling with 250 step DDIM:

```
use `--timestep_respacing ddim250 --use_ddim True`
```


# Training models

# Run the training

```
sh train.sh exp1
```

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
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 4 --lr 3e-4 --save_interval 1000 --weight_decay 0.05"
```

