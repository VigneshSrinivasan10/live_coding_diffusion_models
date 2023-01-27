"""
Train a diffusion model on images.
"""
import os
import wandb 
wandb.init(project="fmnist-inference", entity="research")

import argparse
import pdb
from tqdm import tqdm
import torch as th
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

        

def save_all_steps_samples(args):
    dist_util.setup_dist(args)
    logger.configure()

    #pdb.set_trace()
    wandb.config = vars(args)       
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"Loading the model...")
    check = dist_util.load_state_dict(
        args.resume_checkpoint, map_location=dist_util.dev()
    )
    model.load_state_dict(
        check
    )
    logger.log(f"Finished loading the model")
    model.to(dist_util.dev())
        

    logger.log("Sampling...")
    sample_fn = (
            diffusion.p_all_sample_loop if not args.use_ddim else diffusion.ddim_all_sample_loop
    )
    args.batch_size = 10
    iters = args.num_samples // args.batch_size
    
    model.eval()
    sampling_method = 'ddpm' if not args.use_ddim else 'ddim'    
    
    ### Generate fake samples from the model 
    logger.log(f"Generating fake images from the model...")

    all_samples = []
    for ii in tqdm(range(iters)):
        cond = th.ones([args.batch_size])
        cond *= ii  
        model_kwargs = {"y": cond.to(dist_util.dev())}
        sample = sample_fn(
            model,
            (args.batch_size, 1, 32, 32), # remove hardcode later
            clip_denoised=True, 
            model_kwargs=model_kwargs,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.contiguous()
        all_samples += [sample]

    ### Saving images on wandb 
    grid_img = torchvision.utils.make_grid(torch.cat(all_samples, dim=0), nrow=10).float()
    wandb.log({"Samples": wandb.Image(grid_img)})

    
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=9600,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_all_steps=False,
        gpu_id='0',
        method_name='training_from_scratch',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    if args.save_all_steps:
        save_all_steps_samples(args)
    else:
        compute_fid_score(args)

        
if __name__ == "__main__":
    main()
        
