"""
Train a diffusion model on images.
"""
import os
import wandb 
wandb.init(project="adm-stl-inference", entity="research")

import argparse
import pdb
from tqdm import tqdm
import torch as th
import torchvision
from ignite.metrics.gan import FID
from torchvision.utils import save_image

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
    dist_util.setup_dist()
    logger.configure()

    #pdb.set_trace()
    wandb.config = vars(args)       
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    check = dist_util.load_state_dict(
        args.resume_checkpoint, map_location=dist_util.dev()
    )
    model.load_state_dict(
        check
    )
    logger.log(f"Finished loading")

    model.to(dist_util.dev())
        

    logger.log("Sampling...")
    sample_fn = (
            diffusion.p_all_sample_loop if not args.use_ddim else diffusion.ddim_all_sample_loop
    )
    args.batch_size = 1
    args.num_samples = 10
    
    iters = args.num_samples
    
    model.eval()
    sampling_method = 'ddpm' if not args.use_ddim else 'ddim'    
    save_dir = 'plots/{}'.format(sampling_method)
    
    ### Generate fake samples from the model 
    logger.log(f"Generating fake images from the model...")
    for ii in tqdm(range(iters)):
        sample = sample_fn(
            model,
            (args.batch_size, 3, 256, 256), # remove hardcode later
            clip_denoised=True, 
            model_kwargs={},
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.contiguous()

        save_path = '{}/img{}'.format(save_dir,ii)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        ### Saving images for creating gif
        for s, sam in enumerate(sample):  
            save_image(sam/255,f"{save_path}/{s:04d}.png") #.format(save_path, s))

        len_samples = sample.shape[0]        
        trimmed_samples = th.cat([sample[::len_samples//5], sample[-1].unsqueeze(0)])

        ### Saving images on wandb 
        grid_img = torchvision.utils.make_grid(trimmed_samples, nrow=trimmed_samples.shape[0]).float()
        wandb.log({"Samples": wandb.Image(grid_img)})


def compute_fid_score(args):
    dist_util.setup_dist()
    logger.configure()

    #pdb.set_trace()
    wandb.config = vars(args)       
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    check = dist_util.load_state_dict(
        args.resume_checkpoint, map_location=dist_util.dev()
    )
    model.load_state_dict(
        check
    )
    logger.log(f"Finished loading")

    model.to(dist_util.dev())
        

    logger.log("Sampling...")
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    compute_fid_score = True
    
    if compute_fid_score:
        iters = args.num_samples // args.batch_size
    else:
        iters = 1

    
    model.eval()
    ### Generate fake samples from the model 
    logger.log(f"Generating fake images from the model...")
    all_fake_samples = []
    #pdb.set_trace()
    for ii in tqdm(range(iters)):
        sample = sample_fn(
            model,
            (args.batch_size, 3, 256, 256), # remove hardcode later
            clip_denoised=True, 
            model_kwargs={},
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.contiguous()

        all_fake_samples += [sample] 

        if ii == 0: 
            # Plot only a subset of it on WB 
            grid_img = torchvision.utils.make_grid(sample, nrow=sample.shape[0]//4).float()
            wandb.log({"Fake images": wandb.Image(grid_img)})

            
    #pdb.set_trace()
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    all_real_samples = [] 
    for ii in range(iters):
        batch, cond = next(data)
        all_real_samples += [batch]
    #timer.begin()
    # Compute FID score 
    if compute_fid_score:
        logger.log(f"Computing FID score...")
        ### Generate samples from the training data 
        assert len(all_real_samples) == len(all_fake_samples)

        all_real_samples = th.stack(all_real_samples)[0]
        all_fake_samples = th.stack(all_fake_samples)[0]

        ### Compute FID score using Pytorch ignite.metrics
        metric = FID()
        metric.update((all_fake_samples, all_real_samples))
        fid_score = metric.compute()
        print(fid_score)
        metric.reset()
        wandb.log({"FID": fid_score})
    #timer.stop()
    #pdb.set_trace()

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
        
