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
    dist_util.setup_dist(args)
    logger.configure()

    sampling_method = 'ddpm' if not args.use_ddim else 'ddim'
    if sampling_method == 'ddpm':
        num_steps = 1000 #args.diffusion_steps
    else:
        num_steps = args.timestep_respacing.replace('ddim', '')

    print('Sampling method: {} with steps'.format(sampling_method, num_steps))
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
    # '''
    
    ### Generate fake samples from the model 
    logger.log(f"Generating fake images from the model...")
    all_fake_samples = []
    #pdb.set_trace()
    counter = 0
    save_path = 'fid_score_verification/{}_{}_fake'.format(sampling_method, num_steps)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for ii in tqdm(range(iters)):
        sample = sample_fn(
            model,
            (args.batch_size, 3, 256, 256), # remove hardcode later
            clip_denoised=True, 
            model_kwargs={},
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.contiguous()
        sample = F.interpolate(sample, size=299)
        all_fake_samples += [sample] 

        for k, sam in enumerate(sample):
            save_image(sam/255,f"{save_path}/{counter:05d}.png") #.format(save_path, s))
            counter += 1
        if ii == 0: 
            # Plot only a subset of it on WB 
            grid_img = torchvision.utils.make_grid(sample, nrow=sample.shape[0]//4).float()
            wandb.log({"Fake images": wandb.Image(grid_img)})

    # '''
    #pdb.set_trace()
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    all_real_samples = [] 
    counter = 0
    save_path = 'fid_score_verification/original'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for ii in range(iters):
        batch, cond = next(data)
        batch = F.interpolate(batch, size=299)
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.contiguous()
        all_real_samples += [batch]
        # for k, sam in enumerate(batch):
        #     save_image(sam/255,f"{save_path}/{counter:05d}.png") #.format(save_path, s))
        #     counter += 1
    
            
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
        pdb.set_trace()
        metric.update((all_fake_samples, all_real_samples))
        fid_score = metric.compute()
        print(fid_score)
        metric.reset()
        wandb.log({"FID": fid_score})
    #timer.stop()
    #pdb.set_trace()
    print('Sampling method: {} with steps'.format(sampling_method, num_steps))
    with open("fid_scores/{}_{}.txt".format(sampling_method, num_steps), "w") as text_file:
        print(f"{fid_score}", file=text_file)
    
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
        
