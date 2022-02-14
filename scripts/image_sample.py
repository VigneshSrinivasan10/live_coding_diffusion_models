"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import pdb
from tqdm import tqdm

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from ignite.metrics.gan import FID

from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import  Timer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import wandb 
wandb.init(project="adm-stl-inference", entity="research")

def main():
    args = create_argparser().parse_args()
    wandb.config = vars(args)       

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # print(model.out[0].weight.mean(), model.out[0].weight.var(),model.out[2].weight.mean())
    # import pdb; pdb.set_trace()
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print('Finished loading model...')
    
    # print(model.out[0].weight.mean(), model.out[0].weight.var(),model.out[2].weight.mean())
    
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        total_num_samples = args.num_samples
        iters = total_num_samples // args.batch_size

        ### Generate samples from the training data
        logger.log("creating data loader...")
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

        logger.log("Plotting real images from data loader...")
        all_real_samples = [] 
        for ii in tqdm(range(iters)):
            batch, cond = next(data)
            all_real_samples += [batch]
            grid_img = torchvision.utils.make_grid(batch, nrow=batch.shape[0]//4).float()
            wandb.log({"Real images": wandb.Image(grid_img)})
    
        timer = Timer()
        timer.begin()
        logger.log("Plotting fake images from the model...")
        all_fake_samples = [] 
        for ii in tqdm(range(iters)):
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.contiguous()

            all_fake_samples += [sample]
            grid_img = torchvision.utils.make_grid(sample, nrow=sample.shape[0]//4).float()
            wandb.log({"Fake images": wandb.Image(grid_img)})

        timer.stop()
        
        assert len(all_real_samples) == len(all_fake_samples)

        all_real_samples = torch.stack(all_real_samples)
        all_fake_samples = torch.stack(all_fake_samples)

        ### Compute FID score using Pytorch ignite.metrics
        print("computing FID score...")
        metric = FID()
        metric.update((all_fake_samples, all_real_samples))
        fid_score = metric.compute()
        metric.reset()
        wandb.log({"FID": fid_score})
        
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        '''
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

        
    '''
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    '''
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
