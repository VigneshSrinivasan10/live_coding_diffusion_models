import copy
import functools
import os
import pdb
import time
from tqdm import tqdm 
    
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torchvision
from ignite.metrics.gan import FID

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        wandb='',
        num_classes=10,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.wandb = wandb  
            
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        #import pdb; pdb.set_trace()
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # self.model.to("cpu")
                # print(next(self.model.parameters()).is_cuda)
                # import pdb; pdb.set_trace()
                # self.model.load_state_dict(
                #     dist_util.load_state_dict(resume_checkpoint, map_location="cpu")
                # )             
                check = dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                )
                #pdb.set_trace()
                if '_diffusion' in resume_checkpoint:
                    for k,v in check.items():
                        if 'time_embed' in k or 'input_blocks' in k or 'middle_block' in k:
                            self.model.state_dict()[k].copy_(v.data)
                    logger.log(f"Finished loading only the input and middle blocks")
                else:
                    self.model.load_state_dict(
                        check
                    )
                    logger.log(f"Finished loading")

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            pdb.set_trace()
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        self.previous_best = 1e7
        counter_to_stop = 0 
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            #if self.step % self.log_interval == 0:
            #    logger.dumpkvs()
            if self.step % (self.log_interval*100) == 0:
                self.log_samples()    
            if self.step % self.save_interval == 0 and self.step > 0:
                #if self.step < 100000:
                self.save() 
                # elif self.running_avg_mse < self.previous_best:
                #     self.previous_best = self.running_avg_mse 
                #     counter_to_stop = 0
                #     self.save()
                # else:
                #     print('The running avg MSE is growing, so skipped saving the model')
                #     counter_to_stop += 1
                #     if counter_to_stop == 5:
                #         print('The running avg MSE is growing for the last 5xsave_interval iters...ABANDON SHIP!!!')
                #         import sys
                #         sys.exit()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def log_samples(self):
        self.use_ddim = False # remove hardcode later
        model_kwargs = {}
        sample_fn = (
            self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
        )

        iters = 1

        self.model.eval()
        # logger.log(f"Generating real images from the dataset...")
        all_real_samples = [] 
        for ii in range(iters):
            batch, cond = next(self.data)
            all_real_samples += [batch]
        
        grid_img = torchvision.utils.make_grid(batch, nrow=batch.shape[0]//4).float()
        self.wandb.log({"Real images": self.wandb.Image(grid_img)})

        ### Generate fake samples from the model 
        logger.log(f"Generating fake images from the model...")
        all_fake_samples = [] 
        for ii in tqdm(range(iters)):
            model_kwargs = {"y": cond.to(dist_util.dev())}
            sample = sample_fn(
                self.model,
                (self.batch_size, 1, 32, 32), # remove hardcode later
                clip_denoised=True, 
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.contiguous()

            all_fake_samples += [sample] 

        # Plot only a subset of it on WB 
        grid_img = torchvision.utils.make_grid(sample, nrow=sample.shape[0]//4).float()
        self.wandb.log({"Fake images": self.wandb.Image(grid_img)})


        
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        running_avg_loss = AverageMeter('Loss', ':.4e')
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                "y": cond[i : i + self.microbatch].to(dist_util.dev())
                #for k, v in cond.items()
            }
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            self.wandb.log({"loss": (losses["loss"]* weights).mean()})
                        
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    #filename = f"model{(self.step+self.resume_step):06d}.pt"
                    filename = f"model.pt"
                else:
                    #filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                    filename = f"ema_{rate}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                #bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                bf.join(get_blob_logdir(), f"opt.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()
        

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def return_avg(self):
        return self.avg
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Timer():
    """Computes time taken between two lines of code"""
    def __init__(self):
        self.reset()

    def begin(self): 
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.compute_time_taken()

    def compute_time_taken(self):
        hours, rem = divmod(self.end-self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}H:{:0>2}M:{:05.2f}S".format(int(hours),int(minutes),seconds))
        self.reset()

    def reset(self): 
        self.start = 0
        self.end = 0 
