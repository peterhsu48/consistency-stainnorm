import copy
import functools
import os
import time
import logging

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from skimage.color import rgb2hsv

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
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        args=None,
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
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.args = args

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

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.args.resume_opt_checkpoint:
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

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        if self.args.resume_online_checkpoint:
            if dist.get_rank() == 0:
                logging.info(f"loading online checkpoint: {self.args.resume_online_checkpoint}")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        self.args.resume_online_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        if self.args.resume_opt_checkpoint:
            logging.info(f"loading opt checkpoint: {self.args.resume_opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                self.args.resume_opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
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
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps
        self.target_model_master_params = list(self.target_model.parameters())

        if target_model:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()
            
            # https://github.com/openai/consistency_models/pull/19/commits/52f689a7e5da28612f7232ab0f75b3d3474ff660
            if self.use_fp16:
                self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                    self.target_model.named_parameters()
                )
                self.target_model_master_params = make_master_params(
                    self.target_model_param_groups_and_shapes
                )

        if teacher_model:
            self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.global_step = self.args.global_step

        if self.args.hue:
            logging.info("performing hue conditioning instead of grayscale")

    def _load_and_sync_target_parameters(self):
        if self.args.resume_target_checkpoint:
            if dist.get_rank() == 0:
                logging.info(f"loading target checkpoint: {self.args.resume_target_checkpoint}")
                self.target_model.load_state_dict(
                    dist_util.load_state_dict(
                        self.args.resume_target_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())

    def run_loop(self):
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            start_time = time.time()
            for batch, cond in self.data:
                self.run_step(batch, cond)
                if self.global_step % self.log_interval == 0:
                    logger.dumpkvs()
            end_time = time.time()
            logging.info(f"epoch {epoch} took {end_time - start_time} seconds")
            logging.info(f"global step to resume from: {self.global_step}")
            self.save(epoch=epoch)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            self.step += 1
            self.global_step += 1

        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            # https://github.com/openai/consistency_models/pull/19/commits/52f689a7e5da28612f7232ab0f75b3d3474ff660
            if self.use_fp16:
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        i = 0
        # micro = batch.to(dist_util.dev()) # [-1.,1.] BCHW
        micro = batch # to device calls moved when implementing hue
        micro_cond = {
            k: v[i : i + self.microbatch].to(dist_util.dev())
            for k, v in cond.items()
        }
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        
        if self.args.hue:
            # lazy variable naming, gray represents the conditioning information
            micro_rgb = ((micro + 1.) * 127.5).type(th.uint8) # uint8, tensor, BCHW, [0, 255], loss of precision
            micro_rgb = th.permute(micro_rgb, (0,2,3,1)) # BHWC
            gray = rgb2hsv(micro_rgb)[:,:,:,0] # requires uint8, BHWC, tensor or numpy
            # gray is numpy, float64, BHW, [0.,1.], represents hue channel
            gray = th.unsqueeze(th.from_numpy(gray), 1).type(th.float32).to(dist_util.dev()) # torch, B1HW, [0.,1.], float32
        elif self.args.gray_and_hue:
            # grayscale
            gray = th.unsqueeze(micro[:,0] * 0.2125 + micro[:,1] * 0.7154 + micro[:,2] * 0.0721, dim=1) # B1HW, [-1.,1.]
            # lazy variable naming, gray represents the conditioning information
            micro_rgb = ((micro + 1.) * 127.5).type(th.uint8) # uint8, tensor, BCHW, [0, 255], loss of precision
            micro_rgb = th.permute(micro_rgb, (0,2,3,1)) # BHWC
            hue = rgb2hsv(micro_rgb)[:,:,:,0] # requires uint8, BHWC, tensor or numpy
            # gray is numpy, float64, BHW, [0.,1.], represents hue channel
            hue = th.unsqueeze(th.from_numpy(hue), 1).type(th.float32) # torch, B1HW, [0.,1.], float32
            if self.args.round:
                hue = th.round(hue * 360.) / 180. - 1. # torch, B1HW, [-1.,1.], float32
            gray = th.cat([gray, hue], dim=1).to(dist_util.dev()) #B2HW
        else:
            # grayscale
            gray = th.unsqueeze(micro[:,0] * 0.2125 + micro[:,1] * 0.7154 + micro[:,2] * 0.0721, dim=1).to(dist_util.dev()) # B1HW, [-1.,1.]

        micro = micro.to(dist_util.dev())
        
        ema, num_scales = self.ema_scale_fn(self.global_step)
        if self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.ddp_model,
                micro,
                num_scales,
                target_model=self.target_model,
                model_kwargs=micro_cond,
                args=self.args,
                gray=gray,
            )
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        losses = compute_losses()

        loss = (losses["loss"] * weights).mean()

        logging.info(f"loss: {loss.item()}")

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)

    def save(self, epoch=None):
        import blobfile as bf

        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"online_model_epoch_{epoch}.pt"
                else:
                    filename = f"ema_{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        logger.log("saving optimizer state...")
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt_epoch_{epoch}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        if dist.get_rank() == 0:
           if self.target_model:
                logger.log("saving target model state")
                filename = f"target_model_epoch_{epoch}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_model.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


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
