"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

Modified from https://github.com/openai/consistency_models/blob/main/scripts/image_sample.py
"""

import argparse
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample

# -- CM-Camelyon -- #
import cm.camelyon
import matplotlib.pyplot as plt
import logging
import random
from skimage.color import rgb2hsv
# -- CM-Camelyon -- #

def main():
    args = create_argparser().parse_args()

    th.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dist_util.setup_dist()
    logger.configure(dir=args.log_path)

    logging.basicConfig(filename=args.camelyon_log_file, level=logging.DEBUG)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logging.info(f"loaded model from {args.model_path}")

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    dl = cm.camelyon.get_data_loader(target_domain=False, data_dir=args.data_dir, batch_size=args.batch_size, split_file=args.split_file)

    all_labels = []

    if args.round:
        logging.info("using hue rounding")

    start_time = time.time()
    # while len(all_images) * args.batch_size < args.num_samples:
    for batch_idx, (source_x, y, metadata) in enumerate(dl):
        # source_x = source_x.to(dist_util.dev()) # to device calls moved when implementing hue
        source_x = source_x / 127.5 - 1 # [-1,1]

        if args.hue:
            # lazy variable naming, gray represents the conditioning information
            source_x_rgb = ((source_x + 1.) * 127.5).type(th.uint8) # uint8, tensor, BCHW, [0, 255], loss of precision
            source_x_rgb = th.permute(source_x_rgb, (0,2,3,1)) # BHWC
            gray = rgb2hsv(source_x_rgb)[:,:,:,0] # requires uint8, BHWC, tensor or numpy
            # gray is numpy, float64, BHW, [0.,1.], represents hue channel
            gray = th.unsqueeze(th.from_numpy(gray), 1).type(th.float32).to(dist_util.dev()) # torch, B1HW, [0.,1.], float32
        elif args.gray_and_hue:
            # grayscale
            gray = th.unsqueeze(source_x[:,0] * 0.2125 + source_x[:,1] * 0.7154 + source_x[:,2] * 0.0721, dim=1) # B1HW, [-1.,1.]
            # lazy variable naming, gray represents the conditioning information
            source_x_rgb = ((source_x + 1.) * 127.5).type(th.uint8) # uint8, tensor, BCHW, [0, 255], loss of precision
            source_x_rgb = th.permute(source_x_rgb, (0,2,3,1)) # BHWC
            hue = rgb2hsv(source_x_rgb)[:,:,:,0] # requires uint8, BHWC, tensor or numpy
            # gray is numpy, float64, BHW, [0.,1.], represents hue channel
            hue = th.unsqueeze(th.from_numpy(hue), 1).type(th.float32) # torch, B1HW, [0.,1.], float32
            if args.round:
                hue = th.round(hue * 360.) / 180. - 1. # torch, B1HW, [-1.,1.], float32
            gray = th.cat([gray, hue], dim=1).to(dist_util.dev()) #B2HW
        else:
            # grayscale
            gray = th.unsqueeze(source_x[:,0] * 0.2125 + source_x[:,1] * 0.7154 + source_x[:,2] * 0.0721, dim=1).to(dist_util.dev()) # B1HW, [-1.,1.]

        source_x = source_x.to(dist_util.dev())

        all_images = []
        model_kwargs = {}

        sample = karras_sample(
            diffusion,
            model,
            (source_x.shape[0], 3, args.image_size, args.image_size), # -- CM-Camelyon -- #
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            ts=ts,
            # -- CM-Camelyon -- #
            source_x=source_x,
            gray=gray,
            args=args,
            # -- CM-Camelyon -- #
        )
        # sample is [-1.,1.] BCHW torch
        assert sample.shape[1:] == (3,96,96), f"{sample.shape}"

        sample = (sample + 1) * 0.5 # [0.,1.]

        for im in range(sample.shape[0]):
            subset_idx = batch_idx * args.batch_size + im
            dataset_idx = dl.dataset.indices[subset_idx]
            original_file_name = dl.dataset.dataset._input_array[dataset_idx] # patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png
            components = original_file_name.split("/")
            save_file_name = os.path.join(args.norm_folder, components[-2], components[-1])
            if not os.path.exists(os.path.split(save_file_name)[0]):
                os.makedirs(os.path.split(save_file_name)[0])
            torchvision.utils.save_image(sample[im], save_file_name)

        current_time = time.time()
        logging.info(f"Normalization has taken {current_time - start_time} seconds so far")

    dist.barrier()
    logger.log("sampling complete")
    end_time = time.time()

    logger.log(f"Total sampling time (s): {end_time - start_time}")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        ts="",
        data_dir=None,
        log_path="./logs",
        seed=1234,
        camelyon_log_file="./logs/log.log",
        split_file=None,
        norm_folder="./cm_norm_patches",
        ilvr=False,
        ilvr_phi=0.0,
        hue=False,
        gray_and_hue=False,
        round=False,
        timestep_offset=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
