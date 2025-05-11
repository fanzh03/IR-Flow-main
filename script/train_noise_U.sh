#!/bin/bash


### training ###
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 


# Train
# denoising
python train.py -opt=/basicsr/options/Denoising/train/rf-train-denoising-Unet.yml


# Test
# python test.py -opt=/basicsr/options/Denoising/test/denoise.yml