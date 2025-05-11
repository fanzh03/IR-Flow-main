#!/bin/bash


### training ###
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 


# Train
# derain
# python train.py -opt=/basicsr/options/Dehazing/train/rf-train-haze-UNet.yml




# Test
python test.py -opt=/basicsr/options/Dehazing/test/haze.yml
