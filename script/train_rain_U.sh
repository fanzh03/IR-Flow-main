#!/bin/bash


### training ###
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 


# Train
# derain
python train.py -opt=/basicsr/options/Deraining/train/rf-train-deraining-Unet.yml




# Test
# python test.py -opt=/basicsr/options/Deraining/test/derain.yml
