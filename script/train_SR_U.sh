#!/bin/bash


### training ###
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 


# Train
# derain
# python train.py -opt=/basicsr/options/SR/train/rf-train-SR-UNet.yml




# Test
python test.py -opt=/basicsr/options/SR/test/SR.yml
