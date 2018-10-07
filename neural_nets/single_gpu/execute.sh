#!/bin/bash
module purge
module load shared
module load cuda90/toolkit/9.0.176
module load cudnn/7.0
module load tensorflow/1.8.0p

CUDA_VISIBLE_DEVICES=7 /cm/shared/anaconda2/bin/python train_01.py
