#!/usr/bin/env bash

export PYTHONPATH=$HOME/Code/shuo/deep-networks/sssrlib:$HOME/Code/shuo/deep-networks/sssr:$HOME/Code/shuo/deep-networks/ptxl
export CUDA_VISIBLE_DEVICES=0

# image=/data/smore_simu/simu_data_2/sub-OAS30168_ses-d0059_T2w_initnorm_scale-3p5.nii.gz
image=/data/smore_simu/simu_data_2/sub-OAS30521_ses-d0118_run-01_T1w_initnorm_scale-4p9.nii.gz

rm -rf results_train
../scripts/train.py -i $image -o results_train -e 50000 -I 5000 -b 1
