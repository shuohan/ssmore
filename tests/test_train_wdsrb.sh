#!/usr/bin/env bash

export PYTHONPATH=$HOME/Code/shuo/deep-networks/sssrlib:$HOME/Code/shuo/deep-networks/sssr:$HOME/Code/shuo/deep-networks/ptxl
export CUDA_VISIBLE_DEVICES=0

# image=/data/smore_simu/simu_data_2/sub-OAS30168_ses-d0059_T2w_initnorm_scale-3p5.nii.gz
# image=/data/smore_simu/simu_data_2/sub-OAS30521_ses-d0118_run-01_T1w_initnorm_scale-4p9.nii.gz
image=/data/smore_simu_correct/simu_data/scale-4p9_fwhm-2p45/sub-OAS30167_ses-d0111_T1w_initnorm_scale-4p9_fwhm-2p45.nii.gz
kernel=$(echo $image | sed "s/\.nii\.gz$/.npy/")

num_epochs=10
save_step=10
batch_size=10
learning_rate=1e-4
width=32
depth=8

output_dir=results_train_e${num_epochs}_b${batch_size}_ps-mid_padding_w${width}_interp
rm -rf $output_dir
../scripts/train.py -i $image -o $output_dir -e $num_epochs \
    -I $save_step -b $batch_size -d ${depth} -w ${width} -R 9 \
    -l $learning_rate -s $kernel -P
