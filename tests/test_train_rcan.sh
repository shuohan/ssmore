#!/usr/bin/env bash

export PYTHONPATH=$HOME/Code/shuo/deep-networks/sssrlib:$HOME/Code/shuo/deep-networks/sssr:$HOME/Code/shuo/deep-networks/ptxl:$HOME/Code/shuo/utils/resize
export CUDA_VISIBLE_DEVICES=0

# image=/data/smore_simu/simu_data_2/sub-OAS30168_ses-d0059_T2w_initnorm_scale-3p5.nii.gz
# image=/data/smore_simu/simu_data_2/sub-OAS30521_ses-d0118_run-01_T1w_initnorm_scale-4p9.nii.gz
# image=/data/smore_simu_correct/simu_data/scale-4p9_fwhm-2p45/sub-OAS30167_ses-d0111_T1w_initnorm_scale-4p9_fwhm-2p45.nii.gz
# image=/data/smore_simu_correct/simu_data/scale-4p9_fwhm-2p45/sub-OAS30167_ses-d0111_T1w_initnorm_scale-4p9_fwhm-2p45.nii.gz
# image=/data/smore_simu_correct/simu_data/scale-2p0_fwhm-2p0/sub-OAS30167_ses-d0111_T1w_initnorm_scale-2p0_fwhm-2p0.nii.gz
type=scale-4p9_fwhm-6p125
image=/data/smore_simu_same_fov/simu_data/${type}/sub-OAS30167_ses-d0111_T1w_initnorm_${type}.nii.gz
kernel=$(echo $image | sed "s/\.nii\.gz$/.npy/")

num_epochs=5
num_batches=300
following_num_batches=30
# num_epochs=1
# num_batches=10
# following_num_batches=1

patch_size=32
save_step=1000
batch_size=16
learning_rate=1e-4
num_channels=64
num_blocks=8
num_groups=2
pred_epoch_step=1
pred_batch_step=100
pred_following_batch_step=10

name=$(basename $image | sed "s/\.nii\.gz$//")
output_dir=results_rcan_${name}_ne${num_epochs}_nb${num_batches}_bs${batch_size}_nf${following_num_batches}_nc${num_channels}_nb${num_blocks}_ng${num_groups}_lr${learning_rate}
rm -rf $output_dir
../scripts/train.py -i $image -o $output_dir -e $num_epochs \
    -S $save_step -B $batch_size -d ${num_blocks} -w ${num_channels} \
    -g $num_groups -l $learning_rate -b $num_batches \
    -f $following_num_batches -E $pred_epoch_step -s $kernel \
    -P $pred_batch_step -p ${patch_size} ${patch_size} \
    -F ${pred_following_batch_step}
