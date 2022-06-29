#!/bin/bash
param_shape=(512,512)  #  3840,2160 64,64,64 256,256,256)
param_filt=(binary_erosion binary_dilation binary_opening binary_closing remove_small_objects remove_small_holes erosion dilation opening closing white_tophat black_tophat thin reconstruction)
# Note: user-specified dtype ignored for binary_* functions and thin (these only accept binary input)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_morphology_bench.py -f $filt -i $shape -d $dt -t 10
        done
    done
done
