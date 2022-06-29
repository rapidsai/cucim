#!/bin/bash
param_shape=(512,512)  #  3840,2160 64,64,64 256,256,256)
param_filt=(resize resize_local_mean rescale rotate downscale_local_mean warp_polar integral_image pyramid_gaussian pyramid_laplacian)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_transform_bench.py -f $filt -i $shape -d $dt -t 10
        done
    done
done
