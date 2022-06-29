#!/bin/bash
param_shape=(512,512 3840,2160 3840,2160,3 192,192,192)
param_filt=(denoise_tv_chambolle calibrate_denoiser wiener unsupervised_wiener richardson_lucy)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_restoration_bench.py -f $filt -i $shape -d $dt -t 10
        done
    done
done
