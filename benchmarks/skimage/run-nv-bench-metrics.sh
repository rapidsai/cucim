#!/bin/bash
param_shape=(512,512 3840,2160 3840,2160,3 192,192,192)
param_filt=(structural_similarity) #  mean_squared_error normalized_root_mse peak_signal_noise_ratio normalized_mutual_information)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_metrics_bench.py -f $filt -i $shape -d $dt -t 1
        done
    done
done
