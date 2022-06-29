#!/bin/bash
param_shape=(512,512)  #  3840,2160 64,64,64 256,256,256)
param_filt=(label regionprops moments moments_central centroid inertia_tensor inertia_tensor_eigvals block_reduce shannon_entropy profile_line)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_measure_bench.py -f $filt -i $shape -d $dt -t 10
        done
    done
done
