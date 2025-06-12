#!/bin/bash
param_shape=("512,512" "3840,2160" "3840,2160,3" "192,192,192")
param_filt=(phase_cross_correlation optical_flow_tvl1 optical_flow_ilk)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_registration_bench.py -f "$filt" -i "$shape" -d "$dt" -t 10
        done
    done
done
