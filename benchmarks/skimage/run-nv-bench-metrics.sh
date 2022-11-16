#!/bin/bash
param_shape=(512,512 3840,2160 3840,2160,3 192,192,192)
param_filt=(structural_similarity mean_squared_error normalized_root_mse peak_signal_noise_ratio normalized_mutual_information)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_metrics_bench.py -f $filt -i $shape -d $dt -t 4
        done
    done
done

# can only use integer dtypes and non-color images for the segmentation metrics
param_shape=(512,512 3840,2160 192,192,192)
param_filt=(adapted_rand_error contingency_table variation_of_information)
param_dt=(uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_metrics_bench.py -f $filt -i $shape -d $dt -t 4
        done
    done
done
