#!/bin/bash
param_shape=(512,512 3840,2160 3840,2160,3 192,192,192)
param_filt=(gabor gaussian median rank_order unsharp_mask sobel prewitt scharr roberts roberts_pos_diag roberts_neg_diag farid laplace meijering sato frangi hessian threshold_isodata threshold_otsu threshold_yen threshold_local threshold_li threshold_minimum threshold_mean threshold_triangle threshold_niblack threshold_sauvola apply_hysteresis_threshold threshold_multiotsu)
# param_filt=(rank_order )
param_dt=('fp64' 'fp32' 'fp16')
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_filters_bench.py -f $filt -i $shape -d $dt -t 10
            done
        done
    done