#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param_shape=("512,512" "3840,2160" "3840,2160,3" "192,192,192")
param_filt=(label regionprops moments moments_central centroid inertia_tensor inertia_tensor_eigvals block_reduce shannon_entropy profile_line)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_measure_bench.py -f "$filt" -i "$shape" -d "$dt" -t 10
        done
    done
done

# # commenting out colocalization metrics below this point
# # (scikit-image 0.20 is not yet officially released)
# param_shape=(512,512 3840,2160 192,192,192)
# param_filt=(manders_coloc_coeff manders_overlap_coeff pearson_corr_coeff)
# param_dt=(float32 uint8)
# for shape in "${param_shape[@]}"; do
#     for filt in "${param_filt[@]}"; do
#         for dt in "${param_dt[@]}"; do
#             python cucim_measure_bench.py -f $filt -i $shape -d $dt -t 10
#         done
#     done
# done

# # only supports binary-valued images
# param_shape=(512,512 3840,2160 192,192,192)
# param_filt=(intersection_coeff)
# param_dt=(bool)
# for shape in "${param_shape[@]}"; do
#     for filt in "${param_filt[@]}"; do
#         for dt in "${param_dt[@]}"; do
#             python cucim_measure_bench.py -f $filt -i $shape -d $dt -t 10
#         done
#     done
# done
