#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use env var if set/non-empty, otherwise default to 10
MAX_DURATION="${CUCIM_BENCHMARK_MAX_DURATION:-10}"

param_shape=("512,512" "3840,2160" "3840,2160,3" "192,192,192")
param_filt=(binary_erosion binary_dilation binary_opening binary_closing isotropic_erosion isotropic_dilation isotropic_opening isotropic_closing remove_small_objects remove_small_holes erosion dilation opening closing white_tophat black_tophat medial_axis thin reconstruction)
param_dt=(uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_morphology_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done

# Note: Omit binary_*, medial_axis and thin from floating point benchmarks.
#       (these functions only take binary input).
param_filt_float=(remove_small_objects remove_small_holes erosion dilation opening closing white_tophat black_tophat reconstruction)
param_dt=(float32)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt_float[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_morphology_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done
