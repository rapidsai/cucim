#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use env var if set/non-empty, otherwise default to 10
MAX_DURATION="${CUCIM_BENCHMARK_MAX_DURATION:-3}"

# Use env var if set/non-empty, otherwise default to "1,3,5,7"
RADII="${CUCIM_BENCHMARK_RANK_RADII:-1,3,5,10,20,30}"

param_shape=("512,512" "1920,1080")
param_filt=(autolevel enhance_contrast entropy equalize geometric_mean gradient majority maximum mean median minimum modal noise_filter pop subtract_mean sum threshold autolevel_percentile enhance_contrast_percentile gradient_percentile mean_percentile percentile pop_percentile subtract_mean_percentile sum_percentile threshold_percentile mean_bilateral pop_bilateral sum_bilateral)
param_dt=(uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_filters_rank_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION" --radii "$RADII"
        done
    done
done
