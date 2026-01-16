#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use env var if set/non-empty, otherwise default to 10
MAX_DURATION="${CUCIM_BENCHMARK_MAX_DURATION:-10}"  # [web:1][web:10]

param_shape=("512,512" "3840,2160" "3840,2160,3" "192,192,192")
param_filt=(equalize_adapthist cumulative_distribution equalize_hist rescale_intensity adjust_gamma adjust_log adjust_sigmoid is_low_contrast match_histograms)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_exposure_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done
