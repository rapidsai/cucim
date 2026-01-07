#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use env var if set/non-empty, otherwise default to 10
MAX_DURATION="${CUCIM_BENCHMARK_MAX_DURATION:-10}"  # [web:1][web:10]

param_shape=("512,512" "3840,2160" "192,192,192")
param_filt=(convert_colorspace rgb2hed hed2rgb lab2lch lch2lab xyz2lab lab2xyz rgba2rgb label2rgb)
param_dt=(float32 uint8)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_color_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done

param_shape=("512,512" "3840,2160")
param_filt=(deltaE_cie76 deltaE_ciede94 deltaE_ciede2000 deltaE_cmc)
param_dt=("float32" "float64")
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_color_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done
