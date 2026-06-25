#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use env var if set/non-empty, otherwise default to 10
MAX_DURATION="${CUCIM_BENCHMARK_MAX_DURATION:-10}"

param_shape=("512,512" "3840,2160" "3840,2160,3" "192,192,192")
param_filt=(multiscale_basic_features canny daisy structure_tensor hessian_matrix hessian_matrix_det shape_index corner_kitchen_rosenfeld corner_harris corner_shi_tomasi corner_foerstner corner_peaks match_template blob_dog blob_log blob_doh)
param_dt=(float64 float32)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_feature_bench.py -f "$filt" -i "$shape" -d "$dt" -t "$MAX_DURATION"
        done
    done
done
