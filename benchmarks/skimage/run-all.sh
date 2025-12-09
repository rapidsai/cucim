#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

for file in ./cu*py
do
  echo "$file"
  time python "$file"
done
