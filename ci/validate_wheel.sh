#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

wheel_dir_relative_path=$1

rapids-logger "validate packages with 'pydistcheck'"

# shellcheck disable=SC2116
pydistcheck \
    --inspect \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

# shellcheck disable=SC2116
twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate the stable ABI with 'abi3audit'"

if ! compgen -G "${wheel_dir_relative_path}/*-cp312-abi3-*.whl" > /dev/null; then
    rapids-echo-stderr "No cp312-abi3 wheel found in ${wheel_dir_relative_path}"
    exit 1
fi

abi3audit \
    --strict \
    --summary \
    --verbose \
    "${wheel_dir_relative_path}"/*.whl
