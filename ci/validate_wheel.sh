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

rapids-logger "validate the cp311 ABI3 wheel tag"

if ! find "${wheel_dir_relative_path}" -maxdepth 1 -name '*-cp311-abi3-*.whl' -print -quit | grep -q .; then
    rapids-echo-stderr "No cp311-abi3 wheel found in ${wheel_dir_relative_path}"
    exit 1
fi

rapids-logger "validate extension symbols with 'abi3audit'"

find \
    "${wheel_dir_relative_path}" \
    -maxdepth 1 \
    -type f \
    -name '*-cp311-abi3-*.whl' \
    -exec abi3audit --strict --summary --verbose '{}' \+
