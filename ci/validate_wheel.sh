#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

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
