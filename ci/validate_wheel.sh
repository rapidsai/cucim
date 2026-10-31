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

rapids-logger "validate that the wheel contains the compiled cucim.skimage extension"

# cucim.skimage falls back to slower code paths if this module is missing, so
# its absence would not show up as a test failure.
for wheel in "${wheel_dir_relative_path}"/*.whl; do
    # grep without -q reads the whole listing, so the pipe never breaks early
    if ! python -m zipfile -l "${wheel}" | grep "cucim/skimage/_cucim_skimage_cpp_ext" > /dev/null; then
        rapids-echo-stderr "${wheel} does not contain cucim/skimage/_cucim_skimage_cpp_ext"
        exit 1
    fi
done

rapids-logger "validate the stable ABI with 'abi3audit'"

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag (e.g. 'lib*' wheels).
# Filtering by '*abi*' avoids those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
