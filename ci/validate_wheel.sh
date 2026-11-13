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

rapids-logger "check that the wheel contains a single copy of libcucim"

# 'auditwheel repair' vendors an extra, renamed copy of libcucim (in the
# '*.libs' directory) when the plugins do not find 'libcucim.so.*' next to
# them. The Python extension and the plugins would then use different copies,
# each with its own cache and statistics.
if unzip -Z1 "$(echo "${wheel_dir_relative_path}"/*.whl)" | grep -E '\.libs/libcucim-'; then
    echo "ERROR: the wheel vendors a second copy of libcucim (listed above)" >&2
    exit 1
fi

rapids-logger "validate packages with 'twine'"

# shellcheck disable=SC2116
twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate the stable ABI with 'abi3audit'"

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag (e.g. 'lib*' wheels).
# Filtering by '*abi*' avoids those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
