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

# The packaged SONAME is preloaded by cucim. Check for a missing copy and
# for any extra shared libraries, including renamed auditwheel copies.
libcucim_files=$(unzip -Z1 "$(echo "${wheel_dir_relative_path}"/*.whl)" |
    awk '/(^|\/)libcucim(-[^/]+)?\.so(\.[^/]+)?$/')
if [[ ! "$libcucim_files" =~ ^cucim/clara/libcucim\.so\.[0-9]+$ ]]; then
    printf 'Expected one libcucim shared library under cucim/clara. Found\n%s\n' "$libcucim_files" >&2
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
