#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="cucim"
package_dir="python/cucim/src"

version=$(rapids-generate-version)
# for CMake VERSION need to truncate any trailing 'a'
version_cpp=${version%a*}

commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
echo "${version_cpp}" > "${package_dir}/VERSION"
echo "${version_cpp}" > VERSION_CPP
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/_version.py"

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cucim

rapids-upload-conda-to-s3 python
