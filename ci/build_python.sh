#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="cucim"
package_dir="python/cucim"
package_src_dir="${package_dir}/src/${package_name}"

version=$(rapids-generate-version)

commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_src_dir}/_version.py"

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cucim

rapids-upload-conda-to-s3 python
