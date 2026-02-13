#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

CUCIM_BUILD_TYPE=${CUCIM_BUILD_TYPE:-release}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)
./run build_local cucim "${CUCIM_BUILD_TYPE}"

cp -P python/install/lib/* python/cucim/src/cucim/clara/


PYTHON_ARGS_FOR_INSTALL=(
    --config-settings="rapidsai.disable-cuda=true"
    -vv
)

# If `RAPIDS_PY_VERSION` is set, use that as the lower-bound for the stable ABI CPython version
if [ -n "${RAPIDS_PY_VERSION:-}" ]; then
    RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
    PYTHON_ARGS_FOR_INSTALL+=("--config-settings" "skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

pushd python/cucim

echo "PYTHON: ${PYTHON}"
$PYTHON -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" .

popd
