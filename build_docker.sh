#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -eu

CMAKE_CMD=cmake
CMAKE_BUILD_TYPE=Release
NUM_THREADS=$(nproc)

SRC_ROOT=/work
BUILD_ROOT=/work/temp

CUCIM_SDK_PATH=${BUILD_ROOT}/libcucim

# Build libcucim
${CMAKE_CMD} -S ${SRC_ROOT} -B ${BUILD_ROOT}/libcucim \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${CUCIM_SDK_PATH}
${CMAKE_CMD} --build ${BUILD_ROOT}/libcucim --target cucim -- -j ${NUM_THREADS}
${CMAKE_CMD} --build ${BUILD_ROOT}/libcucim --target install -- -j ${NUM_THREADS}

# Build cuslide plugin
${CMAKE_CMD} -S ${SRC_ROOT}/cpp/plugins/cucim.kit.cuslide -B ${BUILD_ROOT}/cuslide \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${BUILD_ROOT}/cuslide/install \
    -DCUCIM_SDK_PATH=${CUCIM_SDK_PATH}
${CMAKE_CMD} --build ${BUILD_ROOT}/cuslide --target cucim.kit.cuslide -- -j ${NUM_THREADS}
${CMAKE_CMD} --build ${BUILD_ROOT}/cuslide --target install -- -j ${NUM_THREADS}

# Build Python bind

for PYBIN in /opt/python/*/bin; do
    ${CMAKE_CMD} -S ${SRC_ROOT}/python -B ${BUILD_ROOT}/cucim \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${BUILD_ROOT}/cucim/install \
        -DCUCIM_SDK_PATH=${CUCIM_SDK_PATH} \
        -DPYTHON_EXECUTABLE=${PYBIN}/python
    ${CMAKE_CMD} --build ${BUILD_ROOT}/cucim --target cucim -- -j ${NUM_THREADS}
    ${CMAKE_CMD} --build ${BUILD_ROOT}/cucim --target install -- -j ${NUM_THREADS}
done

# Copy .so files to pybind's build folder
# (it uses -P to copy symbolic links as they are)
cp -P ${BUILD_ROOT}/libcucim/install/lib/lib* ${BUILD_ROOT}/cucim/lib/cucim/
cp -P ${BUILD_ROOT}/cuslide/install/lib/cucim* ${BUILD_ROOT}/cucim/lib/cucim/

# Copy .so files from pybind's build folder to cucim Python source folder
cp -P ${BUILD_ROOT}/cucim/lib/cucim/* ${SRC_ROOT}/python/cucim/src/cucim/clara/




set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair --plat "$PLAT" -w wheelhouse/ "$wheel"
    fi
}

PLAT=manylinux2014_x86_64

cd /work/python/cucim
# Compile wheels (one python binary is enough)
for PYBIN in /opt/python/cp36-cp36m/bin; do # /opt/python/*/bin
    "${PYBIN}/python" setup.py bdist_wheel -p $PLAT
done

mkdir -p /work/python/cucim/wheelhouse

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    repair_wheel "$whl"
done

# # Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done

# python setup.py bdist_wheel -p manylinux2014-x86_64
