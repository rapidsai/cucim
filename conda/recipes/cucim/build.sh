# Copyright (c) 2021, NVIDIA CORPORATION.

CUCIM_BUILD_TYPE=${CUCIM_BUILD_TYPE:-release}

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"
echo "CUDAHOSTCXX : ${CUDAHOSTCXX}"

# For now CUDAHOSTCXX is set to `/usr/bin/g++` by
# https://github.com/rapidsai/docker/blob/161b200157206660d88fb02cf69fe58d363ac95e/generated-dockerfiles/rapidsai-core_ubuntu18.04-devel.Dockerfile
# To use GCC-9 in conda build environment, need to set it to $CXX (=$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++)
# This can be removed once we switch to use gcc-9
# : https://docs.rapids.ai/notices/rdn0002/
export CUDAHOSTCXX=${CXX}

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)
./run build_local cucim ${CUCIM_BUILD_TYPE}

cp -P python/install/lib/* python/cucim/src/cucim/clara/

pushd python/cucim

echo "PYTHON: ${PYTHON}"
$PYTHON setup.py install

popd
