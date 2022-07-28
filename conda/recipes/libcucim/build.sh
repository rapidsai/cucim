# Copyright (c) 2021, NVIDIA CORPORATION.

CUCIM_BUILD_TYPE=${CUCIM_BUILD_TYPE:-release}

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"
echo "CUDA        : ${CUDA}"

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)

# Build libcucim core
./run build_local libcucim ${CUCIM_BUILD_TYPE} ${PREFIX}

mkdir -p $PREFIX/bin $PREFIX/lib $PREFIX/include
cp -P -r install/bin/* $PREFIX/bin/ || true
cp -P -r install/lib/* $PREFIX/lib/ || true
cp -P -r install/include/* $PREFIX/include/ || true

# Build plugins
for plugin_name in cuslide cumed; do
    echo "Building cucim.kit.${plugin_name} ..."
    ./run build_local ${plugin_name} ${CUCIM_BUILD_TYPE} ${PREFIX}
    mkdir -p $PREFIX/bin $PREFIX/lib $PREFIX/include
    cp -P -r cpp/plugins/cucim.kit.${plugin_name}/install/bin/* $PREFIX/bin/ || true
    cp -P -r cpp/plugins/cucim.kit.${plugin_name}/install/lib/* $PREFIX/lib/ || true
done
