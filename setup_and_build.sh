#!/bin/bash
# Setup and build script for cuslide2 plugin with nvImageCodec

set -e  # Exit on error

echo "============================================================"
echo " cuslide2 Plugin Build Script"
echo "============================================================"
echo ""

# Check if conda/micromamba environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ ERROR: No conda/micromamba environment is activated!"
    echo ""
    echo "Please run:"
    echo "  micromamba activate cucim-test"
    echo "  # OR create environment first:"
    echo "  micromamba create -n cucim-test python=3.10"
    echo "  micromamba activate cucim-test"
    echo ""
    exit 1
fi

echo "✓ Environment activated: $CONDA_PREFIX"
echo ""

# Install dependencies
echo "============================================================"
echo " Step 1: Installing dependencies"
echo "============================================================"
micromamba install -y \
  python=3.10 \
  cuda-toolkit \
  c-compiler \
  cxx-compiler \
  openslide \
  yasm \
  cmake \
  ninja \
  -c conda-forge

echo ""
echo "============================================================"
echo " Step 2: Setting environment variables"
echo "============================================================"
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDACXX=$CUDACXX"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

# Verify yasm
if ! which yasm > /dev/null 2>&1; then
    echo "❌ ERROR: yasm not found in PATH!"
    exit 1
fi
echo "✓ yasm: $(which yasm)"
echo ""

# Clean old builds
echo "============================================================"
echo " Step 3: Cleaning old build directories"
echo "============================================================"
cd /home/cdinea/Downloads/cucim_pr2/cucim
if [ -d "build-release" ]; then
    echo "Removing build-release..."
    rm -rf build-release
fi
if [ -d "build" ]; then
    echo "Removing build..."
    rm -rf build
fi
if [ -d "install" ]; then
    echo "Removing install..."
    rm -rf install
fi
if [ -d "cpp/plugins/cucim.kit.cuslide2/build-release" ]; then
    echo "Removing cpp/plugins/cucim.kit.cuslide2/build-release..."
    rm -rf cpp/plugins/cucim.kit.cuslide2/build-release
fi
echo "✓ Build directories cleaned"
echo ""

# Build
echo "============================================================"
echo " Step 4: Building cuslide2 plugin"
echo "============================================================"
./run build_local all release $CONDA_PREFIX

echo ""
echo "============================================================"
echo " Build Complete!"
echo "============================================================"
echo ""

# Verify
if [ -f "cpp/plugins/cucim.kit.cuslide2/build-release/lib/cucim.kit.cuslide2@"*.so ]; then
    echo "✓ Plugin built successfully:"
    ls -lh cpp/plugins/cucim.kit.cuslide2/build-release/lib/cucim.kit.cuslide2*.so
else
    echo "⚠️  Warning: Plugin file not found in expected location"
fi

echo ""
echo "Next steps:"
echo "  1. Run verification: python scripts/verify_cuslide2_infrastructure.py"
echo "  2. Test the plugin with your data"

