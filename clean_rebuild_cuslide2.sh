#!/bin/bash
# Clean rebuild script for cuslide2 plugin in cucim-test environment

set -e  # Exit on any error

echo "============================================================"
echo " Clean Rebuild of cuslide2 Plugin"
echo "============================================================"
echo ""

# Step 1: Check environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ ERROR: No conda/micromamba environment activated!"
    echo ""
    echo "Please activate the environment first:"
    echo "  micromamba activate cucim-test"
    echo ""
    exit 1
fi

echo "✅ Environment: $CONDA_PREFIX"
echo ""

# Step 2: Go to project root
cd /home/cdinea/Downloads/cucim_pr2/cucim
echo "✅ Working directory: $(pwd)"
echo ""

# Step 3: Clean ALL old build artifacts
echo "============================================================"
echo " Cleaning old build artifacts..."
echo "============================================================"

# Remove main build directories
if [ -d "build-release" ]; then
    echo "  Removing build-release/"
    rm -rf build-release
fi

if [ -d "build" ]; then
    echo "  Removing build/"
    rm -rf build
fi

if [ -d "install" ]; then
    echo "  Removing install/"
    rm -rf install
fi

# Remove Python build directories
if [ -d "python/build-release" ]; then
    echo "  Removing python/build-release/"
    rm -rf python/build-release
fi

if [ -d "python/build" ]; then
    echo "  Removing python/build/"
    rm -rf python/build
fi

if [ -d "python/install" ]; then
    echo "  Removing python/install/"
    rm -rf python/install
fi

echo "✅ All build artifacts cleaned"
echo ""

# Step 4: Verify the code change is present
echo "============================================================"
echo " Verifying code changes..."
echo "============================================================"

if grep -q "FORCED num_workers=0" cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp; then
    echo "✅ Code change found: 'FORCED num_workers=0' is in ifd.cpp"
else
    echo "⚠️  WARNING: Could not find 'FORCED num_workers=0' in ifd.cpp"
    echo "   The synchronous mode may not be enabled"
fi
echo ""

# Step 5: Set build environment variables
echo "============================================================"
echo " Setting build environment variables..."
echo "============================================================"

export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  CUDACXX=$CUDACXX"
echo ""

# Step 6: Build
echo "============================================================"
echo " Building cuCIM with cuslide2 plugin..."
echo "============================================================"
echo ""
echo "This will take several minutes..."
echo ""

./run build_local all release $CONDA_PREFIX

echo ""
echo "============================================================"
echo " Build Complete!"
echo "============================================================"
echo ""

# Step 7: Verify plugin was built
echo "Checking for plugin library..."
PLUGIN_LIB=$(find install/lib -name "*cuslide2*.so" 2>/dev/null | head -1)

if [ -n "$PLUGIN_LIB" ]; then
    echo "✅ Plugin library found:"
    ls -lh "$PLUGIN_LIB"
else
    echo "❌ Plugin library not found!"
    echo "   Expected location: install/lib/"
    exit 1
fi

echo ""

# Step 8: Verify Python extension was built
echo "Checking for Python extension..."
PYTHON_EXT=$(find python/install -name "_cucim*.so" 2>/dev/null | head -1)

if [ -n "$PYTHON_EXT" ]; then
    echo "✅ Python extension found:"
    ls -lh "$PYTHON_EXT"
else
    echo "❌ Python extension not found!"
    echo "   Expected location: python/install/"
    exit 1
fi

echo ""
echo "============================================================"
echo " ✅ Rebuild Successful!"
echo "============================================================"
echo ""
echo "Next step: Test the plugin"
echo "  ./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs"
echo ""
echo "Look for this in the output:"
echo "  ⚠️  FORCED num_workers=0 for synchronous execution (debugging)"
echo "  📍 location_len=1, batch_size=1, num_workers=0"
echo "                                              ^"
echo "                                              Must be 0, not 1!"
echo ""

