#!/bin/bash
# Rebuild after disabling OpenJPEG patch

set -e

echo "============================================================"
echo " Rebuild cuCIM (OpenJPEG patch disabled)"
echo "============================================================"
echo ""

cd /home/cdinea/Downloads/cucim_pr2/cucim

# Check environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ ERROR: No conda/micromamba environment activated!"
    echo "Please run: micromamba activate cucim-test"
    exit 1
fi

echo "✅ Environment: $CONDA_PREFIX"
echo "✅ OpenJPEG patch disabled (not needed)"
echo "✅ libtiff: building from source with sed patch (provides internal headers)"
echo ""

# Clean everything
echo "============================================================"
echo " Cleaning build artifacts..."
echo "============================================================"

rm -rf build-release build install
rm -rf python/build-release python/build python/install

echo "✅ Cleaned"
echo ""

# Set environment
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

# Build
echo "============================================================"
echo " Building cuCIM..."
echo "============================================================"
echo ""

./run build_local all release $CONDA_PREFIX

BUILD_STATUS=$?

echo ""
if [ $BUILD_STATUS -eq 0 ]; then
    echo "============================================================"
    echo " ✅ Build Successful!"
    echo "============================================================"
    echo ""
    
    # Verify
    PLUGIN=$(find install/lib -name "*cuslide2*.so" 2>/dev/null | head -1)
    if [ -n "$PLUGIN" ]; then
        echo "✅ Plugin: $PLUGIN"
        ls -lh "$PLUGIN"
    fi
    
    PYTHON_EXT=$(find python/install -name "_cucim*.so" 2>/dev/null | head -1)
    if [ -n "$PYTHON_EXT" ]; then
        echo "✅ Python extension: $PYTHON_EXT"
        ls -lh "$PYTHON_EXT"
    fi
    
    echo ""
    echo "Next: Test the plugin"
    echo "  ./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs"
    echo ""
    echo "Look for in output:"
    echo "  ⚠️  FORCED num_workers=0 for synchronous execution (debugging)"
    echo "  📍 location_len=1, batch_size=1, num_workers=0"
    echo ""
else
    echo "❌ Build failed! Check errors above."
    exit 1
fi

