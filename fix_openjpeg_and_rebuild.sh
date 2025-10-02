#!/bin/bash
# Fix OpenJPEG patch error and rebuild

set -e

echo "============================================================"
echo " Fixing OpenJPEG Patch Error"
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
echo ""

# Step 1: Remove ALL build directories (including the problematic _deps)
echo "============================================================"
echo " Step 1: Cleaning ALL build artifacts..."
echo "============================================================"

# Remove main build directories
for dir in build-release build install; do
    if [ -d "$dir" ]; then
        echo "  Removing $dir/"
        rm -rf "$dir"
    fi
done

# Remove Python build directories  
for dir in python/build-release python/build python/install; do
    if [ -d "$dir" ]; then
        echo "  Removing $dir/"
        rm -rf "$dir"
    fi
done

# IMPORTANT: Remove the _deps cache where OpenJPEG source is stored
if [ -d "build-release/_deps" ]; then
    echo "  Removing build-release/_deps/ (contains OpenJPEG source)"
    rm -rf build-release/_deps
fi

echo "✅ All build artifacts cleaned"
echo ""

# Step 2: Set environment variables
echo "============================================================"
echo " Step 2: Setting build environment..."
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

# Step 3: Build with verbose output to see what's happening
echo "============================================================"
echo " Step 3: Building cuCIM..."
echo "============================================================"
echo ""
echo "This will take several minutes. Building with verbose output..."
echo ""

# Use the run script with clean flag
./run build_local all release $CONDA_PREFIX

BUILD_STATUS=$?

echo ""
echo "============================================================"

if [ $BUILD_STATUS -eq 0 ]; then
    echo " ✅ Build Successful!"
    echo "============================================================"
    echo ""
    
    # Verify plugin
    PLUGIN_LIB=$(find install/lib -name "*cuslide2*.so" 2>/dev/null | head -1)
    if [ -n "$PLUGIN_LIB" ]; then
        echo "✅ Plugin library: $PLUGIN_LIB"
        ls -lh "$PLUGIN_LIB"
    fi
    
    echo ""
    echo "Next: Run the test"
    echo "  ./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs"
    echo ""
else
    echo " ❌ Build Failed!"
    echo "============================================================"
    echo ""
    echo "Check the error messages above."
    echo "If OpenJPEG still fails, the patch file may need updating."
    exit 1
fi

