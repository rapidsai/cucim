#!/bin/bash
# Build and run the nvImageCodec sub-stream crash reproducer

set -e

echo "=== Building nvImageCodec Sub-Stream Crash Reproducer ==="

# Check for CONDA_PREFIX
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX not set. Please activate your conda environment."
    exit 1
fi

# Check for nvimgcodec header
if [ ! -f "$CONDA_PREFIX/include/nvimgcodec.h" ]; then
    echo "Error: nvimgcodec.h not found in $CONDA_PREFIX/include"
    echo "Please install nvImageCodec: conda install libnvimgcodec-dev -c conda-forge"
    exit 1
fi

# Build
echo "Compiling..."
g++ -std=c++17 -O2 -o nvimgcodec_substream_crash_reproducer \
    nvimgcodec_substream_crash_reproducer.cpp \
    -I$CONDA_PREFIX/include \
    -I$CUDA_HOME/include \
    -L$CONDA_PREFIX/lib \
    -L$CUDA_HOME/lib64 \
    -lnvimgcodec \
    -lcudart \
    -Wl,-rpath,$CONDA_PREFIX/lib \
    -Wl,-rpath,$CUDA_HOME/lib64

echo "Build successful!"
echo ""

# Check for test file
TEST_FILE="/tmp/CMU-1-Small-Region.svs"
if [ ! -f "$TEST_FILE" ]; then
    echo "Downloading test file..."
    wget -q -O "$TEST_FILE" \
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
fi

echo "Running reproducer..."
echo "=========================================="
./nvimgcodec_substream_crash_reproducer "$TEST_FILE"
echo "=========================================="
echo ""
echo "If you see 'Test completed successfully', there was no crash."
echo "If you see 'free(): invalid pointer', the bug is confirmed."

