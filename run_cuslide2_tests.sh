#!/bin/bash
# Script to build and run cuslide2 tests and benchmarks

set -e

BUILD_DIR="/home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release"
INSTALL_DIR="/home/cdinea/Downloads/cucim_pr2/cucim/install"
TEST_DATA_DIR="${TEST_DATA_DIR:-/tmp}"

echo "========================================"
echo "🧪 cuslide2 Test & Benchmark Suite"
echo "========================================"
echo ""

# Set library paths
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
export CUCIM_PLUGIN_PATH="${BUILD_DIR}/lib"

# Build tests
echo "🔨 Building tests..."
cd "${BUILD_DIR}/tests"
make -j$(nproc) cuslide_tests
echo "✅ Tests built successfully"
echo ""

# Build benchmarks
echo "🔨 Building benchmarks..."
cd "${BUILD_DIR}/benchmarks"
make -j$(nproc) cuslide_benchmarks
echo "✅ Benchmarks built successfully"
echo ""

# Run tests
echo "========================================"
echo "🧪 Running Tests"
echo "========================================"
cd "${BUILD_DIR}/tests"

if [ -f "./cuslide_tests" ]; then
    echo ""
    echo "ℹ️  Available test images: ${TEST_DATA_DIR}"
    echo "ℹ️  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo "ℹ️  CUCIM_PLUGIN_PATH: ${CUCIM_PLUGIN_PATH}"
    echo ""
    
    # Run tests with optional file path argument
    if [ -n "$1" ]; then
        echo "🚀 Running tests with file: $1"
        ./cuslide_tests "$1"
    else
        echo "🚀 Running tests (use default or discovery mode)"
        ./cuslide_tests
    fi
else
    echo "❌ Test executable not found!"
    exit 1
fi

echo ""
echo "========================================"
echo "📊 Running Benchmarks"
echo "========================================"
cd "${BUILD_DIR}/benchmarks"

if [ -f "./cuslide_benchmarks" ]; then
    echo ""
    echo "ℹ️  Available test images: ${TEST_DATA_DIR}"
    echo ""
    
    # Run benchmarks with optional file path argument
    if [ -n "$2" ]; then
        echo "🚀 Running benchmarks with file: $2"
        ./cuslide_benchmarks "$2"
    elif [ -n "$1" ]; then
        echo "🚀 Running benchmarks with file: $1"
        ./cuslide_benchmarks "$1"
    else
        echo "🚀 Running benchmarks (use default or discovery mode)"
        ./cuslide_benchmarks
    fi
else
    echo "❌ Benchmark executable not found!"
    exit 1
fi

echo ""
echo "✅ All tests and benchmarks completed!"

