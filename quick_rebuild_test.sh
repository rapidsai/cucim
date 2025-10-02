#!/bin/bash
set -e

echo "🔨 Quick Rebuild and Test Script"
echo "=================================="

# Navigate to project root
cd /home/cdinea/Downloads/cucim_pr2/cucim

# Rebuild C++ library only (faster than full rebuild)
echo ""
echo "📦 Rebuilding C++ library..."
cd build-release
make cucim -j$(nproc)

echo ""
echo "📦 Rebuilding cuslide2 plugin..."
cd ../cpp/plugins/cucim.kit.cuslide2/build-release
make -j$(nproc)

# Go back to project root
cd /home/cdinea/Downloads/cucim_pr2/cucim

echo ""
echo "✅ Build complete!"
echo ""
echo "🧪 Running test..."
echo "===================="
./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs 2>&1 | tee test_output_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "📋 Test complete. Output saved to test_output_*.log"

