#!/bin/bash
# Fast rebuild script - only rebuilds cuslide2 plugin (not full cucim)

set -e

echo "🔨 Fast rebuild of cuslide2 plugin only"
echo ""

cd /home/cdinea/Downloads/cucim_pr2/cucim

# Touch the file to force recompilation
touch cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp
echo "✅ Touched ifd.cpp"

# Rebuild just the plugin
cd build-release
make cucim.kit.cuslide2 -j$(nproc)

echo ""
echo "✅ Plugin rebuilt!"
echo ""
echo "Now run: ./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs"

