#!/bin/bash
# Remove CPU decoder directories from cuslide2 (pure nvImageCodec implementation)

set -e

CUSLIDE2_DIR="/home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/src/cuslide"

echo "🧹 Cleaning up CPU decoder directories from cuslide2..."
echo "   Target: $CUSLIDE2_DIR"
echo ""

# List of directories to remove
REMOVE_DIRS=(
    "deflate"
    "jpeg"
    "jpeg2k"
    "loader"
    "lzw"
    "raw"
)

echo "📋 Directories to remove:"
for dir in "${REMOVE_DIRS[@]}"; do
    if [ -d "$CUSLIDE2_DIR/$dir" ]; then
        echo "   ❌ $dir/ (CPU decoder - not needed with nvImageCodec)"
    else
        echo "   ⏭️  $dir/ (already removed)"
    fi
done

echo ""
echo "📋 Directories to KEEP:"
echo "   ✅ cuslide.cpp/h (plugin interface)"
echo "   ✅ nvimgcodec/ (GPU-accelerated decoding)"
echo "   ✅ tiff/ (high-level orchestration)"
echo ""

read -p "❓ Remove CPU decoder directories? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🗑️  Removing CPU decoder directories..."
    
    for dir in "${REMOVE_DIRS[@]}"; do
        if [ -d "$CUSLIDE2_DIR/$dir" ]; then
            echo "   Removing $dir/..."
            rm -rf "$CUSLIDE2_DIR/$dir"
        fi
    done
    
    echo ""
    echo "✅ Cleanup complete!"
    echo ""
    echo "📁 Remaining structure:"
    ls -la "$CUSLIDE2_DIR"
    
    echo ""
    echo "🔨 Next steps:"
    echo "   1. Rebuild cuslide2: ./fast_rebuild_plugin.sh"
    echo "   2. Test: ./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs"
else
    echo ""
    echo "❌ Cancelled. No files removed."
fi

