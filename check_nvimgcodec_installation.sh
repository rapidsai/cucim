#!/bin/bash
# Quick diagnostic script to check nvImageCodec installation and available backends

echo "========================================================================"
echo "nvImageCodec Installation & Backend Check"
echo "========================================================================"

echo ""
echo "📍 Step 1: Checking nvImageCodec library files..."
echo "----------------------------------------"

CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-12/lib64"
    "/usr/local/cuda-11/lib64"
    "$HOME/.local/lib"
)

FOUND_NVIMGCODEC=0
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "Checking: $path"
        libs=$(ls "$path"/libnvimgcodec* 2>/dev/null)
        if [ -n "$libs" ]; then
            echo "✅ Found nvImageCodec:"
            echo "$libs" | sed 's/^/   /'
            FOUND_NVIMGCODEC=1
        fi
    fi
done

if [ $FOUND_NVIMGCODEC -eq 0 ]; then
    echo "⚠️  nvImageCodec library not found in standard locations"
fi

echo ""
echo "📍 Step 2: Checking for CPU decoder backends (libjpeg-turbo)..."
echo "----------------------------------------"

FOUND_LIBJPEG=0
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        libs=$(ls "$path"/libjpeg-turbo* "$path"/libjpeg.so* "$path"/libturbojpeg* 2>/dev/null)
        if [ -n "$libs" ]; then
            echo "✅ Found libjpeg-turbo:"
            echo "$libs" | sed 's/^/   /'
            FOUND_LIBJPEG=1
        fi
    fi
done

if [ $FOUND_LIBJPEG -eq 0 ]; then
    echo "⚠️  libjpeg-turbo not found - CPU JPEG decoding not available"
fi

echo ""
echo "📍 Step 3: Checking nvImageCodec extension modules..."
echo "----------------------------------------"

EXT_PATHS=(
    "/usr/local/cuda/lib64/nvimgcodec_extensions"
    "/usr/local/cuda-12/lib64/nvimgcodec_extensions"
    "$HOME/.local/lib/nvimgcodec_extensions"
)

FOUND_EXTENSIONS=0
for path in "${EXT_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "Checking: $path"
        exts=$(ls "$path"/*.so 2>/dev/null)
        if [ -n "$exts" ]; then
            echo "✅ Found extensions:"
            echo "$exts" | sed 's/^/   /'
            FOUND_EXTENSIONS=1
        fi
    fi
done

if [ $FOUND_EXTENSIONS -eq 0 ]; then
    echo "⚠️  No extension modules found"
fi

echo ""
echo "📍 Step 4: Checking nvImageCodec Python package..."
echo "----------------------------------------"

if command -v python3 &> /dev/null; then
    python3 -c "import nvidia.nvimgcodec; print('✅ nvImageCodec Python:', nvidia.nvimgcodec.__version__)" 2>/dev/null || \
    echo "⚠️  nvImageCodec Python package not installed"
else
    echo "⚠️  Python3 not found"
fi

echo ""
echo "========================================================================"
echo "Summary & Recommendations"
echo "========================================================================"

if [ $FOUND_NVIMGCODEC -eq 1 ]; then
    echo "✅ nvImageCodec is installed"
else
    echo "❌ nvImageCodec is NOT installed or not in standard location"
fi

if [ $FOUND_LIBJPEG -eq 0 ]; then
    echo ""
    echo "❌ CPU JPEG backend (libjpeg-turbo) NOT available"
    echo ""
    echo "This explains why CPU decoding fails with status=3"
    echo "(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED)"
    echo ""
    echo "Your options:"
    echo "1. ✅ Continue using GPU + CPU copy fallback (current solution)"
    echo "2. Install libjpeg-turbo and rebuild nvImageCodec with CPU support"
    echo "3. Wait for nvImageCodec 0.7.0 which may have better CPU backend support"
    echo ""
    echo "To install libjpeg-turbo:"
    echo "   Ubuntu/Debian: sudo apt-get install libjpeg-turbo8-dev"
    echo "   RHEL/CentOS:   sudo yum install libjpeg-turbo-devel"
    echo "   From source:   https://github.com/libjpeg-turbo/libjpeg-turbo"
fi

echo ""
echo "🔍 To see detailed backend loading at runtime:"
echo "   export NVIMGCODEC_DEBUG=1"
echo "   # Then run your application"
echo ""
echo "========================================================================"

