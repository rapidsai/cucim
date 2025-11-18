#!/bin/bash
# Download Philips TIFF test data from OpenSlide

set -e

DOWNLOAD_DIR="${1:-/tmp/philips-tiff-testdata}"

echo "========================================"
echo "📥 Downloading Philips TIFF Test Data"
echo "========================================"
echo ""
echo "Download directory: ${DOWNLOAD_DIR}"
echo ""

mkdir -p "${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

echo "🌐 Fetching test data from OpenSlide..."
echo ""

# Download the directory listing first
wget -q https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/ -O index.html

# Extract .tiff file links
TIFF_FILES=$(grep -oP 'href="\K[^"]*\.tiff' index.html | head -5)

if [ -z "$TIFF_FILES" ]; then
    echo "⚠️  No .tiff files found in directory listing"
    echo "Trying alternative approach..."
    
    # Try downloading specific known files
    echo "📥 Attempting to download sample files..."
    
    # These are example URLs - actual files may vary
    FILES=(
        "sample_001.tiff"
        "test_philips_001.tiff"
    )
    
    for file in "${FILES[@]}"; do
        URL="https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/${file}"
        echo "  Trying: ${file}..."
        if wget -q --spider "${URL}" 2>/dev/null; then
            wget -q --show-progress "${URL}" -O "${file}"
            echo "    ✅ Downloaded: ${file}"
        else
            echo "    ⚠️  Not found: ${file}"
        fi
    done
else
    # Download each TIFF file
    for file in $TIFF_FILES; do
        if [ ! -f "${file}" ]; then
            echo "📥 Downloading: ${file}..."
            wget -q --show-progress "https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/${file}"
            echo "  ✅ Downloaded: ${file}"
        else
            echo "  ⏭️  Already exists: ${file}"
        fi
    done
fi

rm -f index.html

echo ""
echo "✅ Download complete!"
echo ""
echo "📂 Downloaded files:"
ls -lh *.tiff 2>/dev/null || echo "  ⚠️  No .tiff files downloaded"
echo ""
echo "🧪 To test with cuslide2, run:"
echo "  cd /home/cdinea/Downloads/cucim_pr2/cucim"
echo "  python test_philips_tiff.py ${DOWNLOAD_DIR}/[file].tiff"
echo ""

