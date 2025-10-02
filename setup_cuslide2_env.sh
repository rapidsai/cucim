#!/bin/bash
# Setup script for building cuslide2 plugin
# This script creates a conda environment with all necessary dependencies

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="cuslide2-build"
ENV_FILE="${SCRIPT_DIR}/cuslide2_build_env.yaml"

echo "=========================================="
echo "cuslide2 Build Environment Setup"
echo "=========================================="
echo ""

# Check if conda/mamba/micromamba is available
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
    echo "✓ Using micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Using mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Using conda"
else
    echo "✗ Error: No conda, mamba, or micromamba found!"
    echo "Please install one of these package managers first."
    exit 1
fi

echo ""
echo "Environment file: ${ENV_FILE}"
echo "Environment name: ${ENV_NAME}"
echo ""

# Check if environment already exists
if ${CONDA_CMD} env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        ${CONDA_CMD} env remove -n ${ENV_NAME} -y
    else
        echo "Updating existing environment..."
        ${CONDA_CMD} env update -n ${ENV_NAME} -f ${ENV_FILE}
        echo ""
        echo "=========================================="
        echo "✓ Environment updated successfully!"
        echo "=========================================="
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate ${ENV_NAME}"
        echo ""
        exit 0
    fi
fi

# Create the environment
echo "Creating conda environment '${ENV_NAME}'..."
echo "This may take several minutes..."
echo ""

${CONDA_CMD} env create -f ${ENV_FILE}

echo ""
echo "=========================================="
echo "✓ Environment created successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To build cuslide2 plugin, follow these steps:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Navigate to the cucim build directory:"
echo "   cd ${SCRIPT_DIR}/branchremote"
echo ""
echo "3. Configure and build the project:"
echo "   mkdir -p build && cd build"
echo "   cmake ../cucim -DCMAKE_BUILD_TYPE=Release"
echo "   make -j\$(nproc)"
echo ""
echo "4. Or build cuslide2 plugin directly:"
echo "   cd ${SCRIPT_DIR}/cpp/plugins/cucim.kit.cuslide2"
echo "   mkdir -p build-release && cd build-release"
echo "   cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "   make -j\$(nproc)"
echo ""
echo "For more details, see:"
echo "  - CMakeLists.txt in cuslide2 directory"
echo "  - CONTRIBUTING.md in cucim directory"
echo ""

