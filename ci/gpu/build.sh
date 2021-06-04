#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# ucx-py GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# apt-get install libnuma libnuma-dev

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Get CUDA and Python version
export CUDA_VERSION=${CUDA_VERSION:-$(cat /usr/local/cuda/version.txt | egrep -o "[[:digit:]]+.[[:digit:]]+.[[:digit:]]+")}
export CUDA_VER=${CUDA_VERSION%.*}
export PYTHON_VER=${PYTHON:-$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")}
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "CUDA_VER    : ${CUDA_VER}"
echo "PYTHON_VER  : ${PYTHON_VER}"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install dependencies"
gpuci_conda_retry install -y \
    sysroot_linux-64=2.17 \
    conda-build \
    "cudatoolkit=${CUDA_VER}.*" \
    "rapids-build-env=$MINOR_VERSION.*" \
    "python=${PYTHON_VER}.*" \

################################################################################
# BUILD - Build cuCIM
################################################################################

# We don't use 'rapids' conda environment here.
# To use 'conda-forge'-based package installation and to use 'Project Flash' feature,
# we fake conda build folder for libcucim to '$WORKSPACE/ci/artifacts/cucim/cpu/conda-bld/' which is
# conda build folder for CPU build.
# For GPU build, we fake conda build folder for cucim to '/opt/conda/envs/rapids/conda-bld'.
LIBCUCIM_BLD_PATH=$WORKSPACE/ci/artifacts/cucim/cpu/.conda-bld
CUCIM_BLD_PATH=/opt/conda/envs/rapids/conda-bld
mkdir -p ${CUCIM_BLD_PATH}


gpuci_conda_retry build -c ${LIBCUCIM_BLD_PATH} -c conda-forge -c rapidsai-nightly \
    --dirty \
    --no-remove-work-dir \
    --croot ${CUCIM_BLD_PATH} \
    conda/recipes/cucim


################################################################################
# TEST - Run py.tests for cuCIM
################################################################################

# Install cuCIM and its dependencies
gpuci_logger "Installing cuCIM"
gpuci_conda_retry install -y -c ${LIBCUCIM_BLD_PATH} -c ${CUCIM_BLD_PATH} \
    "libcucim=$MINOR_VERSION.*" \
    "cucim=$MINOR_VERSION.*"

gpuci_logger "Testing cuCIM import"
python -c 'import cucim' 

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Check NICs"
    awk 'END{print $1}' /etc/hosts
    cat /etc/hosts

    gpuci_logger "Python py.test for cuCIM"
    cd $WORKSPACE/python/cucim
    py.test --cache-clear -v --ignore-glob . --rootdir=src
fi
