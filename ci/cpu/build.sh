#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
######################################
# ucx-py CPU conda build script for CI #
######################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# Get CUDA and Python version
export CUDA_VERSION=${CUDA_VERSION:-$(cat /usr/local/cuda/version.txt | egrep -o "[[:digit:]]+.[[:digit:]]+.[[:digit:]]+")}
export CUDA_VER=${CUDA_VERSION%.*}
export PYTHON_VER=${PYTHON:-$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")}
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "CUDA_VER    : ${CUDA_VER}"
echo "PYTHON_VER  : ${PYTHON_VER}"

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh


conda install -c conda-forge conda-build

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds (conda deps: libcucim)
################################################################################

# We get some error message like below if we use 'rapids' conda environment:
#    CMake Warning at build-release/_deps/deps-rmm-src/tests/CMakeLists.txt:52 (add_executable):
#      Cannot generate a safe runtime search path for target LOGGER_PTDS_TEST
#      because files in some directories may conflict with libraries in implicit
#      directories:
#        runtime library [libcudart.so.11.0] in /opt/conda/envs/rapids/conda-bld/libcucim_1616020264601/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_pl/lib may be hidden by files in:
#          /opt/conda/envs/rapids/lib

if [ "$BUILD_LIBCUCIM" == 1 ]; then
  gpuci_conda_retry build -c conda-forge/label/cupy_rc -c conda-forge -c rapidsai-nightly \
    --python=${PYTHON_VER} \
    --dirty \
    --no-remove-work-dir \
    --no-build-id \
    --croot ${CONDA_BLD_DIR} \
    --use-local \
    conda/recipes/libcucim
    # Copy the conda working directory for Project Flash
    mkdir -p ${CONDA_BLD_DIR}/libcucim/work
    cp -r ${CONDA_BLD_DIR}/work/* ${CONDA_BLD_DIR}/libcucim/work
fi

if [ "$BUILD_CUCIM" == 1 ]; then
  gpuci_conda_retry build -c conda-forge/label/cupy_rc -c conda-forge -c rapidsai-nightly \
    --python=${PYTHON_VER} \
    --dirty \
    --no-remove-work-dir \
    --croot ${CONDA_BLD_DIR} \
    --use-local \
    conda/recipes/cucim
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs"
source ci/cpu/upload_anaconda.sh

# gpuci_logger "Upload pypi pkg..."
# source ci/cpu/upload-pypi.sh
