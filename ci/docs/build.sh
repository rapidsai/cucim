#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#################################
# cuCIM Docs build script for CI #
#################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export DOCS_WORKSPACE=$WORKSPACE/docs
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export HOME=$WORKSPACE
export PROJECT_WORKSPACE=/rapids/cucim
export NIGHTLY_VERSION=$(echo $BRANCH_VERSION | awk -F. '{print $2}')
export PROJECTS=(cucim)

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh

conda install -c conda-forge conda-build -y

# We don't use 'rapids' conda environment here.
# cuCIM depends on CuPy 9 which should be adopted in 0.20
LIBCUCIM_BLD_PATH=$WORKSPACE/ci/artifacts/cucim/cpu/.conda-bld
CUCIM_BLD_PATH=/opt/conda/envs/rapids/conda-bld
mkdir -p ${CUCIM_BLD_PATH}

gpuci_conda_retry build -c ${LIBCUCIM_BLD_PATH} -c conda-forge/label/cupy_rc -c conda-forge -c rapidsai-nightly \
    --python=${PYTHON_VER} \
    --dirty \
    --no-remove-work-dir \
    --croot ${CUCIM_BLD_PATH} \
    conda/recipes/cucim


# TODO: Move installs to docs-build-env meta package
gpuci_conda_retry create -n cucim -y -c conda-forge -c conda-forge/label/cupy_rc -c rapidsai-nightly \
    rapids-doc-env \
    flake8 \
    pytest \
    pytest-cov \
    python=${PYTHON_VER} \
    conda-forge/label/cupy_rc::cupy=9 \
    cudatoolkit=${CUDA_VER} \
    numpy \
    scipy \
    scikit-image=0.18.1 \
    openslide

conda activate cucim

gpuci_logger "Installing cuCIM"
gpuci_conda_retry install -y -c ${LIBCUCIM_BLD_PATH} -c ${CUCIM_BLD_PATH} \
    libcucim \
    cucim

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# Build Python docs
gpuci_logger "Build Sphinx docs"
cd $PROJECT_WORKSPACE/docs
make html

#Commit to Website
cd $DOCS_WORKSPACE

for PROJECT in ${PROJECTS[@]}; do
    if [ ! -d "api/$PROJECT/$BRANCH_VERSION" ]; then
        mkdir -p api/$PROJECT/$BRANCH_VERSION
    fi
    rm -rf $DOCS_WORKSPACE/api/$PROJECT/$BRANCH_VERSION/*	
done

mv $PROJECT_WORKSPACE/docs/build/html/* $DOCS_WORKSPACE/api/cucim/$BRANCH_VERSION
