#!/bin/bash
# Wrapper script to run tests with locally built cuCIM

# Set paths to use local build instead of PyPI package
# Need BOTH the Python source AND the compiled C++ extension
export PYTHONPATH=/home/cdinea/Downloads/cucim_pr2/cucim/python/cucim/src:/home/cdinea/Downloads/cucim_pr2/cucim/python/install/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUCIM_CONFIG_PATH=/tmp/.cucim_aperio_test.json

echo "🔧 Using locally built cuCIM:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Run the test
python "$@"

