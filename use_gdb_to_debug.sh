#!/bin/bash
# Script to debug the segfault with GDB

echo "🔍 Setting up GDB debugging session..."
echo ""

# Enable core dumps
ulimit -c unlimited
echo "✅ Core dumps enabled"

# Set environment (match run_test_with_local_build.sh)
export PYTHONPATH=/home/cdinea/Downloads/cucim_pr2/cucim/python/cucim/src:$PYTHONPATH
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUCIM_CONFIG_PATH=/tmp/.cucim_aperio_test.json

echo ""
echo "🐛 Starting GDB session..."
echo "   When GDB starts, type 'run' and press Enter"
echo "   When it crashes, type 'bt' to see the backtrace"
echo "   Type 'thread apply all bt' to see all threads"
echo ""

# Create GDB commands file
cat > /tmp/gdb_commands.txt << 'EOF'
# GDB commands to run automatically
set pagination off
set print pretty on
handle SIGTERM nostop noprint
handle SIGPIPE nostop noprint
run
bt
thread apply all bt full
info threads
quit
EOF

# Run with GDB
gdb -batch -x /tmp/gdb_commands.txt --args python test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs

echo ""
echo "📊 GDB session completed. Check output above for stack trace."

