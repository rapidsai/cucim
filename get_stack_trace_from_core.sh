#!/bin/bash
# Get stack trace from core dump

set +e  # Don't exit on error

echo "🔍 Enabling core dumps and running test..."
echo ""

# Enable core dumps
ulimit -c unlimited
echo "✅ Core dumps enabled (ulimit -c: $(ulimit -c))"

# Set core dump location
sudo sysctl -w kernel.core_pattern=/tmp/core.%e.%p 2>/dev/null || true
echo "✅ Core dumps will be saved to: /tmp/core.*"
echo ""

# Remove old core dumps
rm -f /tmp/core.python.* 2>/dev/null

# Run the test (will create core dump on crash)
echo "🚀 Running test (this will crash)..."
./run_test_with_local_build.sh test_aperio_svs.py /tmp/CMU-1-JP2K-33005.svs
TEST_EXIT=$?

echo ""
echo "📊 Test exited with code: $TEST_EXIT"
echo ""

# Find the core dump
CORE_FILE=$(ls -t /tmp/core.python.* 2>/dev/null | head -1)

if [ -z "$CORE_FILE" ]; then
    echo "❌ No core dump found!"
    echo "   Core dumps may be disabled system-wide."
    echo "   Try: sudo sysctl -w kernel.core_pattern=/tmp/core.%e.%p"
    echo "   Or check: cat /proc/sys/kernel/core_pattern"
    exit 1
fi

echo "✅ Core dump found: $CORE_FILE"
echo ""

# Get stack trace from core
echo "🔍 Extracting stack trace..."
echo ""

gdb -batch \
    -ex "set pagination off" \
    -ex "thread apply all bt" \
    -ex "info threads" \
    -ex "quit" \
    python "$CORE_FILE" 2>&1 | tee /tmp/stacktrace.txt

echo ""
echo "✅ Stack trace saved to: /tmp/stacktrace.txt"
echo ""
echo "🔍 Key information:"
grep -A 5 "Thread.*SIGSEGV" /tmp/stacktrace.txt || echo "  (No SIGSEGV marker found)"

