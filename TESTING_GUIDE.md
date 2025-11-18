# cuslide2 Testing & Benchmarking Guide

## 📋 Overview

The `cucim.kit.cuslide2` plugin includes:
- **Unit Tests** using Catch2 framework
- **Benchmarks** using Google Benchmark

## 🏗️ Build Tests & Benchmarks

### Option 1: Build from plugin build directory

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release

# Build tests
make -j$(nproc) cuslide_tests

# Build benchmarks
make -j$(nproc) cuslide_benchmarks
```

### Option 2: Use the provided script

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim
chmod +x run_cuslide2_tests.sh
./run_cuslide2_tests.sh [optional_test_file.svs]
```

## 🧪 Run Tests

### Run all tests:

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/tests

# Set library paths
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$LD_LIBRARY_PATH

# Run tests
./cuslide_tests
```

### Run specific test:

```bash
# List available tests
./cuslide_tests --list-tests

# Run specific test by name
./cuslide_tests "test name"

# Run tests matching a tag
./cuslide_tests [tag]
```

### Run with test file:

```bash
./cuslide_tests /path/to/test/file.svs
```

## 📊 Run Benchmarks

### Run all benchmarks:

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/benchmarks

# Set library paths
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$LD_LIBRARY_PATH

# Run benchmarks
./cuslide_benchmarks
```

### Run specific benchmark:

```bash
# List available benchmarks
./cuslide_benchmarks --list_benchmarks

# Run specific benchmark by filter
./cuslide_benchmarks --benchmark_filter=<regex>

# Run with specific test file
./cuslide_benchmarks /path/to/test/file.svs
```

### Benchmark options:

```bash
# Run benchmarks with more iterations
./cuslide_benchmarks --benchmark_repetitions=10

# Output results to JSON
./cuslide_benchmarks --benchmark_format=json --benchmark_out=results.json

# Show time in microseconds
./cuslide_benchmarks --benchmark_time_unit=us
```

## 📁 Test Files

The tests expect test data files. Common locations:

- `/tmp/` - Default test data location
- Set `TEST_DATA_DIR` environment variable to specify custom location:

```bash
export TEST_DATA_DIR=/path/to/test/data
./cuslide_tests
```

## 🧪 Available Test Suites

Based on the source files:

1. **`test_read_region.cpp`** - Tests for region reading functionality
2. **`test_read_rawtiff.cpp`** - Tests for raw TIFF reading
3. **`test_philips_tiff.cpp`** - Tests for Philips TIFF format support

## 🔍 Test Dependencies

The tests use:
- **Catch2** - Test framework
- **OpenSlide** - Reference implementation for validation
- **CLI11** - Command-line argument parsing
- **fmt** - Formatted output

## 📝 Example Test Run

```bash
# Full test run with Aperio SVS file
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/tests
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$LD_LIBRARY_PATH
export CUCIM_PLUGIN_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/lib

./cuslide_tests /tmp/CMU-1-JP2K-33005.svs -s
```

Options:
- `-s` - Show successful assertions
- `-d yes` - Break into debugger on failure
- `-v high` - High verbosity

## 📊 Example Benchmark Run

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/benchmarks
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$LD_LIBRARY_PATH

./cuslide_benchmarks /tmp/CMU-1-JP2K-33005.svs --benchmark_repetitions=5
```

## 🚀 Quick Start

```bash
# Build everything
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release
make -j$(nproc)

# Run tests
cd tests
export LD_LIBRARY_PATH=/home/cdinea/Downloads/cucim_pr2/cucim/install/lib:$LD_LIBRARY_PATH
./cuslide_tests /tmp/CMU-1-JP2K-33005.svs

# Run benchmarks
cd ../benchmarks
./cuslide_benchmarks /tmp/CMU-1-JP2K-33005.svs
```

## 🐛 Debugging Tests

```bash
# Run tests under gdb
gdb --args ./cuslide_tests /tmp/CMU-1-JP2K-33005.svs

# Run with valgrind (memory checks)
valgrind --leak-check=full ./cuslide_tests

# Run with CUDA memory checker
cuda-memcheck ./cuslide_tests
```

## 📈 Continuous Integration

Tests can be run via CTest:

```bash
cd /home/cdinea/Downloads/cucim_pr2/cucim/cpp/plugins/cucim.kit.cuslide2/build-release
ctest --output-on-failure
```

