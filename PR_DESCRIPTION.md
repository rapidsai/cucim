# Feature: nvImageCodec Infrastructure Setup for cuslide2

This PR adds the infrastructure and build system support for integrating nvImageCodec into the cuslide2 plugin. This is a foundational PR that sets up the dependencies, CMake configuration, and conda packaging without implementing the actual nvImageCodec functionality in the plugin.

## Changes Overview

### 1. Dependency Management
- Added nvImageCodec dependencies to `dependencies.yaml`:
  - `libnvimgcodec` (runtime library)
  - `libnvimgcodec-dev` (development headers)
  - `libnvimgcodec0` (base library)
- Regenerated all conda environment files from dependencies

### 2. Build System Updates
- Updated `cpp/plugins/cucim.kit.cuslide2/CMakeLists.txt` with nvImageCodec integration
- Added CMake dependency files in `cpp/plugins/cucim.kit.cuslide2/cmake/deps/`:
  - `cli11.cmake`, `catch2.cmake`, `boost.cmake`
  - `fmt.cmake`, `json.cmake`, `pugixml.cmake`
  - `libdeflate.cmake`, `libtiff.cmake`, `libopenjpeg.cmake`
  - `googletest.cmake`, `googlebenchmark.cmake`, `openslide.cmake`
  - Policy fix files for libjpeg-turbo and libtiff

### 3. Python Package Updates
- Updated `python/cucim/pyproject.toml` with nvImageCodec dependencies
- Ensured Python bindings are ready for future nvImageCodec integration

### 4. Verification Script
- Added `scripts/verify_cuslide2_infrastructure.py` - comprehensive infrastructure verification tool
- Checks nvImageCodec installation, library files, CMake configuration, CUDA environment, and plugin build

### 5. Code Quality
- Updated all copyright headers to SPDX format
- Fixed typo in verification script (asssignment → assignment)

## Setup and Verification Instructions

### Prerequisites
- Micromamba or conda/mamba package manager
- NVIDIA GPU with appropriate drivers

### Step 1: Create and Activate Environment

```bash
# Create new micromamba environment with Python 3.10
micromamba create -n cucim-test python=3.10

# Activate the environment
micromamba activate cucim-test
```

### Step 2: Install Build Dependencies

```bash
# Install Python, CUDA toolkit, compilers, OpenSlide, and build tools
# Note: Python development headers are included in the python package
micromamba install -y python=3.10 cuda-toolkit c-compiler cxx-compiler openslide yasm -c conda-forge
```

### Step 3: Set Compiler Environment Variables

```bash
# Set compilers to use conda environment versions
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDACXX=$CONDA_PREFIX/pkgs/cuda-toolkit/bin/nvcc
```

### Step 4: Build cuCIM with cuslide2 Plugin

```bash
# From the repository root, build all with release binaries
# (you can change 'release' to 'debug' or 'rel-debug')
./run build_local all release $CONDA_PREFIX
```

**Note**: The `build_local` script will:
- Install nvImageCodec packages (libnvimgcodec, libnvimgcodec-dev, libnvimgcodec0) from conda-forge based on `dependencies.yaml`
- Install additional Python and C++ dependencies from `dependencies.yaml`
- Configure and build the C++ library including the cuslide2 plugin
- Build and install the Python package

### Step 5: Run Verification Script

```bash
# From the repository root
python scripts/verify_cuslide2_infrastructure.py
```

## Verification Output

When the infrastructure is properly set up, the verification script produces the following output:

```
============================================================
 cuslide2 Infrastructure Verification
============================================================
Platform: Linux 6.8.0-1029-gcp
Python: 3.13.9 | packaged by conda-forge | (main, Oct 22 2025, 23:33:35) [GCC 14.3.0]

----------------------------------------
 nvImageCodec Installation Detection
----------------------------------------
✓ Method 1: Conda packages detected (from dependencies.yaml):
    libnvimgcodec      0.6.0    hd8ed1ab_0  conda-forge
    libnvimgcodec-dev  0.6.0    h0850aa4_0  conda-forge
    libnvimgcodec0     0.6.0    hb7e823c_0  conda-forge
  Environment: /home/cdinea/micromamba/envs/cucim-test

----------------------------------------
 nvImageCodec Library Files Check
----------------------------------------
✓ Header found: /home/cdinea/micromamba/envs/cucim-test/include/nvimgcodec.h
✓ Library found: /home/cdinea/micromamba/envs/cucim-test/lib/libnvimgcodec.so.0 (33.3 MB)

----------------------------------------
 CMake Build Configuration Check
----------------------------------------
✓ CMake cache found: /home/cdinea/Downloads/cucim_pr1/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/CMakeCache.txt
✓ nvImageCodec configuration found in CMake cache
  AUTO_INSTALL_NVIMGCODEC:BOOL=ON
  NVIMGCODEC_VERSION:STRING=0.6.0
  nvimgcodec_DIR:PATH=/home/cdinea/micromamba/envs/cucim-test/lib/cmake/nvimgcodec

----------------------------------------
 CUDA Environment Check
----------------------------------------
✓ NVIDIA GPU detected:
  |   0  NVIDIA RTX A6000               Off |   00000000:01:00.0  On |                  Off |
✓ CUDA compiler: Cuda compilation tools, release 12.9, V12.9.86

----------------------------------------
 cuslide2 Plugin Check
----------------------------------------
✓ Plugin found: cucim.kit.cuslide2@25.12.00.so
  Size: 2.6 MB
  Location: /home/cdinea/Downloads/cucim_pr1/cucim/cpp/plugins/cucim.kit.cuslide2/build-release/lib
ℹ Plugin uses dynamic loading or static linking for nvImageCodec
  Library integrated at build time via plugin system
✓ Plugin linked with CUDA libraries

----------------------------------------
 Python Import Test
----------------------------------------
✓ nvidia.nvimgcodec module imported successfully
  Version: 0.6.1

============================================================
 Infrastructure Summary
============================================================
🎉 cuslide2 infrastructure is properly set up!
✓ nvImageCodec installed via conda packages (dependencies.yaml)
✓ All required library files accessible
✓ cuslide2 plugin built successfully
```

## Testing

### Local Testing
1. Follow the setup instructions above
2. Run the verification script to confirm infrastructure is working
3. The plugin should build successfully without errors

### CI Testing
The CI pipeline will:
1. Build the C++ library with cuslide2 plugin
2. Build the Python package
3. Run all existing tests
4. Verify copyright headers and code style

## Notes

- **This PR does NOT implement nvImageCodec functionality** - it only sets up the infrastructure
- The cuslide2 plugin builds successfully with nvImageCodec dependencies available
- Future PRs will add the actual nvImageCodec codec implementation
- All existing functionality remains unchanged

## Related Issues

This is part of the work to integrate nvImageCodec support into cuCIM for improved image decoding performance.

## Checklist

- [x] Added nvImageCodec dependencies to `dependencies.yaml`
- [x] Updated CMake build configuration
- [x] Regenerated conda environment files
- [x] Added verification script
- [x] Updated copyright headers to SPDX format
- [x] Plugin builds successfully with nvImageCodec dependencies
- [x] All pre-commit checks pass
- [x] Documentation updated (this PR description)

