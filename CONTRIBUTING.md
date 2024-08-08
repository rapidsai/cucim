# Contribute to cuCIM

If you are interested in contributing to cuCIM, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/cucim/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/rapidsai/cucim/blob/main/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cucim/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cucim/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cucim/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/cucim/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.


## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuCIM OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuCIM from the source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

### Code Formatting

#### Python


cuCIM uses [ruff](https://docs.astral.sh/ruff/) and [black](https://black.readthedocs.io/en/stable/) to ensure a consistent code format
throughout the project. `ruff`, and `black` can be installed with
`conda` or `pip`:

```bash
conda install black ruff
```

```bash
pip install black ruff
```

These tools are used to auto-format the Python code in the repository. Additionally, there is a CI check in place to enforce that the committed code follows our standards. To run only for the python/cucim folder, change to that folder and run

```bash
black .
ruff .
```

To also check formatting in top-level folders like `benchmarks`, `examples` and `experiments`, these tools can also be run from the top level of the repository as follows:

```bash
black --config python/cucim/pyproject.toml .
ruff --config python/cucim/pyproject.toml .
```

In addition to these tools, [codespell](https://github.com/codespell-project/codespell) can be used to help diagnose and interactively fix spelling errors in both Python and C++ code. It can also be run from the top level of the repository in interactive mode using:

```bash
codespell --toml python/cucim/pyproject.toml . -i 3 -w
```

If codespell is finding false positives in newly added code, the `ignore-words-list` entry of the `tool.codespell` section in `pyproject.toml` can be updated as needed.

### Get libcucim Dependencies

Compiler requirements:

* `gcc`     version 9.0+
* `nvcc`    version 11.0+
* `cmake`   version 3.18.0+

CUDA/GPU requirements:

* CUDA 11.0+
* NVIDIA driver 450.36+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).


# Building and Testing cuCIM from Source

First, please clone cuCIM's repository

```bash
CUCIM_HOME=$(pwd)/cucim
git clone https://github.com/rapidsai/cucim.git $CUCIM_HOME
cd $CUCIM_HOME
```
## Local Development using Conda Environment (for gcc 9.x and nvcc 11.0.x)

Conda can be used to setup an environment which includes all of the necessary dependencies (as shown in `./conda/environments/all_cuda-118_arch-x86_64.yaml`) for building cuCIM.

Otherwise, you may need to install dependencies (such as zlib, xz, yasm) through your OS's package manager (`apt`, `yum`, and so on).


### Creating the Conda Development Environment `cucim`

Note that `./conda/environments/all_cuda-118_arch-x86_64.yaml` is currently set to use specific versions of gcc (gxx_linux-64) and CUDA (cudatoolkit & cudatoolkit-dev).

If you want to change the version of gcc or CUDA toolkit package, please update `./conda/environments/all_cuda-118_arch-x86_64.yaml` before executing the following commands.

```bash
mamba env create -n cucim -f conda/environments/all_cuda-125_arch-x86_64.yaml
# activate the environment
mamba activate cucim
```

### Building `libcucim` and install `cucim` (python bindings):

**Building `libcucim`**

```bash
# `CC` and `CXX` environment variable will be set by default by `gxx_linux-64` package.
# Check if the environments are correctly set.
[ ${CC##$CONDA_PREFIX/bin} = "$CC" ] && >&2 echo "Environment variable CC doesn't start with '$CONDA_PREFIX/bin'"
[ ${CXX##$CONDA_PREFIX/bin} = "$CXX" ] && >&2 echo "Environment variable CXX doesn't start with '$CONDA_PREFIX/bin'"

# set to use nvcc in the conda environment
export CUDACXX=$CONDA_PREFIX/pkgs/cuda-toolkit/bin/nvcc

# build all with `release` binaries (you can change it to `debug` or `rel-debug`)
./run build_local all release $CONDA_PREFIX
```

The build command will create the following files:
- `./install/lib/libcucim*`
- `./python/install/lib/_cucim.cpython-*-x86_64-linux-gnu.so`
- `./cpp/plugins/cucim.kit.cuslide/install/lib/cucim.kit.cuslide@*.so`
- `./cpp/plugins/cucim.kit.cumed/install/lib/cucim.kit.cumed@*.so`

And, it will copy the built library files to `python/cucim/src/cucim/clara/` folder:
- `libcucim.so.*`
- `_cucim.cpython-*-x86_64-linux-gnu.so`
- `cucim.kit.cuslide@*.so`
- `cucim.kit.cumed@*.so`


**Building `cucim`(python bindings)**

```bash
python -m pip install python/cucim
```

For contributors interested in working on the Python code from an in-place
(editable) installation, replace the last line above with
```bash
python -m pip install --editable python/cucim
```

**Cleaning build files**

You can execute the following command whenever C++ code is changed during the development:
```bash
./run build_local all release $CONDA_PREFIX
```
Once it is built, the subsequent build doesn't take much time.

However, if a build option or dependent packages are updated, the build can be failed (due to CMakeCache.txt or existing build files). In that case, you can remove use the following commands to remove CMakeCache.txt or build folder, then build it again.

1) Remove CMakeCache.txt for libcucim, cuslide/cumed plugin, and the python wrapper (pybind11).

```bash
# this command wouldn't remove already downloaded dependency so faster than `clean` subcommand
./run build_local clean_cache
```

2) Remove `build-*` and `install` folder for libcucim, cuslide/cumed plugin, and the python wrapper (pybind11).

```bash
# this command is for clean build
./run build_local clean
```

## Building a Conda package

**Setup**

You can build a conda package on top of `cucim` Conda environment created by instructions above:

```bash
conda activate cucim
```

First, please make sure that you have `conda-build` installed:

```bash
# Install conda-build if `conda build` command is not available.
! conda build --help > /dev/null && conda install -c conda-forge conda-build
```

Export necessary environment variables:

```bash
export CUDA="$(conda list | grep cudatoolkit-dev | egrep -o "[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+")"
export PYTHON_VER="$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")"
echo "CUDA       : ${CUDA}"
echo "PYTHON_VER : ${PYTHON_VER}"
```

Then, create `conda-bld` folder:

```bash
CONDA_BLD_DIR=$(pwd)/conda-bld
mkdir -p $CONDA_BLD_DIR
```

**Build**

```bash
conda build -c conda-forge \
    --dirty \
    --no-remove-work-dir \
    --no-build-id \
    --croot ${CONDA_BLD_DIR} \
    --use-local \
    conda/recipes/libcucim \
    conda/recipes/cucim

# Conda Package files would be available at `conda-bld/linux-64`
ls conda-bld/linux-64/*cucim*
```

**Install**

```bash
conda install -y -c ${CONDA_BLD_DIR} -c conda-forge \
    libcucim \
    cucim
```

## Building a package (for distribution. Including a wheel package for pip)

**Wheel Build**

If you are using CUDA 12.x, please update pyproject.toml as follows before building the wheel
```bash
sed -i "s/cupy-cuda11x/cupy-cuda12x/g" python/cucim/pyproject.toml
```
This will switch the CuPy dependency to one based on CUDA 12.x instead of 11.x.

The wheel can then be built using:

```bash
python -m pip wheel python/cucim/ -w dist -vvv --no-deps --disable-pip-version-check
```

**Note:** It is possible to build the wheel in this way even without compiling the C++ library first, but in that case the `cucim.clara` module will not be importable.

**Install**

```bash
python -m pip install dist/cucim*.whl
```

## Running Tests

Once cuCIM is installed, you can test the module through `./run test` command.

```bash
# Arguments:
#   $1 - subcommand [all|python|c++] (default: all)
#   $2 - test_type [all|unit|integration|system|performance] (default: all)
#   $3 - test_component [all|clara|skimage] (default: all)

./run test                      # execute all tests
./run test python               # execute all python tests
./run test python unit          # execute all python unit tests
./run test python unit skimage  # execute all python unit tests in `skimage` module
./run test python unit clara    # execute all python unit tests in `clara` module
./run test python performance   # execute all python performance tests
```
