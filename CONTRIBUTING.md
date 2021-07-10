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

cuCIM uses [isort](https://readthedocs.org/projects/isort/), and
[flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent code format
throughout the project. `isort`, and `flake8` can be installed with
`conda` or `pip`:

```bash
conda install isort flake8
```

```bash
pip install isort flake8
```

These tools are used to auto-format the Python code in the repository. Additionally, there is a CI check in place to enforce
that the committed code follows our standards. You can use the tools to
automatically format your python code by running:

```bash
isort --atomic python/**/*.py
```

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

First, please clone the cuCIM's repository

```bash
CUCIM_HOME=$(pwd)/cucim
git clone https://github.com/rapidsai/cucim.git $CUCIM_HOME
cd $CUCIM_HOME
```
## Local Development using Conda Environment (for gcc 9.x and nvcc 11.0.x)

Conda can be used to setup GCC 9.x, CUDA Toolkit (including nvcc) 11.0.x, and other dependent libraries (as shown in `./conda/environments/env.yml`) for building cuCIM.

Otherwise, you may need to install (such as zlib, xz, yasm) through your OS's package manager (`apt`, `yum`, and so on).


### Creating the Conda Development Environment `cucim`

Note that `./conda/environments/env.yml` is currently set to use gcc 9.x (gxx_linux-64) and CUDA 11.0.x (cudatoolkit & cudatoolkit-dev). If you want to change the version of gcc or CUDA toolkit
package, please update `./conda/environments/env.yml` before executing the following commands.

```bash
conda env create -f ./conda/environments/env.yml
# activate the environment
conda activate cucim
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

And, it will copy the built library files to `python/cucim/src/cucim/clara/` folder:
- `libcucim.so.*`
- `_cucim.cpython-*-x86_64-linux-gnu.so`
- `cucim.kit.cuslide@*.so`


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

1) Remove CMakeCache.txt for libcucim, cuslide plugin, and the python wrapper (pybind11).

```bash
# this command wouldn't remove already downloaded dependency so faster than `clean` subcommand
./run build_local clean_cache
```

2) Remove `build-*` and `install` folder for libcucim, cuslide plugin, and the python wrapper (pybind11).

```bash
# this command is for clean build
./run build_local clean
```

## Building a Package (for distribution. Including a wheel package for pip)

You can execute the following command to build a wheel file for pip.

```bash
./run build_package
```

The command would use `./temp` folder as a local build folder and build a distribution package into `dist` folder using [dockcross](https://github.com/dockcross/dockcross)'s manylinux2014 docker image.

`./run build_package` will reuse local `./temp` folder to reduce the build time.

If C++ code or dependent packages are updated so the build is failing somehow, please retry it after deleting the `temp` folder under the repository root.

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