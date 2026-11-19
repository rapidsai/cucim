#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail

repeat_count=${1:-1}
input_staging=${2:-device}
nvjpeg_override=${3:-}
if [[ ! "${repeat_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected a positive CUDA read test repeat count." >&2
  exit 2
fi
if [[ "${input_staging}" != device && "${input_staging}" != host ]]; then
  echo "Expected host or device CUDA read input staging." >&2
  exit 2
fi

if [[ -n "${nvjpeg_override}" && "${nvjpeg_override}" != 13.1.0.59 ]]; then
  echo "The diagnostic nvJPEG override supports 13.1.0.59." >&2
  exit 2
fi

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcucim cucim --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cucim cucim --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

if [[ -n "${nvjpeg_override}" ]]; then
  rapids-logger "Load nvJPEG ${nvjpeg_override} for this diagnostic run"
  nvjpeg_dir=$(mktemp -d)
  trap 'rm -rf -- "${nvjpeg_dir}"' EXIT
  nvjpeg_archive="libnvjpeg-linux-x86_64-${nvjpeg_override}-archive"
  nvjpeg_library="libnvjpeg.so.${nvjpeg_override}"
  (
    cd "${nvjpeg_dir}"
    curl --fail --silent --show-error --location --retry 3 --max-time 60 \
      "https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/linux-x86_64/${nvjpeg_archive}.tar.xz" \
      --output nvjpeg.tar.xz
    # SHA-256 from NVIDIA's redistrib_13.2.2.json.
    printf '%s  %s\n' cd4d825630e85bee77444cf54cfe1d8a271cb496b6c61d398359e1ad8bcd20e2 nvjpeg.tar.xz | sha256sum --check
    tar -xf nvjpeg.tar.xz --strip-components=2 "${nvjpeg_archive}/lib/${nvjpeg_library}"
  )
  export LD_PRELOAD="${nvjpeg_dir}/${nvjpeg_library}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

rapids-logger "Check the nvJPEG version used by cuslide"
python - "${nvjpeg_override%.*}" <<'PY_VERSION'
import ctypes
import sys
from pathlib import Path

expected_version = sys.argv[1]


def unavailable(message):
    print(message)
    sys.exit(1 if expected_version else 0)


plugin = next((Path(sys.prefix) / "lib").glob("cucim.kit.cuslide@*.so"), None)
if plugin is None:
    unavailable("The installed cuslide plugin was not found")

try:
    lib = ctypes.CDLL(str(plugin))
    get_version = lib.nvjpegGetProperty
except (OSError, AttributeError) as error:
    unavailable(f"Could not query nvJPEG from {plugin.name}: {error}")

get_version.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
get_version.restype = ctypes.c_int
version = []
for part in range(3):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
    value = ctypes.c_int()
    status = get_version(part, ctypes.byref(value))
    if status:
        unavailable(f"nvjpegGetProperty failed with status {status}")
    version.append(str(value.value))
actual_version = ".".join(version)
print(f"{plugin.name} uses nvJPEG {actual_version}")
if expected_version and actual_version != expected_version:
    sys.exit(f"Expected nvJPEG {expected_version}, got {actual_version}")
PY_VERSION

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cucim"
rapids-logger "nvJPEG input staging: ${input_staging}"
export CUCIM_CUDA_TEST_REPEAT="${repeat_count}"
export CUCIM_CUDA_TEST_INPUT_STAGING="${input_staging}"
export CUCIM_CUDA_TEST_TRACE_DIR
CUCIM_CUDA_TEST_TRACE_DIR=$(mktemp -d)
test_timeout=20m
test_targets=(src tests/unit tests/performance)
if [[ "${repeat_count}" != 1 ]]; then
  rapids-logger "Repeat each of the three CUDA read tests ${repeat_count} times"
  test_timeout=6m
  test_targets=(
    tests/unit/clara/test_batch_decoding.py::TestBatchDecodingCUDA::test_batch_read_cuda_memory_cleanup
    tests/unit/clara/test_bindings_cuimage.py::test_cuda_array_interface
    tests/unit/clara/test_tiff_read_region.py::test_cuda_array_interface_support
  )
fi
pushd python/cucim
# Print stacks for stalled xdist tests before the job-wide timeout.
timeout "${test_timeout}" pytest \
  --cache-clear \
  -o faulthandler_timeout=120 \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cucim.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  --cov-config=.coveragerc \
  --cov=cucim \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cucim-coverage.xml" \
  --cov-report=term \
  -v \
  "${test_targets[@]}"
popd

if (( EXITCODE != 0 )); then
  for trace in "${CUCIM_CUDA_TEST_TRACE_DIR}"/*.log; do
    if [[ -f "${trace}" ]]; then
      rapids-logger "CUDA read test trace: ${trace##*/}"
      tail -n 300 "${trace}"
    fi
  done
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
