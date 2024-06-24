#!/bin/bash
# A utility script to upload Python wheel packages to PyPI repository using an OIDC token.
# Positional Arguments:
#   1) wheel name
#   2) package type (one of: 'cpp', 'python'). If not provided, defaults to 'python' for compatibility with older code where python was the only behavior.
#
# [usage]
#
#   # upload any wheels found in CI artifacts with names like '*wheel_python_sparkly-unicorn*.tar.gz'
#   rapids-wheels-pypi 'sparkly-unicorn' 'python'
#
set -eou pipefail
source rapids-constants
export RAPIDS_SCRIPT_NAME="rapids-wheels-pypi"
WHEEL_NAME="$1"
PKG_TYPE="${2:-python}"
WHEEL_DIR="./dist"
_rapids-wheels-prepare "${WHEEL_NAME}" "${PKG_TYPE}"

if [ -z "${PYPI_TOKEN}" ]; then
  rapids-echo-stderr "Must specify input arguments: PYPI_TOKEN"
  exit 1
fi

# shellcheck disable=SC2086
rapids-retry python -m twine \
  upload \
  --repository testpypi \
  --disable-progress-bar \
  --non-interactive \
  --skip-existing \
  "${WHEEL_DIR}"/*.whl

echo ""
