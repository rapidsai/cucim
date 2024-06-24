#!/bin/bash
# A utility script to download and untar Python wheel packages from S3.
# Positional Arguments:
#   1) wheel name
#   2) package type (one of: 'cpp', 'python'). If not provided, defaults to 'python' for compatibility with older code where python was the only behavior.
#
# [usage]
#
#   # download and untar any wheels found in CI artifacts with names like '*wheel_python_sparkly-unicorn*.tar.gz'
#   _rapids-wheels-prepare 'sparkly-unicorn' 'python'
#
set -eou pipefail
source rapids-constants
if [ -z "$1" ]; then
  rapids-echo-stderr "Must specify input arguments: WHEEL_NAME"
  exit 1
fi
WHEEL_NAME="$1"
PKG_TYPE="${2:-python}"
case "${PKG_TYPE}" in
  cpp)
    ;;
  python)
    ;;
  *)
    rapids-echo-stderr 'Pass one of the following package types: "cpp", "python"'
    exit 1
    ;;
esac
WHEEL_SEARCH_KEY="wheel_${PKG_TYPE}_${WHEEL_NAME}"
WHEEL_DIR="./dist"
mkdir -p "${WHEEL_DIR}"
S3_PATH=$(rapids-s3-path)
BUCKET_PREFIX=${S3_PATH/s3:\/\/${RAPIDS_DOWNLOADS_BUCKET}\//} # removes s3://rapids-downloads/ from s3://rapids-downloads/ci/rmm/...
# shellcheck disable=SC2016
WHEEL_TARBALLS=$(
  set -eo pipefail;
  aws \
    --output json \
    s3api list-objects \
    --bucket "${RAPIDS_DOWNLOADS_BUCKET}" \
    --prefix "${BUCKET_PREFIX}" \
    --page-size 100 \
    --query "Contents[?contains(Key, '${WHEEL_SEARCH_KEY}')].Key" \
    | jq -c
)
export WHEEL_TARBALLS
# first untar them all
for OBJ in $(jq -nr 'env.WHEEL_TARBALLS | fromjson | .[]'); do
  FILENAME=$(basename "${OBJ}")
  S3_URI="${S3_PATH}${FILENAME}"
  rapids-echo-stderr "Untarring ${S3_URI} into ${WHEEL_DIR}"
  aws s3 cp --only-show-errors "${S3_URI}" - | tar xzf - -C "${WHEEL_DIR}"
done
