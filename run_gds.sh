#!/bin/bash
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

MNT_PATH=/nvme
GDS_IMAGE=cucim-gds

BUILD_VER=`uname -r`
NV_DRIVER=`nvidia-smi -q -i 0 | sed -n 's/Driver Version.*: *\(.*\) *$/\1/p'`
echo "using nvidia driver version $NV_DRIVER on kernel $BUILD_VER"


ofed_version=$(ofed_info -s | grep MLNX)
if [ $? -eq 0 ]; then
    rdma_core=$(dpkg -s libibverbs-dev | grep "Source: rdma-core")
    if [ $? -eq 0 ]; then
        CONFIG_MOFED_VERSION=$(echo $ofed_version | cut -d '-' -f 2)
        echo "Found MOFED version $CONFIG_MOFED_VERSION"
    fi
    MLNX_SRCS="--volume /usr/src/mlnx-ofed-kernel-${CONFIG_MOFED_VERSION}:/usr/src/mlnx-ofed-kernel-${CONFIG_MOFED_VERSION}:ro"
    MOFED_DEVS="--net=host --volume /sys/class/infiniband_verbs:/sys/class/infiniband_verbs/ "
fi

docker run \
    --ipc host \
    -it
    --rm
    --gpus all \
    --volume /run/udev:/run/udev:ro \
    --volume /sys/kernel/config:/sys/kernel/config/ \
    --volume /usr/src/nvidia-$NV_DRIVER:/usr/src/nvidia-$NV_DRIVER:ro  ${MLNX_SRCS}\
    --volume /dev:/dev:ro \
    --privileged \
    --env NV_DRIVER=${NV_DRIVER} \
    --volume 	/lib/modules/$BUILD_VER/:/lib/modules/$BUILD_VER \
    --volume "${MNT_PATH}/data:/data:rw" \
    --volume "${MNT_PATH}/results:/results:rw"  ${MOFED_DEVS} \
    -itd ${REPO_URI}/${GDS_IMAGE} \
    /bin/bash
