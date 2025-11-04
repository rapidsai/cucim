#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

docker build -t gigony/svs-testdata:little-big ${SCRIPT_DIR}
docker push gigony/svs-testdata:little-big
