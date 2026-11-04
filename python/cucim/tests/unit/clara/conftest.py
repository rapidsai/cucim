#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest


@pytest.fixture
def restore_cache():
    from cucim import CuImage

    config = CuImage.cache().config
    yield
    CuImage.cache(
        config["type"],
        memory_capacity=config["memory_capacity"],
        capacity=config["capacity"],
        mutex_pool_capacity=config["mutex_pool_capacity"],
        list_padding=config["list_padding"],
        extra_shared_memory_size=config["extra_shared_memory_size"],
        record_stat=config["record_stat"],
    )


@pytest.fixture
def restore_trace():
    from cucim import CuImage

    trace = CuImage.profiler().trace()
    yield
    CuImage.profiler().trace(trace)
