#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import time
from pathlib import Path

import pytest

from .fixtures.testimage import *  # noqa


@pytest.fixture
def cuda_read_trace(request, monkeypatch):
    """Write phase markers outside pytest capture so a hung test leaves a trace."""
    trace_dir = os.environ.get("CUCIM_CUDA_TEST_TRACE_DIR")
    path = None
    if trace_dir:
        path = Path(trace_dir) / f"{request.node.name}-{os.getpid()}.log"
        monkeypatch.setenv("CUCIM_CUDA_TEST_TRACE_FILE", str(path))

    def trace(phase):
        if path:
            with path.open("a") as stream:
                stream.write(f"{time.monotonic_ns() // 1000} pytest {phase}\n")

    trace("test_begin")
    yield trace
    trace("test_teardown")
