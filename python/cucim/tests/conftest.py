#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import time
from pathlib import Path

import pytest

from .fixtures.testimage import *  # noqa

_stress_trace = pytest.StashKey[Path]()
_stress_failed = pytest.StashKey[bool]()


def pytest_generate_tests(metafunc):
    if "cuda_read_trace" in metafunc.fixturenames:
        repeats = int(os.environ.get("CUCIM_CUDA_TEST_REPEAT", "1"))
        if repeats > 1:
            metafunc.parametrize(
                "cuda_read_trace",
                range(1, repeats + 1),
                indirect=True,
                ids=lambda attempt: f"repeat-{attempt:03d}",
            )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    path = item.stash.get(_stress_trace, None)
    if path:
        if report.failed:
            item.stash[_stress_failed] = True
        if report.when == "teardown" and not item.stash.get(
            _stress_failed, False
        ):
            path.unlink(missing_ok=True)
    return report


@pytest.fixture
def cuda_read_trace(request, monkeypatch):
    """Write phase markers outside pytest capture so a hung test leaves a trace."""
    trace_dir = os.environ.get("CUCIM_CUDA_TEST_TRACE_DIR")
    path = None
    if trace_dir:
        path = Path(trace_dir) / f"{request.node.name}-{os.getpid()}.log"
        monkeypatch.setenv("CUCIM_CUDA_TEST_TRACE_FILE", str(path))
        if hasattr(request, "param"):
            # Keep only failed or unfinished stress-test traces for the CI log.
            request.node.stash[_stress_trace] = path

    def trace(phase):
        if path:
            with path.open("a") as stream:
                stream.write(f"{time.monotonic_ns() // 1000} pytest {phase}\n")

    trace("test_begin")
    yield trace
    trace("test_teardown")
