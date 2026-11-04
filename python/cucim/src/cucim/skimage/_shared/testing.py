# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import inspect
import time

import pytest
from numpy.testing import (  # noqa
    TestCase,
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_no_warnings,
    assert_warns,
)

from ._warnings import expected_warnings  # noqa

skipif = pytest.mark.skipif
xfail = pytest.mark.xfail
parametrize = pytest.mark.parametrize
raises = pytest.raises
fixture = pytest.fixture

have_fetch = True
try:
    # scikit-image >=0.19
    from skimage.data import _fetchers
    from skimage.data._fetchers import _fetch
except ImportError:
    # skip this test if private API changed on scikit-image end
    have_fetch = False
    _fetchers = None
    _fetch = None

# Number of retries for a failed download that was not rate limited.
_DOWNLOAD_RETRIES = 3
# Message of the first rate-limited download in this session, if any.
_rate_limit_message = None


def _download_or_skip(load, description):
    """Return ``load()``, skipping the test if the data cannot be downloaded.

    A rate-limited (HTTP 429) download is not retried: GitLab.com, which
    hosts the scikit-image data, limits anonymous traffic per IP address and
    hour, so a retry seconds later also fails and counts against the limit.
    Once a download is rate limited, later downloads in the session are
    skipped without a request. pooch retries every failed download, so its
    retries are disabled here and other failures (e.g. a dropped connection)
    are retried in the same way instead.
    """
    global _rate_limit_message

    if _rate_limit_message is not None:
        pytest.skip(
            f"Not downloading {description}: an earlier download was rate "
            f"limited ({_rate_limit_message})"
        )
    fetcher = getattr(_fetchers, "_image_fetcher", None)
    pooch_retries = getattr(fetcher, "retry_if_failed", None)
    if pooch_retries is not None:
        fetcher.retry_if_failed = 0
    try:
        for attempt in range(_DOWNLOAD_RETRIES + 1):
            try:
                return load()
            except (OSError, ModuleNotFoundError) as err:
                # OSError includes the HTTP errors raised by requests.
                response = getattr(err, "response", None)
                rate_limited = getattr(response, "status_code", None) == 429
                if rate_limited:
                    _rate_limit_message = str(err)
                if (
                    rate_limited
                    or isinstance(err, ModuleNotFoundError)
                    or attempt == _DOWNLOAD_RETRIES
                ):
                    pytest.skip(f"Unable to download {description}: {err}")
                time.sleep(attempt + 1)
    finally:
        if pooch_retries is not None:
            fetcher.retry_if_failed = pooch_retries


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    if not have_fetch:
        pytest.skip("skimage _fetch utility not found")
    # CuPy Backend: TODO: avoid call to non-public _fetch method
    return _download_or_skip(lambda: _fetch(data_filename), data_filename)


def load_sample_data(name):
    """Load a scikit-image sample dataset, skipping the test if unavailable.

    Datasets that are not bundled with scikit-image (e.g. ``eagle``,
    ``brain``, ``cells3d``) are downloaded on first use, which can fail.
    Call this from a module-scoped fixture when several tests use the same
    data, so that a failed download is attempted only once per module.

    Parameters
    ----------
    name : str
        Name of the loader function in ``skimage.data``.
    """
    from skimage import data

    loader = getattr(data, name, None)
    if loader is None:
        pytest.skip(
            f"{name} data not available in this version of scikit-image"
        )
    return _download_or_skip(loader, f"{name} data")


def assert_stacklevel(warnings, *, offset=-1):
    """Assert correct stacklevel of captured warnings.

    When cucim.skimage raises warnings, the stacklevel should ideally be set
    so that the origin of the warnings will point to the public function
    that was called by the user and not necessarily the very place where the
    warnings were emitted (which may be inside some internal function).
    This utility function helps with checking that
    the stacklevel was set correctly on warnings captured by `pytest.warns`.

    Parameters
    ----------
    warnings : collections.abc.Iterable[warning.WarningMessage]
        Warnings that were captured by `pytest.warns`.
    offset : int, optional
        Offset from the line this function is called to the line were the
        warning is supposed to originate from. For multiline calls, the
        first line is relevant. Defaults to -1 which corresponds to the line
        right above the one where this function is called.

    Raises
    ------
    AssertionError
        If a warning in `warnings` does not match the expected line number or
        file name.

    Examples
    --------
    >>> def test_something():
    ...     with pytest.warns(UserWarning, match="some message") as record:
    ...         something_raising_a_warning()
    ...     assert_stacklevel(record)
    ...
    >>> def test_another_thing():
    ...     with pytest.warns(UserWarning, match="some message") as record:
    ...         iam_raising_many_warnings(
    ...             "A long argument that forces the call to wrap."
    ...         )
    ...     assert_stacklevel(record, offset=-3)
    """
    __tracebackhide__ = True  # Hide traceback for py.test

    frame = inspect.stack()[1].frame  # 0 is current frame, 1 is outer frame
    line_number = frame.f_lineno + offset
    filename = frame.f_code.co_filename
    expected = f"{filename}:{line_number}"
    for warning in warnings:
        actual = f"{warning.filename}:{warning.lineno}"
        assert actual == expected, f"{actual} != {expected}"
        msg = (
            "Warning with wrong stacklevel:\n"
            f"  Expected: {expected}\n"
            f"  Actual: {actual}\n"
            f"  {warning.category.__name__}: {warning.message}"
        )
        assert actual == expected, msg
