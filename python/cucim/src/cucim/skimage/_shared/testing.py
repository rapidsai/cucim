import inspect

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
    from skimage.data._fetchers import _fetch
except ImportError:
    # skip this test if private API changed on scikit-image end
    have_fetch = False


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    if have_fetch:
        try:
            # CuPy Backend: TODO: avoid call to non-public _fetch method
            return _fetch(data_filename)
        except (ConnectionError, ModuleNotFoundError):
            pytest.skip(f"Unable to download {data_filename}")
    else:
        pytest.skip("skimage _fetch utility not found")


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
