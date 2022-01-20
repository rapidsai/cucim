import pytest
from numpy.testing import (TestCase, assert_, assert_allclose,  # noqa
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_array_equal,
                           assert_array_less, assert_equal, assert_no_warnings,
                           assert_warns)

from ._warnings import expected_warnings  # noqa

skipif = pytest.mark.skipif
xfail = pytest.mark.xfail
parametrize = pytest.mark.parametrize
raises = pytest.raises
fixture = pytest.fixture

have_fetch = True
try:
    # scikit-image 0.19
    from skimage.data._fetchers import _fetch
except ImportError:
    # scikit-image 0.18
    try:
        from skimage.data import _fetch
    except ImportError:
        have_fetch = False


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    if have_fetch:
        try:
            # CuPy Backend: TODO: avoid call to non-public _fetch method
            return _fetch(data_filename)
        except (ConnectionError, ModuleNotFoundError):
            pytest.skip(f'Unable to download {data_filename}')
    else:
        pytest.skip('skimage _fetch utility not found')
