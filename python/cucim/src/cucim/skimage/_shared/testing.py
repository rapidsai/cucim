import os
import warnings

import cupy as cp
import numpy as np
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


_error_on_warnings = os.environ.get('SKIMAGE_TEST_STRICT_WARNINGS_GLOBAL', '0')
if _error_on_warnings.lower() == 'true':
    _error_on_warnings = True
elif _error_on_warnings.lower() == 'false':
    _error_on_warnings = False
else:
    try:
        _error_on_warnings = bool(int(_error_on_warnings))
    except ValueError:
        _error_on_warnings = False


def setup_test():
    """Default package level setup routine for skimage tests.

    Import packages known to raise warnings, and then
    force warnings to raise errors.

    Also set the random seed to zero.
    """
    warnings.simplefilter('default')

    if _error_on_warnings:
        np.random.seed(0)
        cp.random.seed(0)

        warnings.simplefilter('error')

        # do not error on specific warnings from the skimage.io module
        # https://github.com/scikit-image/scikit-image/issues/5337
        warnings.filterwarnings(
            'default', message='TiffFile:', category=DeprecationWarning
        )

        warnings.filterwarnings(
            'default', message='TiffWriter:', category=DeprecationWarning
        )
        # newer tifffile change the start of the warning string
        # e.g. <tifffile.TiffWriter.write> data with shape ...
        warnings.filterwarnings(
            'default',
            message='<tifffile.',
            category=DeprecationWarning
        )

        warnings.filterwarnings(
            'default', message='unclosed file', category=ResourceWarning
        )

        # Ignore other warnings only seen when using older versions of
        # dependencies.
        warnings.filterwarnings(
            'default',
            message='Conversion of the second argument of issubdtype',
            category=FutureWarning
        )

        warnings.filterwarnings(
            'default',
            message='the matrix subclass is not the recommended way',
            category=PendingDeprecationWarning, module='numpy'
        )

        warnings.filterwarnings(
            'default',
            message='Your installed pillow version',
            category=UserWarning,
            module='skimage.io'
        )

        # ignore warning from cycle_spin about Dask not being installed
        warnings.filterwarnings(
            'default',
            message='The optional dask dependency is not installed.',
            category=UserWarning
        )

        # ignore warning from CuPy about deprecated import from scipy.sparse
        warnings.filterwarnings(
            'default',
            message='Please use `spmatrix` from the `scipy.sparse` namespace',
            category=DeprecationWarning,
        )

        warnings.filterwarnings(
            'default',
            message='numpy.ufunc size changed',
            category=RuntimeWarning
        )


def teardown_test():
    """Default package level teardown routine for skimage tests.

    Restore warnings to default behavior
    """
    if _error_on_warnings:
        warnings.resetwarnings()
        warnings.simplefilter('default')
