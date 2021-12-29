import warnings

from .gray import (black_tophat, closing, dilation, erosion, opening,  # noqa
                   white_tophat)

__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat',
           'black_tophat']


warnings.warn(
    "Importing from cucim.skimage.morphology.grey is deprecated. "
    "Please import from cucim.skimage.morphology instead.",
    FutureWarning, stacklevel=2
)
