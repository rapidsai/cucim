import warnings

from .grayreconstruct import reconstruction  # noqa

warnings.warn(
    "Importing from cucim.skimage.morphology.greyreconstruct is deprecated. "
    "Please import from cucim.skimage.morphology instead.",
    FutureWarning, stacklevel=2
)
