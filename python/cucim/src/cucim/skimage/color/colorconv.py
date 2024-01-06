"""Functions for converting between color spaces.

The "central" color space in this module is RGB, more specifically the linear
sRGB color space using D65 as a white-point [1]_.  This represents a
standard monitor (w/o gamma correction). For a good FAQ on color spaces see
[2]_.

The API consists of functions to convert to and from RGB as defined above, as
well as a generic function to convert to and from any supported color space
(which is done through RGB in most cases).


Supported color spaces
----------------------
* RGB : Red Green Blue.
        Here the sRGB standard [1]_.
* HSV : Hue, Saturation, Value.
        Uniquely defined when related to sRGB [3]_.
* RGB CIE : Red Green Blue.
        The original RGB CIE standard from 1931 [4]_. Primary colors are 700 nm
        (red), 546.1 nm (blue) and 435.8 nm (green).
* XYZ CIE : XYZ
        Derived from the RGB CIE color space. Chosen such that
        ``x == y == z == 1/3`` at the whitepoint, and all color matching
        functions are greater than zero everywhere.
* LAB CIE : Lightness, a, b
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LUV CIE : Lightness, u, v
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LCH CIE : Lightness, Chroma, Hue
        Defined in terms of LAB CIE.  C and H are the polar representation of
        a and b.  The polar angle C is defined to be on ``(0, 2*pi)``

:author: Nicolas Pinto (rgb2hsv)
:author: Ralf Gommers (hsv2rgb)
:author: Travis Oliphant (XYZ and RGB CIE functions)
:author: Matt Terry (lab2lch)
:author: Alex Izvorski (yuv2rgb, rgb2yuv and related)

:license: modified BSD

References
----------
.. [1] Official specification of sRGB, IEC 61966-2-1:1999.
.. [2] http://www.poynton.com/ColorFAQ.html
.. [3] https://en.wikipedia.org/wiki/HSL_and_HSV
.. [4] https://en.wikipedia.org/wiki/CIE_1931_color_space
"""

from warnings import warn

import cupy as cp
import numpy as np
from scipy import linalg

from .._shared.utils import (
    _supported_float_type,
    channel_as_last_axis,
    deprecate_func,
    identity,
)
from ..util import dtype, dtype_limits

# TODO: when minimum numpy dependency is 1.25 use:
# np..exceptions.AxisError instead of AxisError
# and remove this try-except
try:
    from numpy import AxisError
except ImportError:
    from numpy.exceptions import AxisError


def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
    """Convert an image array to a new color space.

    Valid color spaces are:
        'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'

    Parameters
    ----------
    arr : (..., 3, ...) array_like
        The image to convert. By default, the final dimension denotes
        channels.
    fromspace : str
        The color space to convert from. Can be specified in lower case.
    tospace : str
        The color space to convert to. Can be specified in lower case.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The converted image. Same dimensions as input.

    Raises
    ------
    ValueError
        If fromspace is not a valid color space
    ValueError
        If tospace is not a valid color space

    Notes
    -----
    Conversion is performed through the "central" RGB color space,
    i.e. conversion from XYZ to HSV is implemented as ``XYZ -> RGB -> HSV``
    instead of directly.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())
    >>> img_hsv = convert_colorspace(img, 'RGB', 'HSV')
    """
    fromdict = {
        "rgb": identity,
        "hsv": hsv2rgb,
        "rgb cie": rgbcie2rgb,
        "xyz": xyz2rgb,
        "yuv": yuv2rgb,
        "yiq": yiq2rgb,
        "ypbpr": ypbpr2rgb,
        "ycbcr": ycbcr2rgb,
        "ydbdr": ydbdr2rgb,
    }
    todict = {
        "rgb": identity,
        "hsv": rgb2hsv,
        "rgb cie": rgb2rgbcie,
        "xyz": rgb2xyz,
        "yuv": rgb2yuv,
        "yiq": rgb2yiq,
        "ypbpr": rgb2ypbpr,
        "ycbcr": rgb2ycbcr,
        "ydbdr": rgb2ydbdr,
    }

    fromspace = fromspace.lower()
    tospace = tospace.lower()
    if fromspace not in fromdict:
        msg = f"`fromspace` has to be one of {fromdict.keys()}"
        raise ValueError(msg)
    if tospace not in todict:
        msg = f"`tospace` has to be one of {todict.keys()}"
        raise ValueError(msg)

    return todict[tospace](
        fromdict[fromspace](arr, channel_axis=channel_axis),
        channel_axis=channel_axis,
    )


def _prepare_colorarray(
    arr, force_copy=False, force_c_contiguous=True, channel_axis=-1
):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    if arr.shape[channel_axis] != 3:
        msg = (
            f"the input array must have size 3 along `channel_axis`, "
            f"got {arr.shape}"
        )
        raise ValueError(msg)
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == cp.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    out = _func(arr, force_copy=force_copy)
    if force_c_contiguous and not out.flags.c_contiguous:
        out = cp.ascontiguousarray(out)
    return out


def _validate_channel_axis(channel_axis, ndim):
    if not isinstance(channel_axis, int):
        raise TypeError("channel_axis must be an integer")
    if channel_axis < -ndim or channel_axis >= ndim:
        raise AxisError("channel_axis exceeds array dimensions")


@cp.memoize(for_each_device=True)
def _rgba2rgb_kernel(background, name="rgba2rgb"):
    code = """
    X alpha = rgba[4*i + 3];
    X val;
    """
    for ch in range(3):
        code += f"""
        val = (1 - alpha) * {background[ch]} + alpha * rgba[4*i + {ch}];
        rgb[3*i + {ch}] = min(max(val, (X)0.0), (X)1.0);
        """
    return cp.ElementwiseKernel(
        "raw X rgba", "raw X rgb", code, name="cucim_skimage_color_" + name
    )


@channel_as_last_axis()  # current CUDA kernel assumes channel_axis is last
def rgba2rgb(rgba, background=(1, 1, 1), *, channel_axis=-1):
    """RGBA to RGB conversion using alpha blending [1]_.

    Parameters
    ----------
    rgba : (..., 4, ...) array_like
        The image in RGBA format. By default, the final dimension denotes
        channels.
    background : array_like
        The color of the background to blend the image with (3 floats
        between 0 to 1 - the RGB value of the background).
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgba` is not at least 2D with shape (..., 4, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import color
    >>> from skimage import data
    >>> img_rgba = cp.array(data.logo())
    >>> img_rgb = color.rgba2rgb(img_rgba)
    """
    _validate_channel_axis(channel_axis, rgba.ndim)
    channel_axis = channel_axis % rgba.ndim

    if rgba.shape[channel_axis] != 4:
        msg = (
            f"the input array must have size 4 along `channel_axis`, "
            f"got {rgba.shape}"
        )
        raise ValueError(msg)

    float_dtype = _supported_float_type(rgba.dtype)
    if float_dtype == cp.float32:
        rgba = dtype.img_as_float32(rgba)
    else:
        rgba = dtype.img_as_float64(rgba)
    if not rgba.flags.c_contiguous:
        rgba = cp.ascontiguousarray(rgba)

    if isinstance(background, cp.ndarray):
        background = cp.asnumpy(background)  # synchronize
    background = tuple(float(b) for b in background)
    if len(background) != 3:
        raise ValueError(
            "background must be an array-like containing 3 RGB "
            f"values. Got {len(background)} items"
        )
    if any((b < 0 or b > 1) for b in background):
        raise ValueError(
            "background RGB values must be floats between " "0 and 1."
        )

    name = f"rgba2rgb_{rgba.dtype.char}"
    kern = _rgba2rgb_kernel(background, name)
    rgb = cp.empty(rgba.shape[:-1] + (3,), dtype=rgba.dtype)
    kern(rgba, rgb, size=rgb.size // 3)
    return rgb


@cp.memoize(for_each_device=True)
def _rgb_to_hsv_kernel(name="rgb2hsv"):
    code = """
    X minv = rgb[3*i];
    X maxv = rgb[3*i];
    X tmp;
    for (int ch=1; ch < 3; ch++)
    {
        tmp = rgb[3*i + ch];
        if (tmp > maxv)
        {
            maxv = tmp;
        } else if (tmp < minv)
        {
            minv = tmp;
        }
    }
    X delta = maxv - minv;
    if (delta == 0.0)
    {
        hsv[3*i] = 0.0;
        hsv[3*i + 1] = 0.0;
    } else {
        hsv[3*i + 1] = delta / maxv;
        if (rgb[3*i] == maxv)
        {
           hsv[3*i]  = (rgb[3*i + 1] - rgb[3*i + 2]) / delta;
        } else if (rgb[3*i + 1] == maxv)
        {
           hsv[3*i]  = 2.0 + (rgb[3*i + 2] - rgb[3*i]) / delta;
        } else if (rgb[3*i + 2] == maxv)
        {
           hsv[3*i]  = 4.0 + (rgb[3*i] - rgb[3*i + 1]) / delta;
        }
        hsv[3*i] /= 6.0;
        hsv[3*i] = hsv[3*i] - floor(hsv[3*i] / (X)1.0);
    }
    hsv[3*i + 2] = maxv;
    """
    return cp.ElementwiseKernel(
        "raw X rgb", "raw X hsv", code, name="cucim_skimage_color_" + name
    )


@channel_as_last_axis()
def rgb2hsv(rgb, *, channel_axis=-1):
    """RGB to HSV color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in HSV format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import color
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())
    >>> img_hsv = color.rgb2hsv(img)
    """
    input_is_one_pixel = rgb.ndim == 1
    if input_is_one_pixel:
        rgb = rgb[np.newaxis, ...]

    rgb = _prepare_colorarray(
        rgb, force_c_contiguous=True, channel_axis=channel_axis
    )
    hsv = cp.empty_like(rgb)

    name = f"rgb2hsv_{rgb.dtype.char}"
    kern = _rgb_to_hsv_kernel(name=name)
    kern(rgb, hsv, size=rgb.size // 3)

    if input_is_one_pixel:
        hsv = cp.squeeze(hsv, axis=0)

    return hsv


@cp.memoize(for_each_device=True)
def _hsv_to_rgb_kernel(name="hsv2rgb"):
    code = """
    int hi = (int)floor(hsv[3*i] * 6.0);

    X f = hsv[3*i] * 6 - hi;
    X v = hsv[3*i + 2];
    X p = v * (1 - hsv[3*i + 1]);

    int rem = (int)hi % 6;
    switch(rem)
    {
        case 0:
            rgb[3*i] = v;
            rgb[3*i + 1] = v * (1 - (1 - f) * hsv[3*i + 1]);
            rgb[3*i + 2] = p;
            break;
        case 1:
            rgb[3*i] = v * (1 - f * hsv[3*i + 1]);
            rgb[3*i + 1] = v;
            rgb[3*i + 2] = p;
            break;
        case 2:
            rgb[3*i] = p;
            rgb[3*i + 1] = v;
            rgb[3*i + 2] = v * (1 - (1 - f) * hsv[3*i + 1]);
            break;
        case 3:
            rgb[3*i] = p;
            rgb[3*i + 1] = v * (1 - f * hsv[3*i + 1]);
            rgb[3*i + 2] = v;
            break;
        case 4:
            rgb[3*i] = v * (1 - (1 - f) * hsv[3*i + 1]);
            rgb[3*i + 1] = p;
            rgb[3*i + 2] = v;
            break;
        case 5:
            rgb[3*i] = v;
            rgb[3*i + 1] = p;
            rgb[3*i + 2] = v * (1 - f * hsv[3*i + 1]);
            break;
    }
    """
    return cp.ElementwiseKernel(
        "raw X hsv", "raw X rgb", code, name="cucim_skimage_color_" + name
    )


@channel_as_last_axis()
def hsv2rgb(hsv, *, channel_axis=-1):
    """HSV to RGB color space conversion.

    Parameters
    ----------
    hsv : (..., 3, ...) array_like
        The image in HSV format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `hsv` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())
    >>> img_hsv = rgb2hsv(img)
    >>> img_rgb = hsv2rgb(img_hsv)
    """
    hsv = _prepare_colorarray(
        hsv, force_c_contiguous=True, channel_axis=channel_axis
    )

    rgb = cp.empty_like(hsv)

    name = f"hsv2rgb_{hsv.dtype.char}"
    kern = _hsv_to_rgb_kernel(name=name)
    kern(hsv, rgb, size=hsv.size // 3)
    return rgb


# ---------------------------------------------------------------
# Primaries for the coordinate systems
# ---------------------------------------------------------------
cie_primaries = np.array([700, 546.1, 435.8])
sb_primaries = np.array([1.0 / 155, 1.0 / 190, 1.0 / 225]) * 1e5

# ---------------------------------------------------------------
# Matrices that define conversion between different color spaces
# ---------------------------------------------------------------

# From sRGB specification
# fmt: off
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = linalg.inv(xyz_from_rgb)

# From https://en.wikipedia.org/wiki/CIE_1931_color_space
# Note: Travis's code did not have the divide by 0.17697
xyz_from_rgbcie = np.array([[0.49, 0.31, 0.20],
                            [0.17697, 0.81240, 0.01063],
                            [0.00, 0.01, 0.99]]) / 0.17697

rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)

# construct matrices to and from rgb:
rgbcie_from_rgb = rgbcie_from_xyz @ xyz_from_rgb
rgb_from_rgbcie = rgb_from_xyz @ xyz_from_rgbcie


gray_from_rgb = np.array([[0.2125, 0.7154, 0.0721],
                          [0, 0, 0],
                          [0, 0, 0]])

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114     ],   # noqa
                         [-0.14714119, -0.28886916,  0.43601035],   # noqa
                         [ 0.61497538, -0.51496512, -0.10001026]])  # noqa

rgb_from_yuv = linalg.inv(yuv_from_rgb)

yiq_from_rgb = np.array([[0.299     ,  0.587     ,  0.114     ],   # noqa
                         [0.59590059, -0.27455667, -0.32134392],   # noqa
                         [0.21153661, -0.52273617,  0.31119955]])  # noqa

rgb_from_yiq = linalg.inv(yiq_from_rgb)


ypbpr_from_rgb = np.array([[ 0.299   ,  0.587   ,  0.114   ],   # noqa
                           [-0.168736, -0.331264,  0.5     ],   # noqa
                           [ 0.5     , -0.418688, -0.081312]])  # noqa
# fmt: on

rgb_from_ypbpr = linalg.inv(ypbpr_from_rgb)

ycbcr_from_rgb = np.array(
    [
        [65.481, 128.553, 24.966],  # noqa
        [-37.797, -74.203, 112.0],  # noqa
        [112.0, -93.786, -18.214],
    ]
)  # noqa

rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)

ydbdr_from_rgb = np.array(
    [
        [0.299, 0.587, 0.114],  # noqa
        [-0.45, -0.883, 1.333],  # noqa
        [-1.333, 1.116, 0.217],
    ]
)  # noqa

rgb_from_ydbdr = linalg.inv(ydbdr_from_rgb)


# CIE LAB constants for Observer=2A, Illuminant=D65
# NOTE: this is actually the XYZ values for the illuminant above.
lab_ref_white = np.array([0.95047, 1.0, 1.08883])

# XYZ coordinates of the illuminants, scaled to [0, 1]. For each illuminant I
# we have:
#
#   illuminant[I]['2'] corresponds to the XYZ coordinates for the 2 degree
#   field of view.
#
#   illuminant[I]['10'] corresponds to the XYZ coordinates for the 10 degree
#   field of view.
#
#   illuminant[I]['R'] corresponds to the XYZ coordinates for R illuminants
#   in grDevices::convertColor
#
# The XYZ coordinates are calculated from [1], using the formula:
#
#   X = x * ( Y / y )
#   Y = Y
#   Z = ( 1 - x - y ) * ( Y / y )
#
# where Y = 1. The only exception is the illuminant "D65" with aperture angle
# 2, whose coordinates are copied from 'lab_ref_white' for
# backward-compatibility reasons.
#
#     References
#    ----------
#    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant

_illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
        "R": (1.098466069456375, 1, 0.3558228003436005),
    },
    "B": {
        "2": (0.9909274480248003, 1, 0.8531327322886154),
        "10": (0.9917777147717607, 1, 0.8434930535866175),
        "R": (0.9909274480248003, 1, 0.8531327322886154),
    },
    "C": {
        "2": (0.980705971659919, 1, 1.1822494939271255),
        "10": (0.9728569189782166, 1, 1.1614480488951577),
        "R": (0.980705971659919, 1, 1.1822494939271255),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
        "R": (0.9639501491621826, 1, 0.8241280285499208),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
        "R": (0.9565317453467969, 1, 0.9202554587037198),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),  # This was: `lab_ref_white`
        "10": (0.94809667673716, 1, 1.0730513595166162),
        "R": (0.9532057125493769, 1, 1.0853843816469158),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
        "R": (0.9497220898840717, 1, 1.226393520724154),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0), "R": (1.0, 1.0, 1.0)},
}


def xyz_tristimulus_values(*, illuminant, observer, dtype=None):
    """Get the CIE XYZ tristimulus values.

    Given an illuminant and observer, this function returns the CIE XYZ
    tristimulus values [2]_ scaled such that :math:`Y = 1`.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function ``grDevices::convertColor`` [3]_.
    dtype : np.dtype, optional
        This argument is ignored in the cuCIM implementation of
        `xyz_tristimulus_values` since an array is not returned. The output is
        always a 3-tuple of float.

    Returns
    -------
    values : 3-tuple of float
        Three elements :math:`X, Y, Z` containing the CIE XYZ tristimulus values
        of the given illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
    .. [2] https://en.wikipedia.org/wiki/CIE_1931_color_space#Meaning_of_X,_Y_and_Z
    .. [3] https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/convertColor

    Notes
    -----
    The return type of this function differs from the one in scikit-image as it
    always returns a 3-tuple of float rather than an array with a
    user-specified dtype.

    The CIE XYZ tristimulus values are calculated from :math:`x, y` [1]_, using the
    formula

    .. math:: X = x / y

    .. math:: Y = 1

    .. math:: Z = (1 - x - y) / y

    The only exception is the illuminant "D65" with aperture angle 2° for
    backward-compatibility reasons.

    Examples
    --------
    Get the CIE XYZ tristimulus values for a "D65" illuminant for a 10 degree
    field of view

    >>> xyz_tristimulus_values(illuminant="D65", observer="10")
    array([0.94809668, 1.        , 1.07305136])
    """  # noqa
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return _illuminants[illuminant][observer]
    except KeyError:
        raise ValueError(
            f"Unknown illuminant/observer combination "
            f"(`{illuminant}`, `{observer}`)"
        )


@deprecate_func(
    hint="Use `skimage.color.xyz_tristimulus_values` instead.",
    deprecated_version="23.08",
    removed_version="24.06",
)
def get_xyz_coords(illuminant, observer, dtype=float):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    dtype: dtype, optional
        Output data type.

    Returns
    -------
    out : array
        Array with 3 elements containing the XYZ coordinates of the given
        illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    return xyz_tristimulus_values(illuminant=illuminant, observer=observer)


# Haematoxylin-Eosin-DAB colorspace
# From original Ruifrok's paper: A. C. Ruifrok and D. A. Johnston,
# "Quantification of histochemical staining by color deconvolution,"
# Analytical and quantitative cytology and histology / the International
# Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4,
# pp. 291-9, Aug. 2001.
# fmt: off


rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)

# Following matrices are adapted form the Java code written by G.Landini.
# The original code is available at:
# https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html

# Hematoxylin + DAB
rgb_from_hdx = np.array([[0.650, 0.704, 0.286],
                         [0.268, 0.570, 0.776],
                         [0.0, 0.0, 0.0]])
rgb_from_hdx[2, :] = np.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
hdx_from_rgb = linalg.inv(rgb_from_hdx)

# Feulgen + Light Green
rgb_from_fgx = np.array([[0.46420921, 0.83008335, 0.30827187],
                         [0.94705542, 0.25373821, 0.19650764],
                         [0.0, 0.0, 0.0]])
rgb_from_fgx[2, :] = np.cross(rgb_from_fgx[0, :], rgb_from_fgx[1, :])
fgx_from_rgb = linalg.inv(rgb_from_fgx)

# Giemsa: Methyl Blue + Eosin
rgb_from_bex = np.array([[0.834750233, 0.513556283, 0.196330403],
                         [0.092789, 0.954111, 0.283111],
                         [0.0, 0.0, 0.0]])
rgb_from_bex[2, :] = np.cross(rgb_from_bex[0, :], rgb_from_bex[1, :])
bex_from_rgb = linalg.inv(rgb_from_bex)

# FastRed + FastBlue +  DAB
rgb_from_rbd = np.array([[0.21393921, 0.85112669, 0.47794022],
                         [0.74890292, 0.60624161, 0.26731082],
                         [0.268, 0.570, 0.776]])
rbd_from_rgb = linalg.inv(rgb_from_rbd)

# Methyl Green + DAB
rgb_from_gdx = np.array([[0.98003, 0.144316, 0.133146],
                         [0.268, 0.570, 0.776],
                         [0.0, 0.0, 0.0]])
rgb_from_gdx[2, :] = np.cross(rgb_from_gdx[0, :], rgb_from_gdx[1, :])
gdx_from_rgb = linalg.inv(rgb_from_gdx)

# Hematoxylin + AEC
rgb_from_hax = np.array([[0.650, 0.704, 0.286],
                         [0.2743, 0.6796, 0.6803],
                         [0.0, 0.0, 0.0]])
rgb_from_hax[2, :] = np.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
hax_from_rgb = linalg.inv(rgb_from_hax)

# Blue matrix Anilline Blue + Red matrix Azocarmine + Orange matrix Orange-G
rgb_from_bro = np.array([[0.853033, 0.508733, 0.112656],
                         [0.09289875, 0.8662008, 0.49098468],
                         [0.10732849, 0.36765403, 0.9237484]])
bro_from_rgb = linalg.inv(rgb_from_bro)

# Methyl Blue + Ponceau Fuchsin
rgb_from_bpx = np.array([[0.7995107, 0.5913521, 0.10528667],
                         [0.09997159, 0.73738605, 0.6680326],
                         [0.0, 0.0, 0.0]])
rgb_from_bpx[2, :] = np.cross(rgb_from_bpx[0, :], rgb_from_bpx[1, :])
bpx_from_rgb = linalg.inv(rgb_from_bpx)

# Alcian Blue + Hematoxylin
rgb_from_ahx = np.array([[0.874622, 0.457711, 0.158256],
                         [0.552556, 0.7544, 0.353744],
                         [0.0, 0.0, 0.0]])
rgb_from_ahx[2, :] = np.cross(rgb_from_ahx[0, :], rgb_from_ahx[1, :])
ahx_from_rgb = linalg.inv(rgb_from_ahx)

# Hematoxylin + PAS
rgb_from_hpx = np.array([[0.644211, 0.716556, 0.266844],
                         [0.175411, 0.972178, 0.154589],
                         [0.0, 0.0, 0.0]])
rgb_from_hpx[2, :] = np.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
hpx_from_rgb = linalg.inv(rgb_from_hpx)
# fmt on

# -------------------------------------------------------------
# The conversion functions that make use of the matrices above
# -------------------------------------------------------------


@cp.memoize(for_each_device=True)
def _get_convert_kernel(matrix_tuple, pre, post, name):
    # pre code may modify x so set both x and y as outputs
    return cp.ElementwiseKernel(
        '',
        'raw X x, raw X y',
        pre + _get_core_colorconv_operation(matrix_tuple) + post,
        name='cucim_skimage_color_' + name)


def _convert(matrix, arr, pre='', post='', name='_convert'):
    """Do the color space conversion.

    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : (..., 3) array_like
        The input array. Final dimension denotes channels.

    Returns
    -------
    out : (..., 3) ndarray
        The converted array. Same dimensions as input.
    """
    arr = _prepare_colorarray(arr)
    name = name + f'_{arr.dtype.char}'
    kern = _get_convert_kernel(tuple(matrix.ravel()), pre, post, name)
    out = cp.empty_like(arr)
    kern(arr, out, size=arr.size // 3)
    return out


def _get_core_colorconv_operation(m):
    """Generate inline CUDA kernel code for color conversions.

    x is the input image with 3 channels on the last axis
    y is the output image with 3 channels on the last axis
    m is a 3x3 color conversion matrix
    """
    return f"""
        y[3*i] = x[3*i] * {m[0]} + x[3*i + 1] * {m[1]} + x[3*i + 2] * {m[2]};
        y[3*i + 1] = x[3*i] * {m[3]} + x[3*i + 1] * {m[4]} + x[3*i + 2] * {m[5]};
        y[3*i + 2] = x[3*i] * {m[6]} + x[3*i + 1] * {m[7]} + x[3*i + 2] * {m[8]};
    """  # noqa


@channel_as_last_axis()
def xyz2rgb(xyz, *, channel_axis=-1):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2xyz, xyz2rgb
    >>> img = cp.array(data.astronaut())
    >>> img_xyz = rgb2xyz(img)
    >>> img_rgb = xyz2rgb(img_xyz)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _prepare_colorarray(xyz, force_c_contiguous=True,
                              channel_axis=channel_axis)

    # scaling applied after the 3x3 conversion matrix multiplication
    # (c indexes over color channels here)
    _post_colorconv = """
        for (int c=0; c < 3; c++) {
            if (y[3*i + c] > 0.0031308) {
                y[3*i + c] = 1.055 * pow(y[3*i + c], (X)(1 / 2.4)) - 0.055;
            } else {
                y[3*i + c] *= 12.92;
            }
            y[3*i + c] = min(max(y[3*i + c], (X)0.0), (X)1.0);
        }
    """
    return _convert(rgb_from_xyz, arr, post=_post_colorconv, name='xyz2rgb')


@channel_as_last_axis()
def rgb2xyz(rgb, *, channel_axis=-1):
    """RGB to XYZ color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())
    >>> img_xyz = rgb2xyz(img)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    rgb = _prepare_colorarray(rgb, force_copy=True, force_c_contiguous=True,
                              channel_axis=channel_axis)

    # scaling applied to the input before 3x3 conversion matrix multiplication
    # (c indexes over color channels here)
    _pre_colorconv = """
        for (int c=0; c < 3; c++) {
            if (x[3*i + c] > 0.04045) {
                x[3*i + c] = pow((x[3*i + c] + (X)0.055) / (X)1.055, (X)2.4);
            } else {
                x[3*i + c] /= 12.92;
            }
        }
    """
    return _convert(xyz_from_rgb, rgb, pre=_pre_colorconv, name='rgb2xyz')


@channel_as_last_axis()
def rgb2rgbcie(rgb, *, channel_axis=-1):
    """RGB to RGB CIE color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB CIE format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2rgbcie
    >>> img = cp.array(data.astronaut())
    >>> img_rgbcie = rgb2rgbcie(img)
    """
    return _convert(rgbcie_from_rgb, rgb, name='rgb2rgbcie')


@channel_as_last_axis()
def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    """RGB CIE to RGB color space conversion.

    Parameters
    ----------
    rgbcie : (..., 3, ...) array_like
        The image in RGB CIE format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgbcie` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2rgbcie, rgbcie2rgb
    >>> img = cp.array(data.astronaut())
    >>> img_rgbcie = rgb2rgbcie(img)
    >>> img_rgb = rgbcie2rgb(img_rgbcie)
    """
    return _convert(rgb_from_rgbcie, rgbcie, name='rgbcie2rgb')


@cp.memoize(for_each_device=True)
def _rgb_to_gray_kernel(dtype):
    return cp.ElementwiseKernel(
        'raw X rgb',
        'raw X gray',
        """
        gray[i] = 0.2125 * rgb[3*i] + 0.7154 * rgb[3*i + 1] + 0.0721 * rgb[3*i + 2];
        """,  # noqa
        name=f'cucim_skimage_color_rgb2gray_{np.dtype(dtype).char}')


@channel_as_last_axis(multichannel_output=False)
def rgb2gray(rgb, *, channel_axis=-1):
    """Compute luminance of an RGB image.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : ndarray
        The luminance image - an array which is the same size as the input
        array, but with the channel dimension removed.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    The weights used in this conversion are calibrated for contemporary
    CRT phosphors::

        Y = 0.2125 R + 0.7154 G + 0.0721 B

    If there is an alpha channel present, it is ignored.

    References
    ----------
    .. [1] http://poynton.ca/PDFs/ColorFAQ.pdf

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.color import rgb2gray
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())
    >>> img_gray = rgb2gray(img)
    """
    rgb = _prepare_colorarray(rgb, force_c_contiguous=True,
                              channel_axis=channel_axis)
    kern = _rgb_to_gray_kernel(rgb.dtype)
    gray = cp.empty(rgb.shape[:-1], dtype=rgb.dtype)
    kern(rgb, gray, size=gray.size)
    return gray


def gray2rgba(image, alpha=None, *, channel_axis=-1):
    """Create a RGBA representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    alpha : array_like, optional
        Alpha channel of the output image. It may be a scalar or an
        array that can be broadcast to ``image``. If not specified it is
        set to the maximum limit corresponding to the ``image`` dtype.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels.

    Returns
    -------
    rgba : ndarray
        RGBA image. A new dimension of length 4 is added to input
        image shape.
    """

    alpha_min, alpha_max = dtype_limits(image, clip_negative=False)

    if alpha is None:
        alpha = alpha_max

    if not cp.can_cast(alpha, image.dtype):
        warn("alpha can't be safely cast to image dtype {}"
             .format(image.dtype.name), stacklevel=2)

    if np.isscalar(alpha):
        alpha = cp.full(image.shape, alpha, dtype=image.dtype)
    elif alpha.shape != image.shape:
        raise ValueError("alpha.shape must match image.shape")
    rgba = np.stack((image,) * 3 + (alpha,), axis=channel_axis)
    return rgba


def gray2rgb(image, *, channel_axis=-1):
    """Create an RGB representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels.

    Returns
    -------
    rgb : (..., 3, ...) ndarray
        RGB image. A new dimension of length 3 is added to input image.

    Notes
    -----
    If the input is a 1-dimensional image of shape ``(M, )``, the output
    will be shape ``(M, 3)``.
    """
    return cp.stack(3 * (image,), axis=channel_axis)


@cp.memoize(for_each_device=True)
def _get_xyz_to_lab_kernel(xyz_ref_white, name='xyz2lab'):
    _xyz_to_lab = f"""
        // scale by CIE XYZ tristimulus values of the reference white point
        arr[3*i] /= {xyz_ref_white[0]};
        arr[3*i + 1] /= {xyz_ref_white[1]};
        arr[3*i + 2] /= {xyz_ref_white[2]};

        // Nonlinear distortion and linear transformation
        for (int ch=0; ch < 3; ch++)
        {{
            if (arr[3*i + ch] > 0.008856)
            {{
                arr[3*i + ch] = cbrt(arr[3*i + ch]);
            }} else {{
                arr[3*i + ch] = 7.787 * arr[3*i + ch] + 16.0 / 116.0;
            }}
        }}

        // Vector scaling
        lab[3*i] = (116. * arr[3*i + 1]) - 16.0;
        lab[3*i + 1] = 500.0 * (arr[3*i] - arr[3*i + 1]);
        lab[3*i + 2] = 200.0 * (arr[3*i + 1] - arr[3*i + 2]);
    """

    # array will be modified in-place
    return cp.ElementwiseKernel(
        '',
        'raw X arr, raw X lab',
        _xyz_to_lab,
        name='cucim_skimage_color_' + name)


@channel_as_last_axis()
def xyz2lab(xyz, illuminant="D65", observer="2", *, channel_axis=-1):
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE-LAB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.

    Notes
    -----
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2xyz, xyz2lab
    >>> img = cp.array(data.astronaut())
    >>> img_xyz = rgb2xyz(img)
    >>> img_lab = xyz2lab(img_xyz)
    """
    xyz = _prepare_colorarray(xyz, force_copy=True, force_c_contiguous=True,
                              channel_axis=channel_axis)

    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer
    )
    name = f'xyz2lab_{xyz.dtype.char}'
    kern = _get_xyz_to_lab_kernel(xyz_ref_white, name=name)
    lab = cp.empty_like(xyz)
    kern(xyz, lab, size=lab.size // 3)
    return lab


@cp.memoize(for_each_device=True)
def _get_lab_to_xyz_kernel(xyz_ref_white, name='lab2xyz'):
    _lab_to_xyz = f"""

        xyz[3*i + 1] = (lab[3*i] + 16.) / 116.;
        xyz[3*i] = (lab[3*i + 1] / 500.0) + xyz[3*i + 1];
        xyz[3*i + 2] = xyz[3*i + 1] - (lab[3*i + 2] /200.0);
        if (xyz[3*i + 2] < 0.0)
        {{
            xyz[3*i + 2] = 0.0;
            warn[i] = 1;
        }}

        for (int ch=0; ch < 3; ch++)
        {{
            if (xyz[3*i + ch] > 0.2068966)
            {{
                xyz[3*i + ch] *= xyz[3*i + ch] * xyz[3*i + ch];
            }} else {{
                xyz[3*i + ch] = (xyz[3*i + ch] - 16.0 / 116.0) / 7.787;
            }}
        }}

        xyz[3*i] *= {xyz_ref_white[0]};
        xyz[3*i + 1] *= {xyz_ref_white[1]};
        xyz[3*i + 2] *= {xyz_ref_white[2]};

        // xyz[3*i] = min(max(xyz[3*i], 0.0), 1.0);
        // xyz[3*i + 1] = min(max(xyz[3*i + 1], 0.0), 1.0);
        // xyz[3*i + 2] = min(max(xyz[3*i + 2], 0.0), 1.0);
    """

    # array will be modified in-place
    return cp.ElementwiseKernel(
        '',
        'raw X lab, raw X xyz, raw int32 warn',
        _lab_to_xyz,
        name='cucim_skimage_color_' + name)


@channel_as_last_axis()
def lab2xyz(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    """Convert image in CIE-LAB to XYZ color space.

    Parameters
    ----------
    lab : (..., 3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    UserWarning
        If any of the pixels are invalid (Z < 0).

    Notes
    -----
    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    See Also
    --------
    xyz2lab

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    xyz, n_invalid = _lab2xyz(lab, illuminant, observer, channel_axis)
    if n_invalid > 0:
        warn(
            "Conversion from CIE-LAB to XYZ color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )

    return xyz


def _lab2xyz(lab, illuminant, observer, channel_axis):
    """Convert CIE-LAB to XYZ color space.

    Internal function for :func:`~.lab2xyz` and others. In addition to the
    converted image, return the number of invalid pixels in the Z channel for
    correct warning propagation.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.
    n_invalid : int
        Number of invalid pixels in the Z channel after conversion.
    """
    lab = _prepare_colorarray(lab, force_c_contiguous=True,
                              channel_axis=channel_axis)

    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer
    )

    name = f'lab2xyz_{lab.dtype.char}'
    kern = _get_lab_to_xyz_kernel(xyz_ref_white, name=name)
    xyz = cp.empty_like(lab)

    # TODO: better to use array for warn or a single element with atomic
    #       operations?
    warnings = cp.zeros(lab.shape[:-1], dtype=np.int32)
    kern(lab, xyz, warnings, size=lab.size // 3)
    n_invalid = int(cp.count_nonzero(warnings))  # synchronize!
    return xyz, n_invalid


@channel_as_last_axis()
def rgb2lab(rgb, illuminant="D65", observer="2", *, channel_axis=-1):
    """Conversion from the sRGB color space (IEC 61966-2-1:1999)
    to the CIE Lab colorspace under the given illuminant and observer.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in Lab format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    RGB is a device-dependent color space so, if you use this function, be
    sure that the image you are analyzing has been mapped to the sRGB color
    space.

    This function uses rgb2xyz and xyz2lab.
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)


@channel_as_last_axis()
def lab2rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    """Convert image in CIE-LAB to sRGB color space.

    Parameters
    ----------
    lab : (..., 3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in sRGB color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.
    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    See Also
    --------
    rgb2lab

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    xyz, n_invalid = _lab2xyz(lab, illuminant, observer, channel_axis)
    if n_invalid != 0:
        warn(
            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )
    return xyz2rgb(xyz, channel_axis=channel_axis)


@cp.memoize(for_each_device=True)
def _get_xyz_to_luv_kernel(xyz_ref_white, dtype):

    eps = np.finfo(dtype).eps

    preamble = f"""
    // u' and v' helper functions

    static __device__ __inline__ X fu(X v0, X v1, X v2)
    {{
        return (4.0 * v0) / (v0 + 15.0 * v1 + 3.0 * v2 + {eps});
    }}

    static __device__ __inline__ X fv(X v0, X v1, X v2)
    {{
        return (9.0 * v1) / (v0 + 15.0 * v1 + 3.0 * v2 + {eps});
    }}
    """

    denom = np.asarray([1, 15, 3]) @ np.asarray(xyz_ref_white, dtype=float)
    denom = float(denom)
    u0 = 4 * xyz_ref_white[0] / denom
    v0 = 9 * xyz_ref_white[1] / denom

    _xyz_to_luv = f"""
        luv[3*i] = xyz[3*i + 1] / {xyz_ref_white[1]};
        if (luv[3*i] > 0.008856)
        {{
            luv[3*i] = 116.0 * cbrt(luv[3*i]) - 16.0;
        }} else {{
            luv[3*i] *= 903.3;
        }}

        luv[3*i + 1] = (
            13.0 * luv[3*i] * (fu(xyz[3*i], xyz[3*i + 1], xyz[3*i + 2]) - {u0})
        );
        luv[3*i + 2] = (
            13.0 * luv[3*i] * (fv(xyz[3*i], xyz[3*i + 1], xyz[3*i + 2]) - {v0})
        );

    """

    # array will be modified in-place
    return cp.ElementwiseKernel(
        '',
        'raw X xyz, raw X luv',
        _xyz_to_luv,
        preamble=preamble,
        name=f'cucim_skimage_color_xyz2luv_{np.dtype(dtype).char}')


@channel_as_last_axis()
def xyz2luv(xyz, illuminant="D65", observer="2", *, channel_axis=-1):
    """XYZ to CIE-Luv color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE-Luv format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    Notes
    -----
    By default XYZ conversion weights use observer=2A. Reference whitepoint
    for D65 Illuminant, with XYZ tristimulus values of ``(95.047, 100.,
    108.883)``. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELUV

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2xyz, xyz2luv
    >>> img = cp.array(data.astronaut())
    >>> img_xyz = rgb2xyz(img)
    >>> img_luv = xyz2luv(img_xyz)
    """
    input_is_one_pixel = xyz.ndim == 1
    if input_is_one_pixel:
        xyz = xyz[np.newaxis, ...]

    xyz = _prepare_colorarray(xyz, force_c_contiguous=True,
                              channel_axis=channel_axis)

    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer
    )
    kern = _get_xyz_to_luv_kernel(xyz_ref_white, xyz.dtype)
    luv = cp.empty_like(xyz)
    kern(xyz, luv, size=xyz.size // 3)

    if input_is_one_pixel:
        luv = cp.squeeze(luv, axis=0)

    return luv


@cp.memoize(for_each_device=True)
def _get_luv_to_xyz_kernel(xyz_ref_white, dtype):

    eps = np.finfo(dtype).eps

    denom = np.asarray([1, 15, 3]) @ np.asarray(xyz_ref_white, dtype=float)
    denom = float(denom)
    u0 = 4 * xyz_ref_white[0] / denom
    v0 = 9 * xyz_ref_white[1] / denom

    _luv_to_xyz = f"""
        if (luv[3*i] > 7.999625)
        {{
            xyz[3*i + 1] = (luv[3 * i] + 16.0) / 116.0;
            xyz[3*i + 1] *= xyz[3*i + 1] * xyz[3*i + 1];
        }} else {{
            xyz[3*i + 1] = luv[3*i] / 903.3;
        }}
        xyz[3*i + 1] *= {xyz_ref_white[1]};

        X a = {u0} + luv[3*i + 1] / (13.0 * luv[3*i] + {eps});
        X b = {v0} + luv[3*i + 2] / (13.0 * luv[3*i] + {eps});
        X c = 3.0 * xyz[3*i + 1] * (5.0 * b - 3.0);

        xyz[3*i + 2] = ((a - 4.0) * c - 15.0 * a * b * xyz[3*i + 1]) / (12.0 * b);
        xyz[3*i] = -(c / b + 3.0 * xyz[3*i + 2]);

    """  # noqa
    return cp.ElementwiseKernel(
        '',
        'raw X luv, raw X xyz',
        _luv_to_xyz,
        name=f'cucim_skimage_color_luv2xyz_{np.dtype(dtype).char}')


@channel_as_last_axis()
def luv2xyz(luv, illuminant="D65", observer="2", *, channel_axis=-1):
    """CIE-Luv to XYZ color space conversion.

    Parameters
    ----------
    luv : (..., 3, ...) array_like
        The image in CIE-Luv format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `luv` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    Notes
    -----
    XYZ conversion weights use observer=2A. Reference whitepoint for D65
    Illuminant, with XYZ tristimulus values of ``(95.047, 100., 108.883)``. See
    function :func:`~.xyz_tristimulus_values` for a list of supported
    illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELUV
    """
    luv = _prepare_colorarray(luv, force_c_contiguous=True,
                              channel_axis=channel_axis)
    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer
    )
    kern = _get_luv_to_xyz_kernel(xyz_ref_white, luv.dtype)
    xyz = cp.empty_like(luv)
    kern(luv, xyz, size=luv.size // 3)
    return xyz


@channel_as_last_axis()
def rgb2luv(rgb, *, channel_axis=-1):
    """RGB to CIE-Luv color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE Luv format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    This function uses rgb2xyz and xyz2luv.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELUV
    """
    return xyz2luv(rgb2xyz(rgb))


@channel_as_last_axis()
def luv2rgb(luv, *, channel_axis=-1):
    """Luv to RGB color space conversion.

    Parameters
    ----------
    luv : (..., 3, ...) array_like
        The image in CIE Luv format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `luv` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    This function uses luv2xyz and xyz2rgb.
    """
    return xyz2rgb(luv2xyz(luv))


@channel_as_last_axis()
def rgb2hed(rgb, *, channel_axis=-1):
    """RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in HED format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2hed
    >>> ihc = cp.array(data.immunohistochemistry())
    >>> ihc_hed = rgb2hed(ihc)
    """
    return separate_stains(rgb, hed_from_rgb)


@channel_as_last_axis()
def hed2rgb(hed, *, channel_axis=-1):
    """Haematoxylin-Eosin-DAB (HED) to RGB color space conversion.

    Parameters
    ----------
    hed : (..., 3, ...) array_like
        The image in the HED color space. By default, the final dimension
        denotes channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB. Same dimensions as input.

    Raises
    ------
    ValueError
        If `hed` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
           staining by color deconvolution.," Analytical and quantitative
           cytology and histology / the International Academy of Cytology [and]
           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2hed, hed2rgb
    >>> ihc = cp.array(data.immunohistochemistry())
    >>> ihc_hed = rgb2hed(ihc)
    >>> ihc_rgb = hed2rgb(ihc_hed)
    """
    return combine_stains(hed, rgb_from_hed)


@cp.memoize(for_each_device=True)
def _separate_stains_kernel(m):
    log_adjust = 1 / np.log(1e-6)
    code = f"""
    X tmp[3];
    for (int ch=0; ch<3; ch++)
    {{
        tmp[ch] = log(max(rgb[3*i + ch], 1e-6)) * {log_adjust};
    }}
    stains[3*i] = tmp[0] * {m[0]} + tmp[1] * {m[3]} + tmp[2] * {m[6]};
    stains[3*i + 1] = tmp[0] * {m[1]} + tmp[1] * {m[4]} + tmp[2] * {m[7]};
    stains[3*i + 2] = tmp[0] * {m[2]} + tmp[1] * {m[5]} + tmp[2] * {m[8]};
    """  # noqa
    return cp.ElementwiseKernel(
        'raw X rgb',
        'raw X stains',
        code,
        name='cucim_skimage_color_seperate_stains')


@channel_as_last_axis()
def separate_stains(rgb, conv_matrix, *, channel_axis=-1):
    """RGB to stain color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in stain color space. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Stain separation matrices available in the ``color`` module and their
    respective colorspace:

    * ``hed_from_rgb``: Hematoxylin + Eosin + DAB
    * ``hdx_from_rgb``: Hematoxylin + DAB
    * ``fgx_from_rgb``: Feulgen + Light Green
    * ``bex_from_rgb``: Giemsa stain : Methyl Blue + Eosin
    * ``rbd_from_rgb``: FastRed + FastBlue +  DAB
    * ``gdx_from_rgb``: Methyl Green + DAB
    * ``hax_from_rgb``: Hematoxylin + AEC
    * ``bro_from_rgb``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``bpx_from_rgb``: Methyl Blue + Ponceau Fuchsin
    * ``ahx_from_rgb``: Alcian Blue + Hematoxylin
    * ``hpx_from_rgb``: Hematoxylin + PAS

    This implementation borrows some ideas from DIPlib [2]_, e.g. the
    compensation using a small value to avoid log artifacts when
    calculating the Beer-Lambert law.

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] https://github.com/DIPlib/diplib/
    .. [3] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import separate_stains, hdx_from_rgb
    >>> ihc = cp.array(data.immunohistochemistry())
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    """  # noqa
    rgb = _prepare_colorarray(rgb, force_c_contiguous=True,
                              channel_axis=channel_axis)

    if conv_matrix.shape != (3, 3):
        raise ValueError("conv_matrix must have shape (3, 3)")
    conv_matrix = tuple(cp.asnumpy(conv_matrix).ravel())

    # #cp.maximum(rgb, 1e-6, out=rgb)  # avoiding log artifacts
    # log_adjust = np.log(1e-6)  # used to compensate the sum above

    # conv_matrix = cp.asarray(conv_matrix, dtype=rgb.dtype)
    # stains = (cp.log(rgb) / log_adjust) @ conv_matrix

    kern = _separate_stains_kernel(conv_matrix)
    stains = cp.empty_like(rgb)
    kern(rgb, stains, size=rgb.size // 3)
    cp.maximum(stains, 0, out=stains)
    return stains


@cp.memoize(for_each_device=True)
def _combine_stains_kernel(m):
    # log_adjust here is used to compensate the sum within separate_stains()
    log_adjust = np.log(1e-6)
    code = f"""
    X tmp[3];
    for (int ch=0; ch<3; ch++)
    {{
        tmp[ch] = stains[3*i + ch] * {log_adjust};
    }}

    rgb[3*i] = tmp[0] * {m[0]} + tmp[1] * {m[3]} + tmp[2] * {m[6]};
    rgb[3*i + 1] = tmp[0] * {m[1]} + tmp[1] * {m[4]} + tmp[2] * {m[7]};
    rgb[3*i + 2] = tmp[0] * {m[2]} + tmp[1] * {m[5]} + tmp[2] * {m[8]};

    for (int ch=0; ch<3; ch++)
    {{
        rgb[3*i + ch] = min(max(exp(rgb[3*i + ch]), (X)0.0), (X)1.0);
    }}
    """  # noqa
    return cp.ElementwiseKernel(
        'raw X stains',
        'raw X rgb',
        code,
        name='cucim_skimage_color_combine_stains')


@channel_as_last_axis()
def combine_stains(stains, conv_matrix, *, channel_axis=-1):
    """Stain to RGB color space conversion.

    Parameters
    ----------
    stains : (..., 3, ...) array_like
        The image in stain color space. By default, the final dimension denotes
        channels.
    conv_matrix: ndarray
        The stain separation matrix as described by G. Landini [1]_.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `stains` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Stain combination matrices available in the ``color`` module and their
    respective colorspace:

    * ``rgb_from_hed``: Hematoxylin + Eosin + DAB
    * ``rgb_from_hdx``: Hematoxylin + DAB
    * ``rgb_from_fgx``: Feulgen + Light Green
    * ``rgb_from_bex``: Giemsa stain : Methyl Blue + Eosin
    * ``rgb_from_rbd``: FastRed + FastBlue +  DAB
    * ``rgb_from_gdx``: Methyl Green + DAB
    * ``rgb_from_hax``: Hematoxylin + AEC
    * ``rgb_from_bro``: Blue matrix Anilline Blue + Red matrix Azocarmine\
                        + Orange matrix Orange-G
    * ``rgb_from_bpx``: Methyl Blue + Ponceau Fuchsin
    * ``rgb_from_ahx``: Alcian Blue + Hematoxylin
    * ``rgb_from_hpx``: Hematoxylin + PAS

    References
    ----------
    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html
    .. [2] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical
           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.
           23, no. 4, pp. 291–299, Aug. 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.color import (separate_stains, combine_stains,
    ...                                    hdx_from_rgb, rgb_from_hdx)
    >>> ihc = cp.array(data.immunohistochemistry())
    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    >>> ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)
    """  # noqa
    stains = _prepare_colorarray(stains, force_c_contiguous=True,
                                 channel_axis=channel_axis)

    if conv_matrix.shape != (3, 3):
        raise ValueError("conv_matrix must have shape (3, 3)")
    conv_matrix = tuple(cp.asnumpy(conv_matrix).ravel())

    kern = _combine_stains_kernel(conv_matrix)
    rgb = cp.empty_like(stains)
    kern(stains, rgb, size=stains.size // 3)

    return rgb


@cp.memoize(for_each_device=True)
def _lab2lch_kernel(nchannels=3, name='lab2lch'):
    code = f"""
    X a = lab[{nchannels}*i + 1];
    X b = lab[{nchannels}*i + 2];

    // update lab array in-place with the lch values
    lab[{nchannels}*i + 1] = hypot(a, b);
    lab[{nchannels}*i + 2] = atan2(b, a);

    // NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than ``(-pi, +pi)``
    if (lab[{nchannels}*i + 2] < 0)
    {{
       lab[{nchannels}*i + 2] += 2 * M_PI;
    }}
    """  # noqa
    return cp.ElementwiseKernel(
        '',
        'raw X lab',
        code,
        name='cucim_skimage_color_' + name)


@channel_as_last_axis()
def lab2lch(lab, *, channel_axis=-1):
    """CIE-LAB to CIE-LCH color space conversion.

    LCH is the cylindrical representation of the LAB (Cartesian) colorspace

    Parameters
    ----------
    lab : (..., 3, ...) array_like
        The N-D image in CIE-LAB format. The last (``N+1``-th) dimension must
        have at least 3 elements, corresponding to the ``L``, ``a``, and ``b``
        color channels. Subsequent elements are copied.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in LCH format, in a N-D array with same shape as input `lab`.

    Raises
    ------
    ValueError
        If `lch` does not have at least 3 color channels (i.e. l, a, b).

    Notes
    -----
    The Hue is expressed as an angle between ``(0, 2*pi)``

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2lab, lab2lch
    >>> img = cp.array(data.astronaut())
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    """
    lab = _prepare_lab_array(lab, force_copy=True)
    nchannels = lab.shape[-1]

    name = f'lab2lch_{nchannels}channel_{lab.dtype}'
    kern = _lab2lch_kernel(nchannels, name=name)
    kern(lab, size=lab.size // nchannels)
    return lab


def _cart2polar_2pi(x, y):
    """convert cartesian coordinates to polar (uses non-standard theta range!)

    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
    """
    r, t = cp.hypot(x, y), cp.arctan2(y, x)
    t += cp.where(t < 0., 2 * np.pi, 0)
    return r, t


@cp.memoize(for_each_device=True)
def _lch2lab_kernel(nchannels=3, name='lch2lab'):
    code = f"""
    X sin_h = sin(lch[{nchannels}*i + 2]);
    X cos_h = cos(lch[{nchannels}*i + 2]);

    // update lch array in-place with the lab values
    lch[{nchannels}*i + 2] = lch[{nchannels}*i + 1] * sin_h;
    lch[{nchannels}*i + 1] = lch[{nchannels}*i + 1] * cos_h;

    """  # noqa
    return cp.ElementwiseKernel(
        '',
        'raw X lch',
        code,
        name='cucim_skimage_color_' + name)


@channel_as_last_axis()
def lch2lab(lch, *, channel_axis=-1):
    """CIE-LCH to CIE-LAB color space conversion.

    LCH is the cylindrical representation of the LAB (Cartesian) colorspace

    Parameters
    ----------
    lch : (..., 3, ...) array_like
        The N-D image in CIE-LCH format. The last (``N+1``-th) dimension must
        have at least 3 elements, corresponding to the ``L``, ``a``, and ``b``
        color channels.  Subsequent elements are copied.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in LAB format, with same shape as input `lch`.

    Raises
    ------
    ValueError
        If `lch` does not have at least 3 color channels (i.e. l, c, h).

    Examples
    --------
    >>> from skimage import data
    >>> from cucim.skimage.color import rgb2lab, lch2lab
    >>> img = cp.array(data.astronaut())
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    >>> img_lab2 = lch2lab(img_lch)
    """

    # make a copy because lch will be modified in-place by the kernel below
    lch = _prepare_lab_array(lch, force_copy=True)
    nchannels = lch.shape[-1]

    name = f'lch2lab_{nchannels}channel_{lch.dtype}'
    kern = _lch2lab_kernel(nchannels, name=name)
    kern(lch, size=lch.size // nchannels)
    return lch


def _prepare_lab_array(arr, force_copy=True):
    """Ensure input for lab2lch, lch2lab are well-posed.

    Arrays must be in floating point and have at least 3 elements in
    last dimension.  Return a new array.
    """
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError('Input array has less than 3 color channels')
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    return _func(arr, force_copy=force_copy)


@channel_as_last_axis()
def rgb2yuv(rgb, *, channel_axis=-1):
    """RGB to YUV color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in YUV format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Y is between 0 and 1.  Use YCbCr instead of YUV for the color space
    commonly used by video codecs, where Y ranges from 16 to 235.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YUV
    """
    return _convert(yuv_from_rgb, rgb, name='rgb2yuv')


@channel_as_last_axis()
def rgb2yiq(rgb, *, channel_axis=-1):
    """RGB to YIQ color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in YIQ format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).
    """
    return _convert(yiq_from_rgb, rgb, name='rgb2yiq')


@channel_as_last_axis()
def rgb2ypbpr(rgb, *, channel_axis=-1):
    """RGB to YPbPr color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in YPbPr format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YPbPr
    """
    return _convert(ypbpr_from_rgb, rgb, name='rgb2ypbpr')


@channel_as_last_axis()
def rgb2ycbcr(rgb, *, channel_axis=-1):
    """RGB to YCbCr color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in YCbCr format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Y is between 16 and 235. This is the color space commonly used by video
    codecs; it is sometimes incorrectly called "YUV".

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YCbCr
    """
    _post_colorconv = """
        y[3*i] += 16;
        y[3*i + 1] += 128;
        y[3*i + 2] += 128;
    """
    arr = _convert(ycbcr_from_rgb, rgb, post=_post_colorconv, name='rgb2ycbcr')
    return arr


@channel_as_last_axis()
def rgb2ydbdr(rgb, *, channel_axis=-1):
    """RGB to YDbDr color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in YDbDr format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    This is the color space commonly used by video codecs. It is also the
    reversible color transform in JPEG2000.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YDbDr
    """
    return _convert(ydbdr_from_rgb, rgb, name='rgb2ydbdr')


@channel_as_last_axis()
def yuv2rgb(yuv, *, channel_axis=-1):
    """YUV to RGB color space conversion.

    Parameters
    ----------
    yuv : (..., 3, ...) array_like
        The image in YUV format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `yuv` is not at least 2-D with shape (..., 3, ...).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YUV
    """
    return _convert(rgb_from_yuv, yuv, name='yuv2rgb')


@channel_as_last_axis()
def yiq2rgb(yiq, *, channel_axis=-1):
    """YIQ to RGB color space conversion.

    Parameters
    ----------
    yiq : (..., 3, ...) array_like
        The image in YIQ format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `yiq` is not at least 2-D with shape (..., 3, ...).
    """
    return _convert(rgb_from_yiq, yiq, name='yiq2rgb')


@channel_as_last_axis()
def ypbpr2rgb(ypbpr, *, channel_axis=-1):
    """YPbPr to RGB color space conversion.

    Parameters
    ----------
    ypbpr : (..., 3, ...) array_like
        The image in YPbPr format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `ypbpr` is not at least 2-D with shape (..., 3).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YPbPr
    """
    return _convert(rgb_from_ypbpr, ypbpr, name='ypbpr2rgb')


@channel_as_last_axis()
def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    """YCbCr to RGB color space conversion.

    Parameters
    ----------
    ycbcr : (..., 3, ...) array_like
        The image in YCbCr format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `ycbcr` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    Y is between 16 and 235. This is the color space commonly used by video
    codecs; it is sometimes incorrectly called "YUV".

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YCbCr
    """
    arr = ycbcr.copy()
    _pre_colorconv = """
        x[3*i] -= 16;
        x[3*i + 1] -= 128;
        x[3*i + 2] -= 128;
    """
    return _convert(rgb_from_ycbcr, arr, pre=_pre_colorconv,
                    name='ycbcr2rgb')


@channel_as_last_axis()
def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    """YDbDr to RGB color space conversion.

    Parameters
    ----------
    ydbdr : (..., 3, ...) array_like
        The image in YDbDr format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `ydbdr` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    This is the color space commonly used by video codecs, also called the
    reversible color transform in JPEG2000.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YDbDr
    """
    return _convert(rgb_from_ydbdr, ydbdr, name='ydbdr2rgb')
