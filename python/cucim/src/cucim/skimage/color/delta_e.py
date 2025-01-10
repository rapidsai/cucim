"""
Functions for calculating the "distance" between colors.

Implicit in these definitions of "distance" is the notion of "Just Noticeable
Distance" (JND).  This represents the distance between colors where a human can
perceive different colors.  Humans are more sensitive to certain colors than
others, which different deltaE metrics correct for with varying degrees of
sophistication.

The literature often mentions 1 as the minimum distance for visual
differentiation, but more recent studies (Mahy 1994) peg JND at 2.3

The delta-E notation comes from the German word for "Sensation" (Empfindung).

Reference
---------
https://en.wikipedia.org/wiki/Color_difference

"""

import warnings

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type
from .colorconv import lab2lch


def _float_inputs(lab1, lab2, allow_float32=True):
    if allow_float32:
        float_dtype = _supported_float_type((lab1.dtype, lab2.dtype))
    else:
        float_dtype = cp.float64
    lab1 = lab1.astype(float_dtype, copy=False)
    lab2 = lab2.astype(float_dtype, copy=False)
    return lab1, lab2


_cie76_kernel = cp.ElementwiseKernel(
    "X L1, X a1, X b1, X L2, X a2, X b2",
    "X out",
    """
// use double for the intermediate calculation of G to preserve accuracy
X tmp = (L2 - L1) * (L2 - L1);
tmp += (a2 - a1) * (a2 - a1);
tmp += (b2 - b1) * (b2 - b1);
out = sqrt(tmp);
""",
    name="cie76_internal",
)


def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """Euclidean distance between two points in Lab color space

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] A. R. Robertson, "The CIE 1976 color-difference formulae,"
           Color Res. Appl. 2, 7-11 (1977).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    out = _cie76_kernel(L1, a1, b1, L2, a2, b2)
    return out


_ciede94_kernel = cp.ElementwiseKernel(
    "X dH2, X L1, X C1, X L2, X C2, float64 k1, float64 k2, float64 kL, float64 kH, float64 kC",  # noqa: E501
    "X dE2",
    """
X dL = L1 - L2;
X dC = C1 - C2;
X SL = 1;
X SC = 1 + k1 * C1;
X SH = 1 + k2 * C1;
dE2 = dL / (kL * SL);
dE2 *= dE2;
X tmp = dC / (kC * SC);
tmp *= tmp;
dE2 += tmp;
tmp = kH * SH;
tmp *= tmp;
dE2 += dH2 / tmp;
dE2 = sqrt(max(dE2, 0.0));
""",
    name="ciede94_internal",
)


def deltaE_ciede94(
    lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015, *, channel_axis=-1
):
    """Color difference according to CIEDE 94 standard

    Accommodates perceptual non-uniformities through the use of application
    specific scale factors (`kH`, `kC`, `kL`, `k1`, and `k2`).

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kH : float, optional
        Hue scale
    kC : float, optional
        Chroma scale
    kL : float, optional
        Lightness scale
    k1 : float, optional
        first scale parameter
    k2 : float, optional
        second scale parameter
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        color difference between `lab1` and `lab2`

    Notes
    -----
    deltaE_ciede94 is not symmetric with respect to lab1 and lab2.  CIEDE94
    defines the scales for the lightness, hue, and chroma in terms of the first
    color.  Consequently, the first color should be regarded as the "reference"
    color.

    `kL`, `k1`, `k2` depend on the application and default to the values
    suggested for graphic arts

    ==========  ==============  ==========
    Parameter    Graphic Arts    Textiles
    ==========  ==============  ==========
    `kL`         1.000           2.000
    `k1`         0.045           0.048
    `k2`         0.015           0.014
    ==========  ==============  ==========

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = cp.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = cp.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1 = lab2lch(lab1, channel_axis=0)[:2]
    L2, C2 = lab2lch(lab2, channel_axis=0)[:2]

    dH2 = get_dH2(lab1, lab2, channel_axis=0)
    return _ciede94_kernel(dH2, L1, C1, L2, C2, k1, k2, kL, kH, kC)


_ciede2000_kernel = cp.ElementwiseKernel(
    "X a1, X b1, X a2, X b2, X L1, X L2, float64 kL, float64 kC, float64 kH",
    "X dE2",
    """

// distort `a` based on average chroma
// then convert to lch coordinates from distorted `a`
// all subsequence calculations are in the new coordinates
// (often denoted "prime" in the literature)

// cide2000 has four terms to delta_e:
// 1) Luminance term
// 2) Hue term
// 3) Chroma term
// 4) hue Rotation term

// use double for the intermediate calculation of G to preserve accuracy
double G = 0.5 * (hypot((double)a1, (double)b1) + hypot((double)a2, (double)b2));
G = pow(G, 7.0);
G = sqrt(G / (G + 6103515625));
X scale = 1.0 + 0.5 * (1.0 - G);

X C1 = hypot(a1 * scale, b1);
X h1 = atan2(b1, a1 * scale);
if (h1 < 0) {
    h1 += 2 * M_PI;
}

X C2 = hypot(a2 * scale, b2);
X h2 = atan2(b2, a2 * scale);
if (h2 < 0) {
    h2 += 2 * M_PI;
}

// lightness term
X Lbar = 0.5 * (L1 + L2);
X tmp = Lbar - 50;
tmp *= tmp;
X SL = 1.0 + 0.015 * tmp / sqrt(20.0 + tmp);
X L_term = (L2 - L1) / (kL * SL);

// chroma term
X Cbar = 0.5 * (C1 + C2);
X SC = 1.0 + 0.045 * Cbar;
X C_term = (C2 - C1) / (kC * SC);

X h_diff = h2 - h1;
X h_sum = h1 + h2;
X CC = C1 * C2;
X dH = h_diff;
if (CC == 0.0) {
    dH = 0;
} else {
    if (h_diff > M_PI) {
      dH -= 2 * M_PI;
    }
    if (h_diff < -M_PI) {
      dH += 2 * M_PI;
    }
}
X dH_term = 2 * sqrt(CC) * sin(dH / 2.0);

X Hbar = h_sum;
if (CC != 0.0) {
  if (abs(h_diff) > M_PI) {
    if (h_sum < 2.0 * M_PI) {
      Hbar += 2.0 * M_PI;
    }
    if (h_sum >= 2.0 * M_PI) {
      Hbar -= 2 * M_PI;
    }
  }
}
if (CC == 0.0) {
  Hbar *= 2;
}
Hbar *= 0.5;

X T = 1.0 - 0.17 * cos(Hbar - 30.0 / 180.0 * M_PI)
    + 0.24 * cos(2 * Hbar)
    + 0.32 * cos(3 * Hbar + 6.0 / 180.0 * M_PI)
    - 0.20 * cos(4 * Hbar - 63.0 / 180.0 * M_PI);
X SH = 1 + 0.015 * Cbar * T;
X H_term = dH_term / (kH * SH);

X c7 = pow(Cbar, (X)7.0);
X c7_term = sqrt(c7 / (c7 + 6103515625));
X Rc = 2.0 * c7_term;

// hue rotation
tmp = (Hbar- 4.799655442984406) / 0.4363323129985824;
tmp *= tmp;
X dtheta = 0.5235987755982988 * exp(-tmp);
X R_term = -sin(2 * dtheta) * Rc * C_term * H_term;

// put it all together
dE2 = L_term * L_term;
dE2 += C_term * C_term;
dE2 += H_term * H_term;
dE2 += R_term;
dE2 = sqrt(max(dE2, 0.0));
""",  # noqa: E501
    name="deltaE_ciede2000_internal",
)


def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1):
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the lightness, chroma,
    and hue (`kL`, `kC`, `kH` respectively).  These default to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           :DOI:`10.1364/AO.33.008069`
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    warnings.warn(
        "The numerical accuracy of this function on the GPU is reduced "
        "relative to the CPU version"
    )
    channel_axis = channel_axis % lab1.ndim
    unroll = False
    if lab1.ndim == 1 and lab2.ndim == 1:
        unroll = True
        if lab1.ndim == 1:
            lab1 = lab1[None, :]
        if lab2.ndim == 1:
            lab2 = lab2[None, :]
        channel_axis += 1
    L1, a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[:3]

    dE2 = _ciede2000_kernel(a1, b1, a2, b2, L1, L2, kL, kC, kH)
    if unroll:
        dE2 = dE2[0]
    return dE2


_cmc_kernel = cp.ElementwiseKernel(
    "X C1, X L1, X h1, X C2, X L2, X dH2, float64 kL, float64 kC",
    "X dE2",
    """

X dC = C1 - C2;
X dL = L1 - L2;

X T = (h1 >= 2.8623399732707004 && h1 <= 6.021385919380437) ?
    0.56 + 0.2 * abs(cos(h1 + 2.9321531433504737)) :
    0.36 + 0.4 * abs(cos(h1 + 0.6108652381980153));

X tmp = pow(C1, (X)4.0);
X F = sqrt(tmp / (tmp + 1900));

X SL = (L1 < 16) ? 0.511 : 0.040975 * L1 / (1.0 + 0.01765 * L1);
X SC = 0.638 + 0.0638 * C1 / (1.0 + 0.0131 * C1);
X SH = SC * (F * T + 1 - F);

dE2 = dL / (kL * SL);
dE2 *= dE2;
tmp = dC / (kC * SC);
tmp *= tmp;
dE2 += tmp;
dE2 += dH2 / (SH * SH);
dE2 = sqrt(max(dE2, 0.0));
""",
    name="deltaE_cmc_internal",
)


def deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1):
    """Color difference from the  CMC l:c standard.

    This color difference was developed by the Colour Measurement Committee
    (CMC) of the Society of Dyers and Colourists (United Kingdom). It is
    intended for use in the textile industry.

    The scale factors `kL`, `kC` set the weight given to differences in
    lightness and chroma relative to differences in hue.  The usual values are
    ``kL=2``, ``kC=1`` for "acceptability" and ``kL=1``, ``kC=1`` for
    "imperceptibility".  Colors with ``dE > 1`` are "different" for the given
    scale factors.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    Notes
    -----
    deltaE_cmc the defines the scales for the lightness, hue, and chroma
    in terms of the first color.  Consequently
    ``deltaE_cmc(lab1, lab2) != deltaE_cmc(lab2, lab1)``

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    .. [3] F. J. J. Clarke, R. McDonald, and B. Rigg, "Modification to the
           JPC79 colour-difference formula," J. Soc. Dyers Colour. 100, 128-132
           (1984).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    lab1 = cp.moveaxis(lab1, source=channel_axis, destination=0)
    lab2 = cp.moveaxis(lab2, source=channel_axis, destination=0)
    L1, C1, h1 = lab2lch(lab1, channel_axis=0)[:3]
    L2, C2, h2 = lab2lch(lab2, channel_axis=0)[:3]

    dH2 = get_dH2(lab1, lab2, channel_axis=0)
    return _cmc_kernel(C1, L1, h1, C2, L2, dH2, kL, kC)


_dH2_kernel = cp.ElementwiseKernel(
    "X a1, X b1, X a2, X b2",
    "X out",
    """
// use double for the intermediate calculation of G to preserve accuracy
double C1 = hypot(a1, b1);
double C2 = hypot(a2, b2);
out = 2.0 * ((C1 * C2) - (a1 * a2 + b1 * b2));
""",
    name="dH2_internal",
)


def get_dH2(lab1, lab2, *, channel_axis=-1):
    """squared hue difference term occurring in deltaE_cmc and deltaE_ciede94

    Despite its name, "dH" is not a simple difference of hue values.  We avoid
    working directly with the hue value, since differencing angles is
    troublesome.  The hue term is usually written as:
        c1 = sqrt(a1**2 + b1**2)
        c2 = sqrt(a2**2 + b2**2)
        term = (a1-a2)**2 + (b1-b2)**2 - (c1-c2)**2
        dH = sqrt(term)

    However, this has poor roundoff properties when a or b is dominant.
    Instead, ab is a vector with elements a and b.  The same dH term can be
    re-written as:
        |ab1-ab2|**2 - (|ab1| - |ab2|)**2
    and then simplified to:
        2*|ab1|*|ab2| - 2*dot(ab1, ab2)
    """
    # This function needs double precision internally for accuracy
    input_is_float_32 = (
        _supported_float_type((lab1.dtype, lab2.dtype)) == cp.float32
    )
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=False)
    a1, b1 = cp.moveaxis(lab1, source=channel_axis, destination=0)[1:3]
    a2, b2 = cp.moveaxis(lab2, source=channel_axis, destination=0)[1:3]

    out = _dH2_kernel(a1, b1, a2, b2)
    if input_is_float_32:
        out = out.astype(np.float32)
    return out
