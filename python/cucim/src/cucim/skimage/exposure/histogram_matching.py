import cupy as cp

from .._shared import utils


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = cp.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = cp.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = cp.cumsum(src_counts) / source.size
    tmpl_quantiles = cp.cumsum(tmpl_counts) / template.size

    interp_a_values = cp.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


@utils.channel_as_last_axis(channel_arg_positions=(0, 1))
@utils.deprecate_multichannel_kwarg()
def match_histograms(image, reference, *, channel_axis=None,
                     multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    multichannel : bool, optional
        Apply the matching separately for each channel. This argument is
        deprecated: specify `channel_axis` instead.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if channel_axis is not None:
        if image.shape[channel_axis] != reference.shape[channel_axis]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = cp.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = utils._supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched
