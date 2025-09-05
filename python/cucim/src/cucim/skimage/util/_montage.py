import math

import cupy as cp

from .. import exposure
from .._shared import utils

__all__ = ["montage"]


@cp.memoize(for_each_device=True)
def _montage_kernel(num_channels, n_pad):
    if n_pad == 0:
        # No padding case
        code = """
    // Get output array dimensions
    int out_height = output.shape()[0];
    int out_width = output.shape()[1];

    // Calculate spatial coordinates from linear index
    int out_col = i % out_width;
    int out_row = i / out_width;

    // Calculate which tile this pixel belongs to
    int tile_row = out_row / n_rows;
    int tile_col = out_col / n_cols;

    // Calculate position within the tile
    int local_row = out_row % n_rows;
    int local_col = out_col % n_cols;

    // Calculate which image this tile corresponds to
    int image_idx = tile_row * ntiles_col + tile_col;

    // Check if we have an image for this tile
    bool copy_from_image = (image_idx < n_images);\n"""
    else:
        # Padding case
        code = f"""
    // Get output array dimensions
    int out_height = output.shape()[0];
    int out_width = output.shape()[1];

    // Calculate spatial coordinates from linear index
    int out_col = i % out_width;
    int out_row = i / out_width;

    // Check if we're in padding areas
    bool in_padding = (out_row < {n_pad} || out_col < {n_pad});

    // Calculate which tile this pixel belongs to
    int tile_row = -1, tile_col = -1, local_row = -1, local_col = -1;
    int image_idx = -1;
    bool copy_from_image = false;

    if (!in_padding) {{
        tile_row = (out_row - {n_pad}) / (n_rows + {n_pad});
        tile_col = (out_col - {n_pad}) / (n_cols + {n_pad});

        // Calculate position within the tile
        local_row = (out_row - {n_pad}) % (n_rows + {n_pad});
        local_col = (out_col - {n_pad}) % (n_cols + {n_pad});

        // Check if we're in inter-tile padding
        if (local_row < n_rows && local_col < n_cols) {{
            // Calculate which image this tile corresponds to
            image_idx = tile_row * ntiles_col + tile_col;

            // Check if we have an image for this tile
            if (image_idx < n_images) {{
                copy_from_image = true;
            }}
        }}
    }}\n"""

    if num_channels == 1:
        code += """
        if (copy_from_image) {
            // Copy single channel pixel from input array
            int input_idx = image_idx * n_rows * n_cols +
                            local_row * n_cols + local_col;
            output[i] = arr_in[input_idx];
        } else {
            output[i] = fill_values[0];
        }\n"""
    else:
        # Generate explicitly unrolled loops over channels
        # Multi-channel: explicitly unroll the channel loop
        fill_statements = []
        copy_statements = []

        code += f"""
            int channel0_idx = image_idx * n_rows * n_cols * {num_channels} +
                               local_row * n_cols * {num_channels} +
                               local_col * {num_channels};\n"""
        for chan in range(num_channels):
            output_idx = f"i * {num_channels} + {chan}"
            fill_statements.append(
                f"        output[{output_idx}] = fill_values[{chan}];"
            )
            copy_statements.append(
                f"        output[{output_idx}] = arr_in[channel0_idx + {chan}];"
            )
        copy_code = "\n".join(copy_statements)
        fill_code = "\n".join(fill_statements)

        code += f"""
        // Process all channels for this spatial location (unrolled)
        if (copy_from_image) {{
            {copy_code}
        }} else {{
            {fill_code}
        }}\n"""

    return cp.ElementwiseKernel(
        "raw X arr_in, raw X fill_values, int32 n_images, int32 n_rows, "
        "int32 n_cols, int32 ntiles_row, int32 ntiles_col",
        "raw X output",
        code,
        name="cucim_montage_kernel",
    )


@utils.channel_as_last_axis(multichannel_output=False)
def montage(
    arr_in,
    fill="mean",
    rescale_intensity=False,
    grid_shape=None,
    padding_width=0,
    *,
    channel_axis=None,
    square_grid_default=True,
    use_fused_kernel=True,
):
    """Create a montage of several single- or multichannel images.

    Create a rectangular montage from an input array representing an ensemble
    of equally shaped single- (gray) or multichannel (color) images.

    For example, ``montage(arr_in)`` called with the following `arr_in`

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    will return

    +---+---+
    | 1 | 2 |
    +---+---+
    | 3 | * |
    +---+---+

    where the '*' patch will be determined by the `fill` parameter.

    Parameters
    ----------
    arr_in : ndarray, shape (K, M, N[, C])
        An array representing an ensemble of `K` images of equal shape.
    fill : float or array-like of floats or 'mean', optional
        Value to fill the padding areas and/or the extra tiles in
        the output array. Has to be `float` for single channel collections.
        For multichannel collections has to be an array-like of shape of
        number of channels. If `mean`, uses the mean value over all images.
    rescale_intensity : bool, optional
        Whether to rescale the intensity of each image to [0, 1].
    grid_shape : tuple, optional
        The desired grid shape for the montage `(ntiles_row, ntiles_column)`.
        The default aspect ratio is square.
    padding_width : int, optional
        The size of the spacing between the tiles and between the tiles and
        the borders. If non-zero, makes the boundaries of individual images
        easier to perceive.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Other Parameters
    ----------------
    square_grid_default : bool, optional
        If ``True``, use a square grid shape by default as suggested by
        scikit-image. Otherwise, allow a rectangular grid to more tightly pack
        the images.
        Note: This argument is not present in scikit-image.
    use_fused_kernel : bool, optional
        Whether to use the single-kernel CuPy elementwise kernel implementation
        (True) or the original slicing-based implementation (False).
        Default is True. The kernel implementation is generally faster for
        large arrays due to better GPU parallelization.
        Note: This argument is not present in scikit-image.

    Returns
    -------
    arr_out : (K*(M+p)+p, K*(N+p)+p[, C]) ndarray
        Output array with input images glued together (including padding `p`).

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.util import montage
    >>> arr_in = cp.arange(3 * 2 * 2).reshape(3, 2, 2)
    >>> arr_in  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1],
            [ 2,  3]],
           [[ 4,  5],
            [ 6,  7]],
           [[ 8,  9],
            [10, 11]]])
    >>> arr_out = montage(arr_in)
    >>> arr_out.shape
    (4, 4)
    >>> arr_out
    array([[ 0,  1,  4,  5],
           [ 2,  3,  6,  7],
           [ 8,  9,  5,  5],
           [10, 11,  5,  5]])
    >>> arr_in.mean()
    5.5
    >>> arr_out_nonsquare = montage(arr_in, grid_shape=(1, 3))
    >>> arr_out_nonsquare
    array([[ 0,  1,  4,  5,  8,  9],
           [ 2,  3,  6,  7, 10, 11]])
    >>> arr_out_nonsquare.shape
    (2, 6)
    """

    if channel_axis is not None:
        arr_in = cp.asarray(arr_in)
    else:
        arr_in = cp.asarray(arr_in)[..., cp.newaxis]

    if arr_in.ndim != 4:
        raise ValueError(
            "Input array has to be 3-dimensional for grayscale "
            "images, or 4-dimensional with a `channel_axis` "
            "specified."
        )

    n_images, n_rows, n_cols, n_chan = arr_in.shape

    if grid_shape:
        ntiles_row, ntiles_col = (int(s) for s in grid_shape)
        if ntiles_row * ntiles_col < n_images:
            raise ValueError(
                "math.prod(grid_shape) must be greater than or equal to the "
                "number of images"
            )
    else:
        ntiles_col = math.ceil(math.sqrt(n_images))
        if square_grid_default:
            ntiles_row = ntiles_col
        else:
            ntiles_row = math.ceil(n_images / ntiles_col)

    # Rescale intensity if necessary
    if rescale_intensity:
        for i in range(n_images):
            arr_in[i] = exposure.rescale_intensity(arr_in[i])

    # Calculate the fill value
    if fill == "mean":
        fill = arr_in.mean(axis=(0, 1, 2))
    fill = cp.atleast_1d(fill).astype(arr_in.dtype, copy=False)

    # Pre-allocate an array with padding for montage
    n_pad = padding_width
    arr_out = cp.empty(
        (
            (n_rows + n_pad) * ntiles_row + n_pad,
            (n_cols + n_pad) * ntiles_col + n_pad,
            n_chan,
        ),
        dtype=arr_in.dtype,
    )

    if use_fused_kernel:
        # Use elementwise kernel to copy the data to the output array
        kern = _montage_kernel(n_chan, n_pad)
        kern(
            arr_in,
            fill,
            n_images,
            n_rows,
            n_cols,
            ntiles_row,
            ntiles_col,
            arr_out,
            size=arr_out.size // n_chan,
        )
    else:
        # Fill array with fill values for slice-based implementation
        for idx_chan in range(n_chan):
            arr_out[..., idx_chan] = fill[idx_chan]

        # Use original slice-based implementation
        slices_row = [
            slice(
                n_pad + (n_rows + n_pad) * n,
                n_pad + (n_rows + n_pad) * n + n_rows,
            )
            for n in range(ntiles_row)
        ]
        slices_col = [
            slice(
                n_pad + (n_cols + n_pad) * n,
                n_pad + (n_cols + n_pad) * n + n_cols,
            )
            for n in range(ntiles_col)
        ]

        # Copy the data to the output array
        for idx_image, image in enumerate(arr_in):
            idx_sr = idx_image // ntiles_col
            idx_sc = idx_image % ntiles_col
            arr_out[slices_row[idx_sr], slices_col[idx_sc], :] = image

    if channel_axis is not None:
        return arr_out
    else:
        return arr_out[..., 0]
