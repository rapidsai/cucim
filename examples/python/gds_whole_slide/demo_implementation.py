import os
import warnings

import cupy as cp
import dask.array as da
import kvikio
import kvikio.defaults
import numpy as np
import openslide
import tifffile
from cucim.clara import filesystem
from kvikio.cufile import IOFuture
from kvikio.zarr import GDSStore
from tifffile import TiffFile
from zarr import DirectoryStore
from zarr.creation import init_array

"""
Developed with Dask 2022.05.2
zarr >= 2.13.2
kvikio >= 2022.10.00  (but had to use a recent development branch on my system to properly find libcufile.so)
"""  # noqa: E501


def get_n_tiles(page):
    """Create a tuple containing the number of tiles along each axis

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        The TIFF page.
    """

    tdepth, tlength, twidth = (page.tiledepth, page.tilelength, page.tilewidth)
    assert page.shaped[-1] == page.samplesperpixel
    imdepth, imlength, imwidth, samples = page.shaped[-4:]

    n_width = (imwidth + twidth - 1) // twidth
    n_length = (imlength + tlength - 1) // tlength
    n_depth = (imdepth + tdepth - 1) // tdepth
    return (n_depth, n_length, n_width)


def _get_tile_multiindex(page, index, n_tiles):
    """Position offsets into the output array for the current tile.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        The TIFF page.
    index : int
        The linear index in page.dataoffsets (or page.databytecounts)
        corresponding to the current tile.
    n_tiles : int
        The total number of tiles as returned by ``get_n_tiles(page)``

    Returns
    -------
    multi_index : tuple of int
        Starting index for the tile along each axis in the output array.
    """
    d, h, w = n_tiles
    wh = w * h
    whd = wh * d
    multi_index = (
        index // whd,
        (index // wh) % d * page.tiledepth,
        (index // w) % h * page.tilelength,
        index % w * page.tilewidth,
        0,
    )
    return multi_index


def _decode(data, tile_shape, truncation_slices):
    """Reshape data from a 1d buffer to the destination tile's shape.

    Parameters
    ----------
    data : buffer or ndarray
        Data array (1d raveled tile data).
    tile_shape : tuple of int
        The shape of the tile
    truncation_slices : tuple of slice
        Slice objects used to truncate the reshaped data. Needed to trim the
        last tile along a given dimension when the page size is not an even
        multiple of the tile width.
    """
    if not hasattr(data, "__cuda_array_interface__"):
        data = np.frombuffer(data, np.uint8)
    data.shape = tile_shape
    # truncate any tiles that extend past the image boundary
    if truncation_slices:
        data = data[truncation_slices]
    return data


def _truncation_slices(check_needed, page_shape, offsets, tile_shape, n_tiles):
    """Determine any necessary truncation of boundary tiles.

    This is essentially generating a tuple of slices for doing:

    tile = tile[
        : page_shape[0] - offsets[0],
        : page_shape[1] - offsets[1],
        : page_shape[2] - offsets[2],
    ]

    but has additional logic to return None in cases where truncation is not
    needed.

    Parameters
    ----------
    check_needed : 3-tuple of bool
        Any axis whose page size is not evenly divisible by the tile size will
        have a True entry in this tuple.
    page_shape : tuple of int
        The shape of the current TIFF page (depth, length, width[, channels])
    offsets : 3-tuple of int
        Starting corner indices for the current tile (depth, length, width).
    tile_shape : tuple of int
        Shape of a single tile (depth, length, width[, channels]).
    n_tiles : 3-tuple of int
        The number of tiles along each axis (depth, length, width)

    Returns
    -------
    tile_slice : 3-tuple of slice or None
        Slices needed to generate a truncated tile via
        ``tile = tile[tile_slice]``. Returns None of no truncation is needed
        at the current tile.
    """
    any_truncated = False
    slices = []
    for ax in range(len(offsets)):
        if (
            check_needed[ax]
            # don't need to truncate unless this is the last tile along an axis
            and (offsets[ax] // tile_shape[ax] == n_tiles[ax] - 1)
        ):
            any_truncated = True
            slices.append(slice(0, page_shape[ax] - offsets[ax]))
        else:
            slices.append(slice(None))
    if any_truncated:
        return tuple(slices)
    return None


def read_openslide(fname, level, clear_cache=True):
    """CPU-based reader using openslide followed by cupy.array."""

    if clear_cache:
        assert filesystem.discard_page_cache(fname)

    slide = openslide.OpenSlide(fname)
    out = slide.read_region(
        location=(0, 0), level=level, size=slide.level_dimensions[level]
    )
    # convert from PIL image to NumPy array
    out = np.asarray(out)
    # transfer to GPU, omitting the alpha channel
    return cp.array(out[..., :3])


def read_tifffile(fname, level, clear_cache=True):
    """CPU-based reader using tifffile followed by cupy.array."""

    if clear_cache:
        assert filesystem.discard_page_cache(fname)

    return cp.array(tifffile.imread(fname, level=level))


def _get_aligned_read_props(offsets, bytecounts, alignment=4096):
    """Adjust offsets and bytecounts to get reads of the desired alignment.

    Parameters
    ----------
    offsets : sequence of int
        The bytes offsets of each tile in the TIFF page.
        (i.e. tifffile's ``page.dataoffsets``).
    bytecounts : sequence of int
        The size in bytes of each tile in the TIFF page.
        (i.e. tifffile's ``page.databytecounts``).
    alignment : int, optional
        The desired alignment for the read operation. For GPUDirect Storage,
        this should be 4096.

    Notes
    -----
    For GPUDirect Storage, it is important that reads occur width 4096-byte
    alignment and have a size that is a multiple of 4096-bytes. So, in this
    function we offset the read from the tile's start position back to the
    previous byte-aligned position. We then round up the total bytes to be read
    so that it is a multiple of `alignment`.
    """
    offsets = np.asarray(offsets, dtype=int)
    bytecounts = np.asarray(bytecounts, dtype=int)
    rounded_offsets = (offsets // alignment) * alignment
    buffer_offsets = offsets - rounded_offsets
    rounded_bytecounts = buffer_offsets + bytecounts
    rounded_bytecounts = (
        np.ceil(rounded_bytecounts / alignment).astype(int) * alignment
    )

    # truncate last bytecounts entry to avoid possibly exceeding file extent
    last = offsets[-1] + bytecounts[-1]
    rounded_last = rounded_offsets[-1] + rounded_bytecounts[-1]
    rounded_bytecounts[-1] += last - rounded_last

    return rounded_offsets, rounded_bytecounts, buffer_offsets


def _bulk_read_is_possible(offsets, bytecounts):
    """Check that all of the page's tile data is contiguous in memory."""
    contiguous_offsets = np.array(offsets)[:-1] + np.array(bytecounts)[:-1]
    return np.all(contiguous_offsets == np.array(offsets)[1:])


def _get_bulk_offset_and_size(offsets, bytecounts, align=4096):
    """Determine offsets and bytecounts for a single bulk read of the desired
    alignment.

    Notes
    -----
    See documentation of `_get_aligned_read_props` for explanation of the
    alignment requirements of GPUDirect Storage.
    """
    if not _bulk_read_is_possible(offsets, bytecounts):
        raise RuntimeError("Tiles are not stored contiguously!")
    total_size = offsets[-1] - offsets[0] + bytecounts[-1]
    # reduce offset to closest value that is aligned to 4096 bytes
    offset_aligned = (offsets[0] // align) * align
    padded_bytes = offsets[0] - offset_aligned
    # increase size to keep the same final byte
    total_size += padded_bytes
    return offset_aligned, total_size, padded_bytes


def read_alltiles_bulk(fname, level, clear_cache=True):
    """Read all tiles from a page in a single, bulk read operation.

    Can be used to compare to the performance of tile-based reads.
    """
    if clear_cache:
        assert filesystem.discard_page_cache(fname)

    with TiffFile(fname) as tif:
        fh = kvikio.CuFile(fname, "r")
        page = tif.pages[level]
        offset, total_size, padded_bytes = _get_bulk_offset_and_size(
            page.dataoffsets, page.databytecounts
        )
        output = cp.empty(total_size, dtype=cp.uint8)
        size = fh.read(output, file_offset=offset)
        if size != total_size:
            raise ValueError("failed to read the expected number of bytes")
    return output[padded_bytes:]


def get_tile_buffers(fname, level, n_buffer):
    with TiffFile(fname) as tif:
        page = tif.pages[level]

        (
            rounded_offsets,
            rounded_bytecounts,
            buffer_offsets,
        ) = _get_aligned_read_props(
            offsets=page.dataoffsets,
            bytecounts=page.databytecounts,
            alignment=4096,
        )
        # Allocate buffer based on size of the largest tile after rounding
        # up to the nearest multiple of 4096.
        # (in the uncompressed TIFF case, all tiles have an equal number of
        #  bytes)
        buffer_bytecount = rounded_bytecounts.max()
        assert buffer_bytecount % 4096 == 0

        # note: tile_buffer is C-contiguous so tile_buffer[i] is contiguous
        tile_buffers = tuple(
            cp.empty(buffer_bytecount, dtype=cp.uint8) for n in range(n_buffer)
        )
    return tile_buffers


def read_tiled(
    fname,
    levels=[0],
    backend="kvikio-pread",
    n_buffer=100,
    tile_func=None,
    tile_func_kwargs={},
    out_dtype=None,
    clear_cache=True,
    preregister_memory_buffers=False,
    tile_buffers=None,
):
    """Read an uncompressed, tiled multiresolution TIFF image to GPU memory.

    Parameters
    ----------
    fname : str, optional
        File name.
    levels : sequence of int or 'all', optional
        The resolution levels to read. If 'all' then all levels in the file
        will be read. Level 0 corresponds to the highest resolution in the
        file.
    backend : {'kvikio-pread', 'kvikio-read', 'kvikio-raw_read'}, optional
        The approach to use when reading the file. The kvikio options can make
        use of GPUDirect Storage. Best performance was observed with the
        default 'kvikio-pread', which does asynchronous, multithreaded tile
        reads.
    n_buffer : int, optional
        Scratch space equal to `n_buffer` TIFF tiles will be allocated.
        Providing scratch space for multiple tiles helps the performance in the
        recommended asynchronous 'kvikio-pread' mode.
    tile_func : function, optional
        A CuPy-based function to apply to each tile after it is read. Must
        take as input a single CuPy array and return a CuPy array of the same
        shape (with possibly different dtype). The default of None does not
        apply any processing to each tile.
    tile_func_kwargs : dict, optional
        Keyword arguments to pass into `tile_func`.
    out_dtype : cp.dtype, optional
        The output dtype of the output array. If unspecified, it will be equal
        to the dtype of the input TIFF data.
    clear_cache : bool, optional
        If True, clear the file system's page cache prior to starting any
        read operations. This is necessary to avoid caching so that one gets
        accurate benchmark results if running this function repeatedly on the
        same input data.
    preregister_memory_buffers : bool, optional
        If True, explicitly preregister the memory buffers with kvikio via
        `kvikio.memory_register`.
    tile_buffers : tuple or list of ndarray
        If provided, use this preallocated set of tile buffers instead of
        allocating tile buffers within this function. Will also override
        n_buffer, setting it to ``n_buffer = len(tile_buffers)``. If the
        provided buffers are too small, a warning will be printed and new
        buffers will be allocated instead.

    Returns
    -------
    out : list of cupy.ndarray
        One CuPy array per requested level in `levels`.

    """
    if not os.path.isfile(fname):
        raise ValueError(f"file not found: {fname}")

    if clear_cache:
        assert filesystem.discard_page_cache(fname)

    if tile_buffers is not None:
        if not (
            isinstance(tile_buffers, (tuple, list))
            and all(isinstance(b, cp.ndarray) for b in tile_buffers)
        ):
            raise ValueError("tile_buffers must be a list of ndarray")
        # override user-provided n_buffer
        if len(tile_buffers) != n_buffer:
            n_buffer = len(tile_buffers)

    with TiffFile(fname) as tif:
        if isinstance(levels, int):
            levels = (levels,)
        if levels == "all":
            pages = tuple(tif.pages[n] for n in range(len(tif.pages)))
        elif isinstance(levels, (tuple, list)):
            pages = tuple(tif.pages[n] for n in levels)
        else:
            raise ValueError(
                "pages must be a tuple or list of int or the " "string 'all'"
            )

        # sanity check: identical tile size for all TIFF pages
        #               todo: is this always true?
        assert len(set(p.tilelength for p in pages)) == 1
        assert len(set(p.tilewidth for p in pages)) == 1

        outputs = []

        fh = kvikio.CuFile(fname, "r")
        page = pages[0]
        n_chan = page.shaped[-1]

        for page in pages:
            if out_dtype is None:
                out_dtype = page.dtype
            out_array = cp.ndarray(shape=page.shape, dtype=out_dtype)

            (
                rounded_offsets,
                rounded_bytecounts,
                buffer_offsets,
            ) = _get_aligned_read_props(
                offsets=page.dataoffsets,
                bytecounts=page.databytecounts,
                alignment=4096,
            )
            # Allocate buffer based on size of the largest tile after rounding
            # up to the nearest multiple of 4096.
            # (in the uncompressed TIFF case, all tiles have an equal number of
            #  bytes)
            buffer_bytecount = rounded_bytecounts.max()
            assert buffer_bytecount % 4096 == 0

            tile_shape = (page.tilelength, page.tilewidth, n_chan)
            # note: tile_buffer is C-contiguous so tile_buffer[i] is contiguous
            if tile_buffers is None:
                tile_buffers = tuple(
                    cp.empty(buffer_bytecount, dtype=cp.uint8)
                    for n in range(n_buffer)
                )
            elif tile_buffers[0].size < buffer_bytecount:
                warnings.warn(
                    "reallocating tile buffers to accommodate data size"
                )
                tile_buffers = tuple(
                    cp.empty(buffer_bytecount, dtype=cp.uint8)
                    for n in range(n_buffer)
                )
            else:
                buffer_bytecount = tile_buffers[0].size

            if preregister_memory_buffers:
                for n in range(n_buffer):
                    kvikio.memory_register(tile_buffers[n])

            # compute number of tiles up-front to make _decode more efficient
            n_tiles = get_n_tiles(page)

            tile_shape = (
                page.tiledepth,
                page.tilelength,
                page.tilewidth,
                page.samplesperpixel,
            )
            keyframe = page.keyframe
            truncation_check_needed = (
                (keyframe.imagedepth % page.tiledepth != 0),
                (keyframe.imagelength % page.tilelength != 0),
                (keyframe.imagewidth % page.tilewidth != 0),
            )
            any_truncated = any(truncation_check_needed)
            page_shape = page.shaped[
                1:
            ]  # Any reason to prefer page.keyframe.imagedepth, etc. here as opposed to page.shape or page.shaped?    # noqa: E501

            if backend == "kvikio-raw_read":

                def read_tile_raw(fh, tile_buffer, bytecount, offset):
                    """returns the # of bytes read"""
                    size = fh.raw_read(
                        tile_buffer[:bytecount], file_offset=offset
                    )
                    if size != bytecount:
                        raise ValueError(
                            "failed to read the expected number of bytes"
                        )
                    return size

                kv_read = read_tile_raw

            elif backend == "kvikio-read":

                def read_tile(fh, tile_buffer, bytecount, offset):
                    """returns the # of bytes read"""
                    size = fh.read(tile_buffer[:bytecount], file_offset=offset)
                    if size != bytecount:
                        raise ValueError(
                            "failed to read the expected number of bytes"
                        )
                    return size

                kv_read = read_tile

            elif backend == "kvikio-pread":

                def read_tile_async(fh, tile_buffer, bytecount, offset):
                    """returns a future"""
                    future = fh.pread(
                        tile_buffer[:bytecount], file_offset=offset
                    )
                    # future.get()
                    return future

                kv_read = read_tile_async
            else:
                raise ValueError(f"unrecognized backend: {backend}")

            # note: page.databytecounts contains the size of all tiles in
            #       bytes. It will only vary in the case of compressed data
            for index, (
                offset,
                tile_bytecount,
                rounded_bytecount,
                buffer_offset,
            ) in enumerate(
                zip(
                    rounded_offsets,
                    page.databytecounts,
                    rounded_bytecounts,
                    buffer_offsets,
                )
            ):
                index_mod = index % n_buffer
                if index == 0:
                    # initialize lists for storage of future results
                    all_futures = []
                    all_tiles = []
                    all_slices = []
                elif index_mod == 0:
                    # process the prior group of n_buffer futures
                    for tile, sl, future in zip(
                        all_tiles, all_slices, all_futures
                    ):
                        if isinstance(future, IOFuture):
                            size = future.get()
                            if size != rounded_bytecount:
                                raise ValueError(
                                    "failed to read the expected number of "
                                    "bytes"
                                )
                        tile = tile[0]  # omit depth axis
                        if tile_func is None:
                            out_array[sl] = tile
                        else:
                            out_array[sl] = tile_func(tile, **tile_func_kwargs)
                    # reset the lists to prepare for the next n_buffer tiles
                    all_futures = []
                    all_tiles = []
                    all_slices = []

                # read a multiple of 4096 bytes into the current buffer at an
                # offset that is aligned to 4096 bytes
                read_output = kv_read(
                    fh,
                    tile_buffers[index_mod],
                    rounded_bytecount,
                    offset,
                )

                # Determine offsets into `out_array` for the current tile
                # and determine slices to truncate the tile if needed.
                offset_indices = _get_tile_multiindex(page, index, n_tiles)
                (s, d, h, w, _) = offset_indices
                if any_truncated:
                    trunc_sl = _truncation_slices(
                        truncation_check_needed,
                        page_shape,
                        offset_indices[1:4],
                        tile_shape,
                        n_tiles,
                    )
                else:
                    trunc_sl = None

                # Reads are aligned to 4096-byte boundaries, so buffer_offset
                # is needed to discard any initial bytes prior to the actual
                # tile start.
                buffer_start = buffer_offset
                buffer_end = buffer_start + tile_bytecount
                tile = _decode(
                    tile_buffers[index_mod][buffer_start:buffer_end],
                    tile_shape,
                    trunc_sl,
                )
                all_futures.append(read_output)
                all_tiles.append(tile)
                all_slices.append(
                    (slice(h, h + tile_shape[1]), slice(w, w + tile_shape[2]))
                )

            for tile, sl, future in zip(all_tiles, all_slices, all_futures):
                if isinstance(future, IOFuture):
                    # make sure the buffer is filled with data
                    future.get()
                tile = tile[0]  # omit depth axis
                if tile_func is None:
                    out_array[sl] = tile
                else:
                    out_array[sl] = tile_func(tile, **tile_func_kwargs)
            outputs.append(out_array)

            if preregister_memory_buffers:
                for n in range(n_buffer):
                    kvikio.memory_deregister(tile_buffers[n])

    return outputs


def _cupy_to_zarr_via_dask(
    image,
    output_path="./example-output.zarr",
    chunk_shape=(2048, 2048, 3),
    zarr_kwargs=dict(overwrite=False, compressor=None),
):
    """Write output to Zarr via GDSStore"""
    store = GDSStore(output_path)

    dask_image = da.from_array(image, chunks=chunk_shape)
    dask_image.to_zarr(store, meta_array=cp.empty(()), **zarr_kwargs)
    return dask_image


def _cupy_to_zarr_kvikio_write_sync(
    image,
    output_path="./example-output.zarr",
    chunk_shape=(2048, 2048, 3),
    zarr_kwargs=dict(overwrite=False, compressor=None),
    backend="kvikio-raw_write",
):
    """Write output to Zarr via GDSStore"""

    # 1.) create a zarr store
    # 2.) call init_array to initialize Zarr .zarry metadata
    #     this will also remove any existing files in output_path when
    #     overwrite = True.

    output_path = os.path.realpath(output_path)
    store = DirectoryStore(output_path)
    init_array(
        store,
        shape=image.shape,
        chunks=chunk_shape,
        dtype=image.dtype,
        **zarr_kwargs,
    )

    c0, c1, c2 = chunk_shape
    s0, s1, s2 = image.shape
    for i0, start0 in enumerate(range(0, s0, c0)):
        for i1, start1 in enumerate(range(0, s1, c1)):
            for i2, start2 in enumerate(range(0, s2, c2)):
                tile = image[
                    start0 : start0 + c0,
                    start1 : start1 + c1,
                    start2 : start2 + c2,
                ]
                if tile.shape == chunk_shape:
                    # copy so the tile is contiguous in memory
                    tile = tile.copy()
                else:
                    pad_width = (
                        (0, c0 - tile.shape[0]),
                        (0, c1 - tile.shape[1]),
                        (0, c2 - tile.shape[2]),
                    )
                    tile = cp.pad(
                        tile, pad_width, mode="constant", constant_values=0
                    )

                chunk_key = ".".join(map(str, (i0, i1, i2)))
                fname = os.path.join(output_path, chunk_key)
                with kvikio.CuFile(fname, "w") as fh:
                    if backend == "kvikio-raw_write":
                        size = fh.raw_write(tile)
                    elif backend == "kvikio-write":
                        size = fh.write(tile)
                    else:
                        raise ValueError(f"unknown backend {backend}")
                    assert size == tile.nbytes
    return


def cupy_to_zarr(
    image,
    output_path="./example-output.zarr",
    chunk_shape=(512, 512, 3),
    n_buffer=16,
    zarr_kwargs=dict(overwrite=False, compressor=None),
    backend="kvikio-pwrite",
):
    """Write output to Zarr via GDSStore"""
    if backend == "dask":
        return _cupy_to_zarr_via_dask(
            image,
            output_path=output_path,
            chunk_shape=chunk_shape,
            zarr_kwargs=zarr_kwargs,
        )

    elif backend in ["kvikio-write", "kvikio-raw_write"]:
        return _cupy_to_zarr_kvikio_write_sync(
            image,
            output_path=output_path,
            chunk_shape=chunk_shape,
            zarr_kwargs=zarr_kwargs,
            backend=backend,
        )
    elif backend != "kvikio-pwrite":
        raise ValueError(f"unrecognized backend: {backend}")

    # 1.) create a zarr store
    # 2.) call init_array to initialize Zarr .zarry metadata
    #     this will also remove any existing files in output_path when
    #     overwrite = True.
    output_path = os.path.realpath(output_path)
    store = DirectoryStore(output_path)
    init_array(
        store,
        shape=image.shape,
        chunks=chunk_shape,
        dtype=image.dtype,
        **zarr_kwargs,
    )

    # asynchronous write using pwrite
    index = 0
    c0, c1, c2 = chunk_shape
    s0, s1, s2 = image.shape
    tile_cache = cp.zeros((n_buffer,) + chunk_shape, dtype=image.dtype)
    for i0, start0 in enumerate(range(0, s0, c0)):
        for i1, start1 in enumerate(range(0, s1, c1)):
            for i2, start2 in enumerate(range(0, s2, c2)):
                index_mod = index % n_buffer
                if index == 0:
                    # initialize lists for storage of future results
                    all_handles = []
                    all_futures = []
                elif index_mod == 0:
                    for fh, future in zip(all_handles, all_futures):
                        if isinstance(future, IOFuture):
                            size = future.get()
                            if size != tile_cache[0].nbytes:
                                raise ValueError(
                                    "failed to write the expected number of "
                                    "bytes"
                                )
                        fh.close()
                        # reset the lists to prepare for the next n_buffer tiles
                        all_futures = []
                        all_handles = []

                tile = image[
                    start0 : start0 + c0,
                    start1 : start1 + c1,
                    start2 : start2 + c2,
                ]
                if tile.shape == chunk_shape:
                    # copy so the tile is contiguous in memory
                    tile_cache[index_mod] = tile
                else:
                    pad_width = (
                        (0, c0 - tile.shape[0]),
                        (0, c1 - tile.shape[1]),
                        (0, c2 - tile.shape[2]),
                    )
                    tile_cache[index_mod] = cp.pad(
                        tile, pad_width, mode="constant", constant_values=0
                    )

                chunk_key = ".".join(map(str, (i0, i1, i2)))
                fname = os.path.join(output_path, chunk_key)

                fh = kvikio.CuFile(fname, "w")
                future = fh.pwrite(tile_cache[index_mod])
                all_futures.append(future)
                all_handles.append(fh)
                # assert written == a.nbytes
                index += 1
    for fh, future in zip(all_handles, all_futures):
        if isinstance(future, IOFuture):
            size = future.get()
            if size != tile_cache[0].nbytes:
                raise ValueError("failed to write the expected number of bytes")
        fh.close()
    return
