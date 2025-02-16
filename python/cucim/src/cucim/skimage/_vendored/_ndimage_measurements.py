import math

import cupy


def _unravel_loop_index_declarations(var_name, ndim, uint_t="unsigned int"):
    """Declare variables needed for unraveling the index to nd coordinates"""
    if ndim == 1:
        code = f"""
        {uint_t} in_coord[1];"""
        return code

    code = f"""
        // variables for unraveling a linear index to a coordinate array
        {uint_t} in_coord[{ndim}];
        {uint_t} temp_floor;"""
    for d in range(ndim):
        code += f"""
        {uint_t} dim{d}_size = {var_name}.shape()[{d}];"""
    return code


def _unravel_loop_index(
    var_name,
    ndim,
    uint_t="unsigned int",
    raveled_index="i",
    omit_declarations=False,
):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    code = (
        ""
        if omit_declarations
        else _unravel_loop_index_declarations(var_name, ndim, uint_t)
    )
    if ndim == 1:
        code = f"""
        in_coord[0] = {raveled_index};\n"""
        return code

    code += f"{uint_t} temp_idx = {raveled_index};"
    for d in range(ndim - 1, 0, -1):
        code += f"""
        temp_floor = temp_idx / dim{d}_size;
        in_coord[{d}] = temp_idx - temp_floor * dim{d}_size;
        temp_idx = temp_floor;"""
    code += """
        in_coord[0] = temp_idx;"""
    return code


@cupy.memoize(for_each_device=True)
def get_bbox_coords_kernel(coord_dtype, ndim, pixels_per_thread=32):
    coord_dtype = cupy.dtype(coord_dtype)

    # maximum number of unique labels this thread might encounter
    # (could use a smaller value for most datasets, but for safety assume
    # every pixel in the window could have a separate label.)
    max_unique = pixels_per_thread

    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    # declare storage for local min/max of each label within the pixel window
    source = f"""
    {uint_t} start_index = {pixels_per_thread}*i;
    // bounding box variables
    {uint_t} bbox_min[{ndim * max_unique}];
    {uint_t} bbox_max[{ndim * max_unique}] = {{0}};
    // initialize minimum coordinate to array size
    for (unsigned int ii = 0; ii < {ndim * max_unique}; ii++) {{
      bbox_min[ii] = image_size;
    }}\n"""
    source += _unravel_loop_index_declarations("image", ndim, uint_t=uint_t)

    # declare inner loop operation
    inner_op = _unravel_loop_index(
        "image",
        ndim=ndim,
        uint_t=uint_t,
        raveled_index="ii",
        omit_declarations=True,
    )
    for d in range(ndim):
        inner_op += f"""
          bbox_min[{ndim}*offset + {d}] = min(
              in_coord[{d}], bbox_min[{ndim}*offset + {d}]);
          bbox_max[{ndim}*offset + {d}] = max(
              in_coord[{d}] + 1, bbox_max[{ndim}*offset + {d}]);"""

    # Find min and max coordinates among the next pixels_per_thread pixels
    source += f"""
      X encountered_labels[{max_unique}] = {{0}};
      X current_label;
      X prev_label = image[start_index];
      int offset = 0;
      encountered_labels[0] = prev_label;
      {uint_t} ii_max = min(start_index + {pixels_per_thread}, image_size);
      for ({uint_t} ii = start_index; ii < ii_max; ii++) {{
        current_label = image[ii];
        if (current_label <= 0) {{ continue; }}
        if (current_label != prev_label) {{
            offset += 1;
            prev_label = current_label;
            encountered_labels[offset] = current_label;
        }}
        // inner loop operation for boundary box min/max
        {inner_op}
      }}\n"""

    # Update the global min/max values with the min/max from the pixel group
    source += """
      for (unsigned int ii = 0; ii <= offset; ii++) {
        X lab = encountered_labels[ii];
        if (lab > 0) {"""
    for d in range(ndim):
        source += f"""
          atomicMin(&bbox[(lab - 1)*{2*ndim} + {2*d}],
                    bbox_min[{ndim}*ii + {d}]);
          atomicMax(&bbox[(lab - 1)*{2*ndim} + {2*d + 1}],
                    bbox_max[{ndim}*ii + {d}]);"""
    source += """
        }
      }\n"""

    inputs = f"raw X image, raw {coord_dtype.name} image_size"
    outputs = f"raw {coord_dtype.name} bbox"
    name = f"cucim_bbox_{ndim}d_{coord_dtype.char}"
    name += f"_batch{pixels_per_thread}"
    return cupy.ElementwiseKernel(inputs, outputs, source, name=name)


def find_objects(input, max_label=0):
    """
    Find objects in a labeled array.

    Parameters
    ----------
    input : ndarray of ints
        Array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        given, the positions of all objects are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label ``l`` corresponds to
        the index ``l-1`` in the returned list.

    See Also
    --------
    label, center_of_mass

    .. warning::

        This function will synchronize the device.
    """

    image = input
    if image.dtype.kind not in "bui":
        raise TypeError(
            f"Input dtype {image.dtype.name} cannot be interpreted as an "
            "integer"
        )
    if max_label < 1:
        max_label = int(image.max())  # synchronize

    # choose 32 or 64-bit coordinate type for atomicMin and atomicMax
    coord_dtype = cupy.uint32 if image.size < 2**32 else cupy.uint64

    # Could potentially expose pixels per thread as a tuning parameter instead
    # of using a fixed value here.
    pixels_per_thread = 32
    bbox_coords_kernel = get_bbox_coords_kernel(
        coord_dtype, image.ndim, pixels_per_thread
    )

    ndim = image.ndim
    # 0 is the correct initial value for coordinate maxima
    bbox_coords = cupy.zeros((max_label, 2 * ndim), dtype=coord_dtype)

    # Initialize value for coordinate minima. Note that the order of
    # coordinates on axis 1 is min_0, max_0, min_1, max_1 ... min_n, max_n.
    int_max = cupy.iinfo(coord_dtype).max
    bbox_coords[:, ::2] = int_max

    # make a copy if the inputs are not already C-contiguous
    if not image.flags.c_contiguous:
        image = cupy.ascontiguousarray(image)

    size = math.ceil(image.size / pixels_per_thread)
    bbox_coords_kernel(image, image.size, bbox_coords, size=size)

    # Copy bounding box coordinates to the CPU to create Python slice objects
    bbox_coords_cpu = cupy.asnumpy(bbox_coords[:max_label, :])  # synchronize

    # Since cuCIM does not use Cython, we use a pure Python conversion to slice
    # objects instead.
    bbox_slices = [
        tuple(slice(int(box[2 * d]), int(box[2 * d + 1])) for d in range(ndim))
        for box in bbox_coords_cpu
    ]
    return bbox_slices
