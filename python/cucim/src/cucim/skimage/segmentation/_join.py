import cupy as cp

from ..util._map_array import ArrayMap, map_array


def join_segmentations(s1, s2, return_mapping: bool = False):
    """Return the join of the two input segmentations.

    The join J of S1 and S2 is defined as the segmentation in which two
    voxels are in the same segment if and only if they are in the same
    segment in *both* S1 and S2.

    Parameters
    ----------
    s1, s2 : numpy arrays
        s1 and s2 are label fields of the same shape.
    return_mapping : bool, optional
        If true, return mappings for joined segmentation labels to the original
        labels.

    Returns
    -------
    j : numpy array
        The join segmentation of s1 and s2.
    map_j_to_s1 : ArrayMap, optional
        Mapping from labels of the joined segmentation j to labels of s1.
    map_j_to_s2 : ArrayMap, optional
        Mapping from labels of the joined segmentation j to labels of s2.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.segmentation import join_segmentations
    >>> s1 = cp.array([[0, 0, 1, 1],
    ...                [0, 2, 1, 1],
    ...                [2, 2, 2, 1]])
    >>> s2 = cp.array([[0, 1, 1, 0],
    ...                [0, 1, 1, 0],
    ...                [0, 1, 1, 1]])
    >>> join_segmentations(s1, s2)
    array([[0, 1, 3, 2],
           [0, 5, 3, 2],
           [4, 5, 5, 3]])
    """
    if s1.shape != s2.shape:
        raise ValueError("Cannot join segmentations of different shape. "
                         f"s1.shape: {s1.shape}, s2.shape: {s2.shape}")
    s1_relabeled, _, backward_map1 = relabel_sequential(s1)
    s2_relabeled, _, backward_map2 = relabel_sequential(s2)
    factor = s2.max() + 1
    j_initial = factor * s1_relabeled + s2_relabeled
    j, _, map_j_to_j_initial = relabel_sequential(j_initial)
    if not return_mapping:
        return j
    # Determine label mapping
    labels_j = cp.unique(j_initial)
    labels_s1_relabeled, labels_s2_relabeled = cp.divmod(labels_j, factor)
    map_j_to_s1 = ArrayMap(map_j_to_j_initial.in_values,
                           backward_map1[labels_s1_relabeled])
    map_j_to_s2 = ArrayMap(map_j_to_j_initial.in_values,
                           backward_map2[labels_s2_relabeled])
    return j, map_j_to_s1, map_j_to_s2


def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : ArrayMap
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The output data type will be the same as `relabeled`.
    inverse_map : ArrayMap
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The output data type will be the same as `label_field`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.segmentation import relabel_sequential
    >>> label_field = cp.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> print(fw)
    ArrayMap:
      1 → 1
      5 → 2
      8 → 3
      42 → 4
      99 → 5
    >>> cp.array(fw)
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> cp.array(inv)
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    array(True)
    >>> (inv[relab] == label_field).all()
    array(True)
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if label_field.min() < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    offset = int(offset)
    in_vals = cp.unique(label_field)
    # Cupy Backend: currently only int32
    if len(in_vals) > cp.iinfo(cp.int32).max:
        raise ValueError(
            "Too many unique values in label_field (current implementation "
            "uses 32-bit indexing)."
        )

    out_val_dtype = cp.min_scalar_type(offset + len(in_vals))
    if int(in_vals[0]) == 0:
        # always map 0 to 0
        out_vals = cp.concatenate(
            [
                cp.asarray([0], dtype=out_val_dtype),
                cp.arange(
                    offset, offset + len(in_vals) - 1, dtype=out_val_dtype
                ),
            ]
        )
    else:
        out_vals = cp.arange(offset, offset + len(in_vals), dtype=out_val_dtype)
    input_type = label_field.dtype
    if input_type.kind not in "iu":
        raise TypeError("label_field must have an integer dtype")

    # Some logic to determine the output type:
    #  - we don't want to return a smaller output type than the input type,
    #  ie if we get uint32 as labels input, don't return a uint8 array.
    #  - but, in some cases, using the input type could result in overflow. The
    #  input type could be a signed integer (e.g. int32) but
    #  `np.min_scalar_type` will always return an unsigned type. We check for
    #  that by casting the largest output value to the input type. If it is
    #  unchanged, we use the input type, else we use the unsigned minimum
    #  required type
    out_max = int(out_vals[-1])
    required_type = cp.min_scalar_type(out_max)
    if input_type.itemsize < required_type.itemsize:
        output_type = required_type
    else:
        if out_max <= cp.iinfo(input_type).max:
            output_type = input_type
        else:
            output_type = required_type
    out_array = cp.empty(label_field.shape, dtype=output_type)
    out_vals = out_vals.astype(output_type, copy=False)
    map_array(label_field, in_vals, out_vals, out=out_array)
    fw_map = ArrayMap(in_vals, out_vals)
    inv_map = ArrayMap(out_vals, in_vals)
    return out_array, fw_map, inv_map
