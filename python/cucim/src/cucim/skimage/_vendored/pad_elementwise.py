import cupy


def _pad_boundary_ops(mode, var_name, size, int_t="int", no_singleton=False):
    T = 'int' if int_t == 'int' else 'long long'
    min_func = 'min'
    max_func = 'max'
    if mode == 'constant':
        ops = f'''
        if (({var_name} < 0) || {var_name} >= {size}) {{
            {var_name} = -1;
        }}'''
    elif mode == 'symmetric':
        ops = f'''
            if ({var_name} < 0) {{
                {var_name} = - 1 -{var_name};
            }}
            {var_name} %= {size} * 2;
            {var_name} = {min_func}({var_name}, 2 * {size} - 1 - {var_name});
        '''
    elif mode == 'reflect':
        ops = f'''
        if ({size} == 1) {{
            {var_name} = 0;
        }} else {{
            if ({var_name} < 0) {{
                {var_name} = -{var_name};
            }}
            if ({var_name} >= {size}) {{
                {var_name} = 1 + ({var_name} - 1) % (({size} - 1) * 2);
                {var_name} = {min_func}({var_name},
                                        2 * {size} - 2 - {var_name});
            }}
        }}'''  # noqa
    elif mode == 'reflect_no_singleton_dim':
        # the same as reflect, but without the extra `{size} == 1` check
        ops = f'''
        if ({var_name} < 0) {{
            {var_name} = -{var_name};
        }}
        if ({var_name} >= {size}) {{
            {var_name} = 1 + ({var_name} - 1) % (({size} - 1) * 2);
            {var_name} = {min_func}({var_name}, 2 * {size} - 2 - {var_name});
        }}
        '''
    elif mode == 'edge':
        ops = f'''
        {var_name} = {min_func}(
            {max_func}(static_cast<{T}>({var_name}), static_cast<{T}>(0)),
            static_cast<{T}>({size} - 1));
        '''
    elif mode == 'wrap':
        ops = f'''
        {var_name} %= {size};
        if ({var_name} < 0) {{
            {var_name} += {size};
        }}
        '''
    return ops + "\n"


def _generate_size_vars(
    ndim, arr_name='arr', size_prefix='size', int_type='int'
):
    """Store shape of a raw array into individual variables.

    Examples
    --------
    >>> print(_generate_size_vars(3, 'arr', 'size', 'int'))
    int size_0 = arr.shape()[0];
    int size_1 = arr.shape()[1];
    int size_2 = arr.shape()[2];
    """
    set_size_vars = [f'{int_type} {size_prefix}_{i} = {arr_name}.shape()[{i}];'
                     for i in range(ndim)]
    return '\n'.join(set_size_vars) + '\n'


def _generate_stride_vars(
    ndim, arr_name='arr', size_prefix='stride', int_type='int'
):
    """Store stride (in bytes) of a raw array into individual variables.

    Examples
    --------
    >>> print(_generate_size_vars(3, 'arr', 'size', 'int'))
    int stride_0 = arr.strides()[0];
    int stride_1 = arr.strides()[1];
    int stride_2 = arr.strides()[2];
    """
    set_size_vars = [
        f'{int_type} {size_prefix}_{i} = {arr_name}.strides()[{i}];'
        for i in range(ndim)
    ]
    return '\n'.join(set_size_vars) + '\n'


def _generate_indices_ops(
    ndim, size_prefix='size', int_type='int', index_prefix='ind', order='C',
):
    """Generate indices based existing variables.

    Assumes variables f'{size_prefix}_{i}' has the size along axis, i.

    Examples
    --------
    >>> print(_generate_indices_ops(3, 'size', 'int', 'ind', 'C'))
    int _i = i;
    int ind_2 = _i % size_2; _i /= size_2;
    int ind_1 = _i % size_1; _i /= size_1;
    int ind_0 = _i;
    """
    if order == 'C':
        _range = range(ndim - 1, 0, -1)
        idx_largest_stride = 0
    elif order == 'F':
        _range = range(ndim - 1)
        idx_largest_stride = ndim - 1
    else:
        raise ValueError(f"Unknown order: {order}. Must be one of {'C', 'F'}.")
    body = [f'{int_type} {index_prefix}_{j} = _i % {size_prefix}_{j}; _i /= {size_prefix}_{j};'  # noqa
            for j in _range]
    body = '\n'.join(body)
    code = f'{int_type} _i = i;\n'
    code += body + '\n'
    code += f'{int_type} {index_prefix}_{idx_largest_stride} = _i;\n'
    return code


def _gen_raveled(ndim, stride_prefix='stride', index_prefix='i', order=None):
    """Generate raveled index for c-ordered memory layout

    For index_prefix='i', the indices are (i_0, i_1, ....)
    For stride_prefix='stride', the stride is (stride_0, stride_1, ....)
    """
    return ' + '.join(
        f'{stride_prefix}_{j} * {index_prefix}_{j}' for j in range(ndim)
    )


def _get_pad_kernel_code(pad_starts, int_type='int', mode='edge', order='C'):
    # variables storing shape of the output array
    ndim = len(pad_starts)
    out_size_prefix = 'shape'
    operation = _generate_size_vars(
        ndim, arr_name='out', size_prefix=out_size_prefix, int_type=int_type
    )

    # variables storing shape of the input array
    in_size_prefix = 'ishape'
    in_stride_prefix = 'istride'
    operation += _generate_size_vars(
        ndim, arr_name='arr', size_prefix=in_size_prefix, int_type=int_type
    )
    operation += _generate_stride_vars(
        ndim, arr_name='arr', size_prefix=in_stride_prefix, int_type=int_type
    )

    # unraveled indices into the output array
    out_index_prefix = 'oi'
    # Note: Regardless of actual memory layout, need order='C' here to match
    #       the behavior of the index raveling used by ElementwiseKernel.
    operation += _generate_indices_ops(
        ndim, size_prefix=out_size_prefix, int_type=int_type,
        index_prefix=out_index_prefix, order='C'
    )

    # compute unraveled indices into the input array
    # (i_0, i_1, ...)
    in_index_prefix = 'i'
    operation += '\n'.join(
        [f'{int_type} {in_index_prefix}_{j} = {out_index_prefix}_{j} - {pad_starts[j]};'  # noqa
         for j in range(ndim)]
    )
    operation += '\n'
    input_indices = tuple(f'{in_index_prefix}_{j}' for j in range(ndim))

    # impose boundary condition

    range_cond = " || ".join(
        f"({coord} < 0) || ({coord} >= {in_size_prefix}_{j})"
        for j, coord in enumerate(input_indices)
    )
    operation += f"bool range_cond = {range_cond};"
    operation += "if (range_cond) {\n"
    if mode == "constant":
        for j, coord in enumerate(input_indices):
            operation += _pad_boundary_ops(
                mode, coord, f"{in_size_prefix}_{j}", int_type
            )
            operation += f"""
                if ({coord} == -1) {{
                    out[i] = static_cast<F>(cval);
                    return;
                }}
            """
    else:
        for j, coord in enumerate(input_indices):
            operation += _pad_boundary_ops(
                mode, coord, f"{in_size_prefix}_{j}", int_type
            )
    operation += "}\n"

    raveled_idx = _gen_raveled(
        ndim, stride_prefix=in_stride_prefix, index_prefix=in_index_prefix,
        order=order,
    )
    operation += f"""
    // set output based on raveled index into the input array
    const char* char_arr = reinterpret_cast<const char*>(&arr[0]);
    out[i] = *reinterpret_cast<const F*>(char_arr + {raveled_idx});
    """
    return operation


@cupy._util.memoize(for_each_device=True)
def _get_pad_kernel(pad_starts, int_type='int', mode='edge', order='C'):
    in_params = "raw F arr"
    if mode == 'constant':
        in_params += ", float64 cval"

    kernel_name = f"pad_{len(pad_starts)}d_order{order}_{mode}"
    if int_type != "int":
        kernel_name += f"_{int_type.replace(' ', '_')}_idx"

    return cupy.ElementwiseKernel(
        in_params=in_params,
        out_params="raw F out",
        operation=_get_pad_kernel_code(pad_starts, int_type, mode, order),
        name=kernel_name)
