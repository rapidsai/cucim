"""A vendored subset of cupyx.scipy.ndimage._filters_core"""

import cupy
import numpy

from cucim.skimage._vendored import _ndimage_util as _util

includes = r'''
// workaround for HIP: line begins with #include
#include <type_traits>  // let Jitify handle this
#include <cupy/math_constants.h>
'''


_CAST_FUNCTION = """
// Implements a casting function to make it compatible with scipy
// Use like cast<to_type>(value)
template<> struct std::is_floating_point<float16> : std::true_type {};
template<> struct std::is_signed<float16> : std::true_type {};
template<class T> struct std::is_signed<complex<T>> : std::is_signed<T> {};

template <class B, class A>
__device__ __forceinline__
typename std::enable_if<(!std::is_floating_point<A>::value
                         || std::is_signed<B>::value), B>::type
cast(A a) { return (B)a; }

template <class B, class A>
__device__ __forceinline__
typename std::enable_if<(std::is_floating_point<A>::value
                         && (!std::is_signed<B>::value)), B>::type
cast(A a) { return (a >= 0) ? (B)a : -(B)(-a); }

template <class T>
__device__ __forceinline__ bool nonzero(T x) { return x != static_cast<T>(0); }
"""


def _generate_nd_kernel(name, pre, found, post, mode, w_shape, int_type,
                        offsets, cval, ctype='X', preamble='', options=(),
                        has_weights=True, has_structure=False, has_mask=False,
                        binary_morphology=False, all_weights_nonzero=False):
    # Currently this code uses CArray for weights but avoids using CArray for
    # the input data and instead does the indexing itself since it is faster.
    # If CArray becomes faster than follow the comments that start with
    # CArray: to switch over to using CArray for the input data as well.

    ndim = len(w_shape)
    in_params = 'raw X x'
    if has_weights:
        in_params += ', raw W w'
    if has_structure:
        in_params += ', raw S s'
    if has_mask:
        in_params += ', raw M mask'
    out_params = 'Y y'

    # for filters, "wrap" is a synonym for "grid-wrap"
    mode = 'grid-wrap' if mode == 'wrap' else mode

    # CArray: remove xstride_{j}=... from string
    size = ('%s xsize_{j}=x.shape()[{j}], ysize_{j} = _raw_y.shape()[{j}]'
            ', xstride_{j}=x.strides()[{j}];' % int_type)
    sizes = [size.format(j=j) for j in range(ndim)]
    inds = _util._generate_indices_ops(ndim, int_type, offsets)
    # CArray: remove expr entirely
    expr = ' + '.join(['ix_{}'.format(j) for j in range(ndim)])

    ws_init = ws_pre = ws_post = ''
    if has_weights or has_structure:
        ws_init = 'int iws = 0;'
        if has_structure:
            ws_pre = 'S sval = s[iws];\n'
        if has_weights:
            ws_pre += 'W wval = w[iws];\n'
            if not all_weights_nonzero:
                ws_pre += 'if (nonzero(wval))'
        ws_post = 'iws++;'

    loops = []
    for j in range(ndim):
        if w_shape[j] == 1:
            # CArray: string becomes 'inds[{j}] = ind_{j};', remove (int_)type
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.
                         format(j=j, type=int_type))
        else:
            boundary = _util._generate_boundary_condition_ops(
                mode, 'ix_{}'.format(j), 'xsize_{}'.format(j), int_type)
            # CArray: last line of string becomes inds[{j}] = ix_{j};
            loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=w_shape[j], boundary=boundary, type=int_type))

    # CArray: string becomes 'x[inds]', no format call needed
    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        cond = ' || '.join(['(ix_{} < 0)'.format(j) for j in range(ndim)])

    if cval is numpy.nan:
        cval = 'CUDART_NAN'
    elif cval == numpy.inf:
        cval = 'CUDART_INF'
    elif cval == -numpy.inf:
        cval = '-CUDART_INF'

    if binary_morphology:
        found = found.format(cond=cond, value=value)
    else:
        if mode == 'constant':
            value = '(({cond}) ? cast<{ctype}>({cval}) : {value})'.format(
                cond=cond, ctype=ctype, cval=cval, value=value)
        found = found.format(value=value)

    # CArray: replace comment and next line in string with
    #   {type} inds[{ndim}] = {{0}};
    # and add ndim=ndim, type=int_type to format call
    operation = '''
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {ws_init}
    {pre}
    {loops}
        // inner-most loop
        {ws_pre} {{
            {found}
        }}
        {ws_post}
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               ws_init=ws_init, ws_pre=ws_pre, ws_post=ws_post,
               loops='\n'.join(loops), found=found, end_loops='}' * ndim)

    mode_str = mode.replace('-', '_')  # avoid potential hyphen in kernel name
    name = 'cupyx_scipy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode_str, '_'.join(['{}'.format(x) for x in w_shape]))
    if all_weights_nonzero:
        name += '_all_nonzero'
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    if has_mask:
        name += '_with_mask'
    preamble = includes + _CAST_FUNCTION + preamble
    options += ('--std=c++11', '-DCUPY_USE_JITIFY')
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)
