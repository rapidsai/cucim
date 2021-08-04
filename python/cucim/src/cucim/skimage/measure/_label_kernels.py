"""Kernels for scikit-image label.

These are copied from CuPy, with modification to add a greyscale_mode
parameter as needed for scikit-image.

"""
import cupy
import numpy


def _label(x, structure, y, greyscale_mode=False):
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]
    dirs = cupy.array(dirs, dtype=numpy.int32)
    ndirs = indxs.shape[0]
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    count = cupy.zeros(2, dtype=numpy.int32)
    _kernel_init()(x, y)
    if greyscale_mode:
        _kernel_connect(True)(x, y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    else:
        _kernel_connect(False)(y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    _kernel_count()(y, count, size=y.size)
    maxlabel = int(count[0])  # synchronize
    labels = cupy.empty(maxlabel, dtype=numpy.int32)
    _kernel_labels()(y, count, labels, size=y.size)
    _kernel_finalize()(maxlabel, cupy.sort(labels), y, size=y.size)
    return maxlabel


"""
Elementwise kernels for use by label
"""


def _kernel_init():
    return cupy.ElementwiseKernel(
        "X x",
        "Y y",
        "if (x == 0) { y = -1; } else { y = i; }",
        "cucim_skimage_measure_label_init",
    )


def _kernel_connect(greyscale_mode=False, int_t="int"):
    """
    Notes
    -----
    dirs is a (n_neig//2, ndim) of relative offsets to the neighboring voxels.
    For example, for structure = np.ones((3, 3)):
        dirs = array([[-1, -1],
                      [-1,  0],
                      [-1,  1],
                      [ 0, -1]], dtype=int32)
    (Implementation assumes a centro-symmetric structure)
    ndirs = dirs.shape[0]

    In the dirs loop below, there is a loop over the ndim neighbors:
        Here, index j corresponds to the current pixel and k is the current
        neighbor location.
    """
    in_params = "raw int32 shape, raw int32 dirs, int32 ndirs, int32 ndim"
    if greyscale_mode:
        # greyscale mode -> different values receive different labels
        x_condition = "if (x[k] != x[j]) continue;"
        in_params = "raw X x, " + in_params
    else:
        # binary mode -> all non-background voxels treated the same
        x_condition = ""

    # Note: atomicCAS is implemented for int, unsigned short, unsigned int, and
    # unsigned long long

    code = """
        if (y[i] < 0) continue;
        for (int dr = 0; dr < ndirs; dr++) {{
            {int_t} j = i;
            {int_t} rest = j;
            {int_t} stride = 1;
            {int_t} k = 0;
            for (int dm = ndim-1; dm >= 0; dm--) {{
                int pos = rest % shape[dm] + dirs[dm + dr * ndim];
                if (pos < 0 || pos >= shape[dm]) {{
                    k = -1;
                    break;
                }}
                k += pos * stride;
                rest /= shape[dm];
                stride *= shape[dm];
            }}
            if (k < 0) continue;
            if (y[k] < 0) continue;
            {x_condition}
            while (1) {{
                while (j != y[j]) {{ j = y[j]; }}
                while (k != y[k]) {{ k = y[k]; }}
                if (j == k) break;
                if (j < k) {{
                    {int_t} old = atomicCAS( &y[k], (Y)k, (Y)j );
                    if (old == k) break;
                    k = old;
                }}
                else {{
                    {int_t} old = atomicCAS( &y[j], (Y)j, (Y)k );
                    if (old == j) break;
                    j = old;
                }}
            }}
        }}
        """.format(
        x_condition=x_condition, int_t=int_t
    )

    return cupy.ElementwiseKernel(
        in_params, "raw Y y", code, "cucim_skimage_measure_label_connect",
    )


def _kernel_count():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw int32 count",
        """
        if (y[i] < 0) continue;
        int j = i;
        while (j != y[j]) { j = y[j]; }
        if (j != i) y[i] = j;
        else atomicAdd(&count[0], 1);
        """,
        "cucim_skimage_measure_label_count",
    )


def _kernel_labels():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw int32 count, raw int32 labels",
        """
        if (y[i] != i) continue;
        int j = atomicAdd(&count[1], 1);
        labels[j] = i;
        """,
        "cucim_skimage_measure_label_labels",
    )


def _kernel_finalize():
    return cupy.ElementwiseKernel(
        "int32 maxlabel",
        "raw int32 labels, raw Y y",
        """
        if (y[i] < 0) {
            y[i] = 0;
            continue;
        }
        int yi = y[i];
        int j_min = 0;
        int j_max = maxlabel - 1;
        int j = (j_min + j_max) / 2;
        while (j_min < j_max) {
            if (yi == labels[j]) break;
            if (yi < labels[j]) j_max = j - 1;
            else j_min = j + 1;
            j = (j_min + j_max) / 2;
        }
        y[i] = j + 1;
        """,
        "cucim_skimage_measure_label_finalize",
    )
