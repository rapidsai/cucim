import math
from collections import namedtuple
from textwrap import dedent

import cupy as cp
import numpy as np

from .._shared.utils import _to_np_mode

try:
    # Device Attributes require CuPy > 6.0.b3
    d = cp.cuda.Device()
    cuda_MaxBlockDimX = d.attributes["MaxBlockDimX"]
    cuda_MaxGridDimX = d.attributes["MaxGridDimX"]
except (cp.cuda.runtime.CUDARuntimeError, AttributeError):
    # guess
    cuda_MaxBlockDimX = 1024
    cuda_MaxGridDimX = 2147483647


def _dtype_to_CUDA_int_type(dtype):
    cpp_int_types = {
        cp.uint8: 'unsigned char',
        cp.uint16: 'unsigned short',
        cp.uint32: 'unsigned int',
        cp.uint64: 'unsigned long long',
        cp.int8: 'signed char',
        cp.int16: 'short',
        cp.int32: 'int',
        cp.int64: 'long long',
    }
    dtype = cp.dtype(dtype)
    if dtype.type not in cpp_int_types:
        raise ValueError(f"unrecognized dtype: {dtype.type}")
    return cpp_int_types[dtype.type]


def _get_hist_dtype(footprint_shape):
    """Determine C++ type and cupy.dtype to use for the histogram."""
    max_possible_count = math.prod(footprint_shape)

    if max_possible_count < 128:
        dtype = cp.int8
    elif max_possible_count < 32768:
        dtype = cp.int16
    else:
        dtype = cp.int32
    return _dtype_to_CUDA_int_type(dtype), dtype


def _gen_global_definitions(
    image_t='unsigned char',
    hist_offset=0,
    hist_int_t='int',
    hist_size=256,
    hist_size_coarse=8
):
    """Generate C++ #define statements needed for the CUDA kernels.

    The definitions used depend on the number of histogram bins and the
    histogram data type.
    """

    if hist_size % hist_size_coarse != 0:
        raise ValueError(
            "`hist_size` must be a multiple of `hist_size_coarse`"
        )
    hist_size_fine = hist_size // hist_size_coarse
    log2_coarse = math.log2(hist_size_coarse)
    log2_fine = math.log2(hist_size_fine)
    if abs(math.remainder(log2_coarse, 1)) > 1e-7:
        raise ValueError("log2_coarse must be a power of two")
    elif abs(math.remainder(log2_fine, 1)) > 1e-7:
        raise ValueError("log2_fine must be a power of two")
    else:
        log2_coarse = round(log2_coarse)
        log2_fine = round(log2_fine)

    global_defs = f"""
#define HIST_SIZE {hist_size}
#define HIST_SIZE_COARSE {hist_size_coarse}
#define HIST_SIZE_FINE {hist_size_fine}
#define HIST_INT_T {hist_int_t}
#define HIST_OFFSET {hist_offset}
#define IMAGE_T {image_t}
#define LOG2_COARSE {log2_coarse}
#define LOG2_FINE {log2_fine}
    """
    return global_defs


# This preamble code is common across all generated kernels
preamble_common = """

__device__ void histogramAddAndSubCoarse(HIST_INT_T* H, const HIST_INT_T* hist_colAdd, const HIST_INT_T* hist_colSub){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE){
        H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
    }
}

__device__ void histogramMultipleAddCoarse(HIST_INT_T* H, const HIST_INT_T* hist_col, int histCount){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE){
        HIST_INT_T temp=H[tx];
        for(int i=0; i<histCount; i++)
            temp+=hist_col[(i<<LOG2_COARSE)+tx];
        H[tx]=temp;
    }
}

__device__ void histogramClearCoarse(HIST_INT_T* H){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE){
        H[tx]=0;
    }
}

__device__ void histogramAddCoarse(HIST_INT_T* H, const HIST_INT_T* hist_col){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE){
        H[tx]+=hist_col[tx];
    }
}

__device__ void histogramSubCoarse(HIST_INT_T* H, const HIST_INT_T* hist_col){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE){
        H[tx]-=hist_col[tx];
    }
}

__device__ void histogramAddFine(HIST_INT_T* H, const HIST_INT_T* hist_col){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_FINE){
        H[tx]+=hist_col[tx];
    }
}

__device__ void histogramAddAndSubFine(HIST_INT_T* H, const HIST_INT_T* hist_colAdd, const HIST_INT_T* hist_colSub){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_FINE){
        H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
    }
}

__device__ void histogramClearFine(HIST_INT_T* H){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_FINE){
        H[tx]=0;
    }
}

__device__ void lucClearCoarse(int* luc){
    int tx = threadIdx.x;
    if (tx<HIST_SIZE_COARSE)
        luc[tx]=0;
}

#define scanNeighbor(array, range, index, threadIndex)             \\
        {                                                          \\
            HIST_INT_T v = 0;                                             \\
            if (index <= threadIndex && threadIndex < range)       \\
                v = array[threadIndex] + array[threadIndex-index]; \\
            __syncthreads();                                       \\
            if (index <= threadIndex && threadIndex < range)       \\
                array[threadIndex] = v;                            \\
        }

#define findMedian(array, range, threadIndex, result, count, position) \\
        if (threadIndex < range)                                       \\
        {                                                              \\
            if (array[threadIndex+1] > position && array[threadIndex] <= position) \\
            {                                                          \\
                *result = threadIndex+1;                               \\
                *count  = array[threadIndex];                          \\
            }                                                          \\
        }


"""  # noqa


# TODO: look into using CUB's Block-wise collectives (e.g. BlockScan)
def _gen_preamble_median(hist_size_coarse, hist_size_fine):
    """Generate bin size-dependent reduction functions.

    This helper generates the C++ code for the following two functions.

    histogramMedianParCoarseLookupOnly
    histogramMedianParFineLookupOnly

    For each of these the number of calls to scanNeighbor is equal to
    log2 of the number of bins.
    """
    n_log2_coarse = math.log2(hist_size_coarse)
    if hist_size_coarse < 2 or n_log2_coarse % 1.0 != 0:
        raise ValueError("hist_size_coarse must be a positive power of 2")

    n_log2_fine = math.log2(hist_size_fine)
    if hist_size_fine < 2 or n_log2_fine % 1.0 != 0:
        raise ValueError("hist_size_fine must be a positive power of 2")

    ops = """

        __device__ void histogramMedianParCoarseLookupOnly(HIST_INT_T* H, HIST_INT_T* Hscan, const int medPos, int* retval, int* countAtMed){
            int tx=threadIdx.x;
            *retval=*countAtMed=0;
            if(tx<HIST_SIZE_COARSE){
                Hscan[tx]=H[tx];
            }
            __syncthreads();\n"""  # noqa

    for d in range(round(n_log2_coarse)):
        ops += f"""
            scanNeighbor(Hscan, {hist_size_coarse}, {2**d}, tx);
            __syncthreads();"""
    ops += f"""
            findMedian(Hscan, {hist_size_coarse - 1}, tx, retval, countAtMed, medPos);
        }}"""  # noqa

    ops += """

        __device__ void histogramMedianParFineLookupOnly(HIST_INT_T* H, HIST_INT_T* Hscan, const int medPos, int* retval, int* countAtMed){
            int tx=threadIdx.x;
            *retval=*countAtMed=0;
            if(tx<HIST_SIZE_FINE){
                Hscan[tx]=H[tx];
            }
            __syncthreads();\n"""  # noqa

    for d in range(round(n_log2_fine)):
        ops += f"""
            scanNeighbor(Hscan, {hist_size_fine}, {2**d}, tx);
            __syncthreads();"""
    ops += f"""
            findMedian(Hscan, {hist_size_fine - 1}, tx, retval, countAtMed, medPos);
        }}\n"""  # noqa

    return dedent(ops)


def _gen_median_kernel_preamble(
    image_t, hist_offset, hist_int_t, hist_size=256, hist_size_coarse=8
):
    src = _gen_global_definitions(
        image_t=image_t,
        hist_offset=hist_offset,
        hist_int_t=hist_int_t,
        hist_size=hist_size,
        hist_size_coarse=hist_size_coarse,
    )
    src += preamble_common
    hist_size_fine = hist_size // hist_size_coarse
    src += _gen_preamble_median(
        hist_size_coarse=hist_size_coarse, hist_size_fine=hist_size_fine
    )
    return src


rank_filter_kernel = """

extern "C" __global__
void cuRankFilterMultiBlock(IMAGE_T* src, IMAGE_T* dest, HIST_INT_T* histPar, HIST_INT_T* coarseHistGrid, int r0, int r1, int medPos_, int rows, int cols)
{
    __shared__ HIST_INT_T HCoarse[HIST_SIZE_COARSE];
    __shared__ HIST_INT_T HCoarseScan[HIST_SIZE_FINE];
    __shared__ HIST_INT_T HFine[HIST_SIZE_COARSE][HIST_SIZE_FINE];

    __shared__ int luc[HIST_SIZE_COARSE];

    __shared__ int firstBin, countAtMed, retval;

    // extract values from params array
    const int row_stride = cols;  // stride (in elements) along axis 0

    int extraRowThread = rows % gridDim.x;
    int doExtraRow = blockIdx.x < extraRowThread;
    int startRow = 0, stopRow = 0;
    int rowsPerBlock = rows/gridDim.x + doExtraRow;

    // The following code partitions the work to the blocks. Some blocks will do one row more
    // than other blocks. This code is responsible for doing that balancing
    if(doExtraRow){
        startRow=rowsPerBlock*blockIdx.x;
        stopRow=min(rows, startRow+rowsPerBlock);
    }
    else{
        startRow=(rowsPerBlock+1)*extraRowThread+(rowsPerBlock)*(blockIdx.x-extraRowThread);
        stopRow=min(rows, startRow+rowsPerBlock);
    }

    HIST_INT_T* hist = histPar + cols*HIST_SIZE*blockIdx.x;
    HIST_INT_T* histCoarse = coarseHistGrid + cols*HIST_SIZE_COARSE*blockIdx.x;

    if (blockIdx.x==(gridDim.x-1))
        stopRow=rows;
    __syncthreads();
    int initNeeded=0, initStartRow, initStopRow;
    HIST_INT_T initVal;

    if(blockIdx.x==0){
        // Note: skips one iteration in the initialization loop by starting at
        // row 1 instead of 0 and using initVal r0+2 instead of r0+1.
        initNeeded=1; initVal=r0+2; initStartRow=1;  initStopRow=r0;
    }
    else if (startRow<(r0+2)){
        initNeeded=1; initVal=r0+2-startRow; initStartRow=1; initStopRow=r0+startRow;
    }
    else{
        initNeeded=0; initVal=0; initStartRow=startRow-(r0+1);   initStopRow=r0+startRow;
    }
   __syncthreads();


    // In the original algorithm an initialization phase was required as part of the window was outside the
    // image. In this parallel version, the initializtion is required for all thread blocks that part
    // of the median filter is outside the window.
    // For all threads in the block the same code will be executed.
    if (initNeeded){
        for (int j=threadIdx.x; j<(cols); j+=blockDim.x){
            hist[j*HIST_SIZE + src[j] + HIST_OFFSET]=initVal;
            histCoarse[j*HIST_SIZE_COARSE + ((src[j] + HIST_OFFSET)>>LOG2_FINE)]=initVal;
        }
    }
    __syncthreads();

    // For all remaining rows in the median filter, add the values to the the histogram
    for (int j=threadIdx.x; j<cols; j+=blockDim.x){
        for(int i=initStartRow; i<initStopRow; i++){
            int pos=min(i,rows-1);
            hist[j*HIST_SIZE + src[pos * row_stride + j] + HIST_OFFSET]++;
            histCoarse[j*HIST_SIZE_COARSE + ((src[pos * row_stride + j] + HIST_OFFSET)>>LOG2_FINE)]++;
        }
    }
    __syncthreads();
     // Going through all the rows that the block is responsible for.
     int inc=blockDim.x*HIST_SIZE;
     int incCoarse=blockDim.x*HIST_SIZE_COARSE;
     for(int i=startRow; i< stopRow; i++){
         // For every new row that is started the global histogram for the entire window is restarted.

         histogramClearCoarse(HCoarse);
         lucClearCoarse(luc);
         // Computing some necessary indices
         int possub=max(0,i-r0-1),posadd=min(rows-1,i+r0);
         int histPos=threadIdx.x*HIST_SIZE;
         int histCoarsePos=threadIdx.x*HIST_SIZE_COARSE;
         // Going through all the elements of a specific row. For each histogram, a value is taken out and
         // one value is added.
         for (int j=threadIdx.x; j<cols; j+=blockDim.x){
            hist[histPos+ src[possub * row_stride + j] + HIST_OFFSET]--;
            hist[histPos+ src[posadd * row_stride + j] + HIST_OFFSET]++;
            histCoarse[histCoarsePos+ ((src[possub * row_stride + j] + HIST_OFFSET)>>LOG2_FINE) ]--;
            histCoarse[histCoarsePos+ ((src[posadd * row_stride + j] + HIST_OFFSET)>>LOG2_FINE) ]++;

            histPos+=inc;
            histCoarsePos+=incCoarse;
         }
        __syncthreads();

        histogramMultipleAddCoarse(HCoarse, histCoarse, 2*r1+1);
        int cols_m_1=cols-1;

        for(int j=r1;j<cols-r1;j++){
            int possub=max(j-r1, 0);
            int posadd=min(j+1+r1, cols_m_1);
            int medPos=medPos_;
            __syncthreads();

            histogramMedianParCoarseLookupOnly(HCoarse, HCoarseScan, medPos, &firstBin, &countAtMed);
            __syncthreads();

            int loopIndex = luc[firstBin];
            if (loopIndex <= (j-r1))
            {
                histogramClearFine(HFine[firstBin]);
                for ( loopIndex = j-r1; loopIndex < min(j+r1+1,cols); loopIndex++ ){
                    histogramAddFine(HFine[firstBin], hist+(loopIndex*HIST_SIZE+(firstBin<<LOG2_FINE) ) );
                }
            }
            else{
                for ( ; loopIndex < (j+r1+1);loopIndex++ ) {
                    histogramAddAndSubFine(HFine[firstBin],
                    hist+(min(loopIndex,cols_m_1)*HIST_SIZE+(firstBin<<LOG2_FINE) ),
                    hist+(max(loopIndex-2*r1-1,0)*HIST_SIZE+(firstBin<<LOG2_FINE) ) );
                    __syncthreads();
                }
            }
            __syncthreads();
            luc[firstBin] = loopIndex;

            int leftOver=medPos-countAtMed;
            if(leftOver>=0){
                histogramMedianParFineLookupOnly(HFine[firstBin], HCoarseScan, leftOver, &retval, &countAtMed);
            }
            else retval=0;
            __syncthreads();

            if (threadIdx.x==0){
                dest[i * row_stride + j]=(firstBin<<LOG2_FINE) + retval - HIST_OFFSET;
            }
            histogramAddAndSubCoarse(HCoarse, histCoarse+(int)(posadd<<LOG2_COARSE), histCoarse+(int)(possub<<LOG2_COARSE));

            __syncthreads();
        }
         __syncthreads();
    }
}
"""  # noqa


@cp.memoize(for_each_device=True)
def _get_median_rawkernel(
    image_t, hist_offset, hist_int_t, hist_size=256, hist_size_coarse=8
):
    preamble = _gen_median_kernel_preamble(
        image_t=image_t,
        hist_offset=hist_offset,
        hist_int_t=hist_int_t,
        hist_size=hist_size,
        hist_size_coarse=hist_size_coarse,
    )
    return cp.RawKernel(
        code=preamble + rank_filter_kernel,
        name="cuRankFilterMultiBlock",
    )


def _check_shared_memory_requirement_bytes(
    hist_dtype, hist_size_coarse, hist_size_fine
):
    """computes amount of shared memory required by cuRankFilterMultiBlock"""
    s = np.dtype(hist_dtype).itemsize
    shared_size = hist_size_coarse * s                     # for HCoarse
    shared_size += hist_size_fine * s                      # for HCoarseScane
    shared_size += hist_size_coarse * hist_size_fine * s   # for HFine
    shared_size += hist_size_coarse * 4                    # for luc
    shared_size += 12                                      # three more ints
    return shared_size


def _check_global_scratch_space_size(
    image_shape, hist_size, hist_size_coarse, hist_dtype, partitions
):
    """Determine amount of histogram scratch space that will be allocated.

    Returns the total size in bytes.
    """
    n_last = image_shape[-1]  # this is the contiguous memory dimension
    n_fine = n_last * hist_size * partitions
    n_coarse = n_last * hist_size_coarse * partitions
    return (n_fine + n_coarse) * cp.dtype(hist_dtype).itemsize


def _can_use_histogram(image, footprint):
    """Validate compatibility with histogram-based median.

    Parameters
    ----------
    image : cupy.ndarray
        The image to filter.
    footprint : cupy.ndarray
        The filter footprint.

    Returns
    -------
    compatible : bool
        Indicates whether the provided image and footprint are compatible with
        the histogram-based median.
    reason : str
        Description of the reason for the incompatibility
    """
    # only 2D uint8 images are supported
    if image.ndim != 2:
        return False, "only 2D images are supported"
    if image.dtype not in [cp.uint8, cp.uint16, cp.int8, cp.int16]:
        return False, "Only 8 and 16-bit integer image types (signed or "
        "unsigned)."

    # only odd-sized footprints are supported
    if not all(s % 2 == 1 for s in footprint.shape):
        return False, "footprint must have odd size on both axes"

    if any(s == 1 for s in footprint.shape):
        return False, "footprint must have size >= 3"

    # footprint radius can't be larger than the image
    # TODO: need to check if we need this exact restriction
    #       (may be specific to OpenCV's boundary handling)
    radii = tuple(s // 2 for s in footprint.shape)
    if any(r > s for r, s in zip(radii, image.shape)):
        return False, "footprint half-width cannot exceed the image extent"

    # only fully populated footprint is supported
    if not np.all(footprint):  # synchronizes!
        return False, "footprint must be 1 everywhere"

    return True, None


class KernelResourceError(RuntimeError):
    pass


def _get_kernel_params(image, footprint_shape, value_range='auto',
                       partitions=128, hist_size_coarse=None):
    """Determine kernel launch parameters and #define values for its code.

    Parameters
    ----------
    image : cupy.ndarray
        The histogram bin range will depend on the image dtype unless specified
        explicitly via `value_range`
    footprint_shape : tuple of int
        The shape of the footprint. The dtype used for storing the histogram
        will depend on the footprint size. For small footprints, histograms
        will be stored using int8, otherwise int16 will be used.
    value_range : {'auto', 'dtype', 'image'}, optional
        When value_range='dtype', the range will be determined based on the
        maximal range of the data type. When ``value_range='image'``, the
        minimum and maximum intensities present in the image will be used. When
        set to auto 'auto', `dtype` is used for 8-bit images and otherwise
        'image' is used.
    partitions : positive int, optional
        The grid size used during kernel launch will be (partitions, 1, 1).
        Increasing this will increase parallelism (and thus performance), but
        at cost of additional GPU memory usage. Will be automatically truncated
        to a value no larger than image.shape[0] // 2.
    hist_size_coarse : int or None, optional
        Can be used to override the default choice of the number of coarse
        histogram bins. It is not generally recommended to set this as
        infeasible values can easily be chosen. Using None, will give
        automatically selected values that have been validated in previous
        testing.

    Returns
    -------
    CUDAParams : namedtuple
        Various parameters used in kernel code generation and at launch time.
        See comments next to the KernelParams declaration below for details.
    """

    if value_range == 'auto':
        if image.dtype.itemsize < 2:
            value_range = 'dtype'
        else:
            # to save memory, try using actual value range for >8-bit images
            # (e.g. DICOM images often have 12-bit range)
            value_range = 'image'

    if value_range == 'dtype':
        if image.dtype.itemsize > 2:
            raise ValueError(
                "dtype range only supported for 8 and 16-bit integer dtypes."
            )
        iinfo = cp.iinfo(image.dtype)
        minv, maxv = iinfo.min, iinfo.max
    elif value_range == 'image':
        minv = int(image.min())
        maxv = int(image.max())
    else:
        if len(value_range) != 2:
            raise ValueError(
                "value_range must be either 'dtype', 'image' or a "
                "(min, max) sequence."
            )
        minv, maxv = value_range

    hist_offset = -min(minv, 0)
    hist_size = maxv - minv + 1
    # round hist_size up to the nearest power of 2
    hist_size = round(2**math.ceil(math.log2(hist_size)))
    hist_size = max(hist_size, 32)

    if hist_size_coarse is None:
        if hist_size < 256:
            hist_size_coarse = 4   # tests pass for 2, 4 or 8 only.
        elif hist_size == 256:
            hist_size_coarse = 8   # tests pass for 2, 4, 8 or 16. fail for 32.
        elif hist_size == 512:
            hist_size_coarse = 8   # tests pass for 2, 4, 8 or 16. fail for 32.
        elif hist_size < 4096:
            hist_size_coarse = 32  # tests pass for 2, 4, 8, 16 or 32.
        elif hist_size <= 65536:
            hist_size_coarse = 64
        else:
            raise KernelResourceError(
                "More than 65536 bins is currently untested due to excessive "
                "shared memory requirements."
            )

    # have to set block[0] large enough that histogramMedianParFineLookupOnly
    # and histogramMedianParCoarseLookupOnly search sizes fit within the number
    # of threads in the block.
    # Use the maximum of the coarse and fine sizes, rounded up to the nearest
    # multiple of 32.
    hist_size_fine = hist_size // hist_size_coarse
    hist_size_max = max(hist_size_fine, hist_size_coarse)
    block0 = 32 * math.ceil(hist_size_max / 32)
    if block0 > 256:
        d = cp.cuda.Device()
        max_block_x = d.attributes["MaxBlockDimX"]
        if block0 > max_block_x:
            raise KernelResourceError(
                f"The requested block size of {block0} for the first dimension"
                f", exceeds MaxBlockDimX={max_block_x} for this device."
            )

    if partitions > image.shape[0]:
        partitions = image.shape[0] // 2  # can be chosen
    grid = (partitions, 1, 1)
    # block[0] must be at least the warp size
    block = (block0, 1, 1)

    # print(f"{hist_size=}, {hist_size_coarse=}, {block[0]=}, {hist_offset=}")

    hist_int_t, hist_dtype = _get_hist_dtype(footprint_shape)

    # All recent GPUs (CC>=3.5) allow at least 48k of shared memory per block,
    # so don't bother checking the requirements unless thousands of histogram
    # bins are requested.
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications  # noqa
    if hist_size >= 8192:
        smem_size = _check_shared_memory_requirement_bytes(
            hist_dtype, hist_size_coarse, hist_size_fine
        )
        d = cp.cuda.Device()
        smem_available = d.attributes['MaxSharedMemoryPerBlock']
        if smem_size > smem_available:
            raise KernelResourceError(
                f"Shared memory requirement of {smem_size} bytes per block"
                f"exceeds the device limit of {smem_available}."
            )
    CUDAParams = namedtuple(
        'HistogramMedianKernelParams',
        [
            'grid',
            'block',
            'hist_size',           # total number of histogram bins
            'hist_size_coarse',    # number of coarse-level histogram bins
            'hist_dtype',          # cupy.dtype of the histogram
            'hist_int_t',          # C++ type of the histogram
            'hist_offset',         # offset from 0 for the first bin
            'partitions'           # number of parallel bands to use
        ]
    )
    return CUDAParams(
        grid,
        block,
        hist_size,
        hist_size_coarse,
        hist_dtype,
        hist_int_t,
        hist_offset,
        partitions,
    )


def _median_hist(image, footprint, output=None, mode='mirror', cval=0,
                 value_range='auto'):

    if output is not None:
        raise NotImplementedError(
            "Use of a user-defined output array has not been implemented"
        )

    compatible_image, reason = _can_use_histogram(image, footprint)
    if not compatible_image:
        raise ValueError(reason)

    # kernel pointer offset calculations assume C-contiguous image data
    image = cp.ascontiguousarray(image)
    n_rows, n_cols = image.shape[:2]
    if image.dtype.kind == 'b':
        image = image.view(cp.uint8)
    if image.dtype.kind not in 'iu':
        raise ValueError("only integer-type images are accepted")

    radii = tuple(s // 2 for s in footprint.shape)
    # med_pos is the index corresponding to the median
    # (calculation here assumes all elements of the footprint are True)
    med_pos = footprint.size // 2

    params = _get_kernel_params(image, footprint.shape, value_range)

    # pad as necessary to avoid boundary artifacts
    # Don't have to pad along axis 0 if mode is already 'nearest' because the
    # kernel already assumes 'nearest' mode internally.
    autopad = True
    pad_both_axes = mode != 'nearest'
    if autopad:
        if pad_both_axes:
            npad = tuple((r, r) for r in radii)
        else:
            npad = ((0, 0),) * (image.ndim - 1) + ((radii[-1], radii[-1]),)
        mode = _to_np_mode(mode)
        if mode == 'constant':
            pad_kwargs = dict(mode=mode, constant_values=cval)
        else:
            pad_kwargs = dict(mode=mode)
        image = cp.pad(image, npad, **pad_kwargs)
        # must update n_rows, n_cols after padding!
        n_rows, n_cols = image.shape[:2]

    # generate the kernel
    kern = _get_median_rawkernel(
        image_t=_dtype_to_CUDA_int_type(image.dtype),
        hist_offset=params.hist_offset,
        hist_int_t=params.hist_int_t,
        hist_size=params.hist_size,
        hist_size_coarse=params.hist_size_coarse,
    )

    # allocate output and scratch space, `hist` and `coarse_hist`.
    out = cp.empty_like(image)
    hist = cp.zeros(
        (n_cols * params.hist_size * params.partitions,),
        params.hist_dtype,
    )
    coarse_hist = cp.zeros(
        (n_cols * params.hist_size_coarse * params.partitions,),
        params.hist_dtype,
    )

    # call the kernel
    r0, r1 = radii[:2]
    s0, s1 = image.shape[:2]
    kernel_args = (image, out, hist, coarse_hist, r0, r1, med_pos, s0, s1)
    kern(params.grid, params.block, kernel_args)

    # remove any padding that was added
    if autopad:
        if pad_both_axes:
            out_sl = tuple(slice(r, -r) for r in radii)
            out = out[out_sl]
        else:
            out = out[..., radii[-1]:-radii[-1]]
    return out
