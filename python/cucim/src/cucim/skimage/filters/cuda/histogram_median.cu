/* Several functions and the primary kernel used for the histogram-based
 * median are in this file.
 *
 * Note that this file cannot be compiled standalone as various definitions
 * and a couple of the supporting functions get dynamically generated based
 * on the actual histogram sizes. See the Python function
 * `_get_median_rawkernel` defined in `_median_hist.py`. This function will
 * generate the full kernel code given a set of parameters.
 */

__device__ void histogramAddAndSubCoarse(HIST_INT_T* H,
                                         const HIST_INT_T* hist_colAdd,
                                         const HIST_INT_T* hist_colSub) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) {
    H[tx] += hist_colAdd[tx] - hist_colSub[tx];
  }
}

__device__ void histogramMultipleAddCoarse(HIST_INT_T* H,
                                           const HIST_INT_T* hist_col,
                                           int histCount) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) {
    HIST_INT_T temp = H[tx];
    for (int i = 0; i < histCount; i++)
      temp += hist_col[(i << LOG2_COARSE) + tx];
    H[tx] = temp;
  }
}

__device__ void histogramClearCoarse(HIST_INT_T* H) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) {
    H[tx] = 0;
  }
}

__device__ void histogramAddCoarse(HIST_INT_T* H, const HIST_INT_T* hist_col) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) {
    H[tx] += hist_col[tx];
  }
}

__device__ void histogramSubCoarse(HIST_INT_T* H, const HIST_INT_T* hist_col) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) {
    H[tx] -= hist_col[tx];
  }
}

__device__ void histogramAddFine(HIST_INT_T* H, const HIST_INT_T* hist_col) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_FINE) {
    H[tx] += hist_col[tx];
  }
}

__device__ void histogramAddAndSubFine(HIST_INT_T* H,
                                       const HIST_INT_T* hist_colAdd,
                                       const HIST_INT_T* hist_colSub) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_FINE) {
    H[tx] += hist_colAdd[tx] - hist_colSub[tx];
  }
}

__device__ void histogramClearFine(HIST_INT_T* H) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_FINE) {
    H[tx] = 0;
  }
}

__device__ void lucClearCoarse(int* luc) {
  int tx = threadIdx.x;
  if (tx < HIST_SIZE_COARSE) luc[tx] = 0;
}

extern "C" __global__ void cuRankFilterMultiBlock(IMAGE_T* src, IMAGE_T* dest,
                                                  HIST_INT_T* histPar,
                                                  HIST_INT_T* coarseHistGrid,
                                                  int r0, int r1, int medPos_,
                                                  int rows, int cols) {
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
  int rowsPerBlock = rows / gridDim.x + doExtraRow;

  // The following code partitions the work to the blocks. Some blocks will do
  // one row more than other blocks. This code is responsible for doing that
  // balancing
  if (doExtraRow) {
    startRow = rowsPerBlock * blockIdx.x;
    stopRow = min(rows, startRow + rowsPerBlock);
  } else {
    startRow = (rowsPerBlock + 1) * extraRowThread +
               (rowsPerBlock) * (blockIdx.x - extraRowThread);
    stopRow = min(rows, startRow + rowsPerBlock);
  }

  HIST_INT_T* hist = histPar + cols * HIST_SIZE * blockIdx.x;
  HIST_INT_T* histCoarse =
      coarseHistGrid + cols * HIST_SIZE_COARSE * blockIdx.x;

  if (blockIdx.x == (gridDim.x - 1)) stopRow = rows;
  __syncthreads();
  int initNeeded = 0, initStartRow, initStopRow;
  HIST_INT_T initVal;

  if (blockIdx.x == 0) {
    // Note: skips one iteration in the initialization loop by starting at
    // row 1 instead of 0 and using initVal r0+2 instead of r0+1.
    initNeeded = 1;
    initVal = r0 + 2;
    initStartRow = 1;
    initStopRow = r0;
  } else if (startRow < (r0 + 2)) {
    initNeeded = 1;
    initVal = r0 + 2 - startRow;
    initStartRow = 1;
    initStopRow = r0 + startRow;
  } else {
    initNeeded = 0;
    initVal = 0;
    initStartRow = startRow - (r0 + 1);
    initStopRow = r0 + startRow;
  }
  __syncthreads();

  // In the original algorithm an initialization phase was required as part of
  // the window was outside the image. In this parallel version, the
  // initializtion is required for all thread blocks that part of the median
  // filter is outside the window. For all threads in the block the same code
  // will be executed.
  if (initNeeded) {
    for (int j = threadIdx.x; j < (cols); j += blockDim.x) {
      hist[j * HIST_SIZE + src[j] + HIST_OFFSET] = initVal;
      histCoarse[j * HIST_SIZE_COARSE + ((src[j] + HIST_OFFSET) >> LOG2_FINE)] =
          initVal;
    }
  }
  __syncthreads();

  // For all remaining rows in the median filter, add the values to the the
  // histogram
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    for (int i = initStartRow; i < initStopRow; i++) {
      int pos = min(i, rows - 1);
      hist[j * HIST_SIZE + src[pos * row_stride + j] + HIST_OFFSET]++;
      histCoarse[j * HIST_SIZE_COARSE +
                 ((src[pos * row_stride + j] + HIST_OFFSET) >> LOG2_FINE)]++;
    }
  }
  __syncthreads();
  // Going through all the rows that the block is responsible for.
  int inc = blockDim.x * HIST_SIZE;
  int incCoarse = blockDim.x * HIST_SIZE_COARSE;
  for (int i = startRow; i < stopRow; i++) {
    // For every new row that is started the global histogram for the entire
    // window is restarted.

    histogramClearCoarse(HCoarse);
    lucClearCoarse(luc);
    // Computing some necessary indices
    int possub = max(0, i - r0 - 1), posadd = min(rows - 1, i + r0);
    int histPos = threadIdx.x * HIST_SIZE;
    int histCoarsePos = threadIdx.x * HIST_SIZE_COARSE;
    // Going through all the elements of a specific row. For each histogram, a
    // value is taken out and one value is added.
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      hist[histPos + src[possub * row_stride + j] + HIST_OFFSET]--;
      hist[histPos + src[posadd * row_stride + j] + HIST_OFFSET]++;
      histCoarse[histCoarsePos +
                 ((src[possub * row_stride + j] + HIST_OFFSET) >> LOG2_FINE)]--;
      histCoarse[histCoarsePos +
                 ((src[posadd * row_stride + j] + HIST_OFFSET) >> LOG2_FINE)]++;

      histPos += inc;
      histCoarsePos += incCoarse;
    }
    __syncthreads();

    histogramMultipleAddCoarse(HCoarse, histCoarse, 2 * r1 + 1);
    int cols_m_1 = cols - 1;

    for (int j = r1; j < cols - r1; j++) {
      int possub = max(j - r1, 0);
      int posadd = min(j + 1 + r1, cols_m_1);
      int medPos = medPos_;
      __syncthreads();

      histogramMedianParCoarseLookupOnly(HCoarse, HCoarseScan, medPos,
                                         &firstBin, &countAtMed);
      __syncthreads();

      int loopIndex = luc[firstBin];
      if (loopIndex <= (j - r1)) {
        histogramClearFine(HFine[firstBin]);
        for (loopIndex = j - r1; loopIndex < min(j + r1 + 1, cols);
             loopIndex++) {
          histogramAddFine(HFine[firstBin], hist + (loopIndex * HIST_SIZE +
                                                    (firstBin << LOG2_FINE)));
        }
      } else {
        for (; loopIndex < (j + r1 + 1); loopIndex++) {
          histogramAddAndSubFine(
              HFine[firstBin],
              hist + (min(loopIndex, cols_m_1) * HIST_SIZE +
                      (firstBin << LOG2_FINE)),
              hist + (max(loopIndex - 2 * r1 - 1, 0) * HIST_SIZE +
                      (firstBin << LOG2_FINE)));
          __syncthreads();
        }
      }
      __syncthreads();
      luc[firstBin] = loopIndex;

      int leftOver = medPos - countAtMed;
      if (leftOver >= 0) {
        histogramMedianParFineLookupOnly(HFine[firstBin], HCoarseScan, leftOver,
                                         &retval, &countAtMed);
      } else
        retval = 0;
      __syncthreads();

      if (threadIdx.x == 0) {
        dest[i * row_stride + j] =
            (firstBin << LOG2_FINE) + retval - HIST_OFFSET;
      }
      histogramAddAndSubCoarse(HCoarse,
                               histCoarse + (int)(posadd << LOG2_COARSE),
                               histCoarse + (int)(possub << LOG2_COARSE));

      __syncthreads();
    }
    __syncthreads();
  }
}
