/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define OP_PERCENTILE 0
#define OP_THRESHOLD 1
#define OP_MEAN 2
#define OP_SUM 3
#define OP_POP 4
#define OP_GRADIENT 5
#define OP_AUTOLEVEL 6
#define OP_ENTROPY 7
#define OP_ENHANCE_CONTRAST 8
#define OP_SUBTRACT_MEAN 9
#define OP_EQUALIZE 10
#define OP_BILATERAL_MEAN 11
#define OP_BILATERAL_POP 12
#define OP_BILATERAL_SUM 13
#define OP_MODAL 14
#define OP_GEOMETRIC_MEAN 15

#ifndef HIST_COUNTER_T
#    define HIST_COUNTER_T int
#endif

#ifndef RANK_HIST_OUTPUT_T
#    define RANK_HIST_OUTPUT_T unsigned char
#endif

#if RANK_HIST_OP == OP_GEOMETRIC_MEAN
__device__ __constant__ double geometricMeanLogLut[256] = { 0,
                                                            0.69314718055994529,
                                                            1.0986122886681098,
                                                            1.3862943611198906,
                                                            1.6094379124341003,
                                                            1.791759469228055,
                                                            1.9459101490553132,
                                                            2.0794415416798357,
                                                            2.1972245773362196,
                                                            2.3025850929940459,
                                                            2.3978952727983707,
                                                            2.4849066497880004,
                                                            2.5649493574615367,
                                                            2.6390573296152584,
                                                            2.7080502011022101,
                                                            2.7725887222397811,
                                                            2.8332133440562162,
                                                            2.8903717578961645,
                                                            2.9444389791664403,
                                                            2.9957322735539909,
                                                            3.044522437723423,
                                                            3.0910424533583161,
                                                            3.1354942159291497,
                                                            3.1780538303479458,
                                                            3.2188758248682006,
                                                            3.2580965380214821,
                                                            3.2958368660043291,
                                                            3.3322045101752038,
                                                            3.3672958299864741,
                                                            3.4011973816621555,
                                                            3.4339872044851463,
                                                            3.4657359027997265,
                                                            3.4965075614664802,
                                                            3.5263605246161616,
                                                            3.5553480614894135,
                                                            3.5835189384561099,
                                                            3.6109179126442243,
                                                            3.6375861597263857,
                                                            3.6635616461296463,
                                                            3.6888794541139363,
                                                            3.713572066704308,
                                                            3.7376696182833684,
                                                            3.7612001156935624,
                                                            3.784189633918261,
                                                            3.8066624897703196,
                                                            3.8286413964890951,
                                                            3.8501476017100584,
                                                            3.8712010109078911,
                                                            3.8918202981106265,
                                                            3.912023005428146,
                                                            3.9318256327243257,
                                                            3.9512437185814275,
                                                            3.970291913552122,
                                                            3.9889840465642745,
                                                            4.0073331852324712,
                                                            4.0253516907351496,
                                                            4.0430512678345503,
                                                            4.0604430105464191,
                                                            4.0775374439057197,
                                                            4.0943445622221004,
                                                            4.1108738641733114,
                                                            4.1271343850450917,
                                                            4.1431347263915326,
                                                            4.1588830833596715,
                                                            4.1743872698956368,
                                                            4.1896547420264252,
                                                            4.2046926193909657,
                                                            4.219507705176107,
                                                            4.2341065045972597,
                                                            4.2484952420493594,
                                                            4.2626798770413155,
                                                            4.2766661190160553,
                                                            4.290459441148391,
                                                            4.3040650932041702,
                                                            4.3174881135363101,
                                                            4.3307333402863311,
                                                            4.3438054218536841,
                                                            4.3567088266895917,
                                                            4.3694478524670215,
                                                            4.3820266346738812,
                                                            4.3944491546724391,
                                                            4.4067192472642533,
                                                            4.4188406077965983,
                                                            4.4308167988433134,
                                                            4.4426512564903167,
                                                            4.4543472962535073,
                                                            4.4659081186545837,
                                                            4.4773368144782069,
                                                            4.4886363697321396,
                                                            4.499809670330265,
                                                            4.5108595065168497,
                                                            4.5217885770490405,
                                                            4.5325994931532563,
                                                            4.5432947822700038,
                                                            4.5538768916005408,
                                                            4.5643481914678361,
                                                            4.5747109785033828,
                                                            4.5849674786705723,
                                                            4.5951198501345898,
                                                            4.6051701859880918,
                                                            4.6151205168412597,
                                                            4.6249728132842707,
                                                            4.6347289882296359,
                                                            4.6443908991413725,
                                                            4.6539603501575231,
                                                            4.6634390941120669,
                                                            4.6728288344619058,
                                                            4.6821312271242199,
                                                            4.6913478822291435,
                                                            4.7004803657924166,
                                                            4.7095302013123339,
                                                            4.7184988712950942,
                                                            4.7273878187123408,
                                                            4.7361984483944957,
                                                            4.7449321283632502,
                                                            4.7535901911063645,
                                                            4.7621739347977563,
                                                            4.7706846244656651,
                                                            4.7791234931115296,
                                                            4.7874917427820458,
                                                            4.7957905455967413,
                                                            4.8040210447332568,
                                                            4.8121843553724171,
                                                            4.8202815656050371,
                                                            4.8283137373023015,
                                                            4.836281906951478,
                                                            4.8441870864585912,
                                                            4.8520302639196169,
                                                            4.8598124043616719,
                                                            4.8675344504555822,
                                                            4.8751973232011512,
                                                            4.8828019225863706,
                                                            4.8903491282217537,
                                                            4.8978397999509111,
                                                            4.9052747784384296,
                                                            4.9126548857360524,
                                                            4.9199809258281251,
                                                            4.9272536851572051,
                                                            4.9344739331306915,
                                                            4.9416424226093039,
                                                            4.9487598903781684,
                                                            4.9558270576012609,
                                                            4.962844630259907,
                                                            4.9698132995760007,
                                                            4.9767337424205742,
                                                            4.9836066217083363,
                                                            4.990432586778736,
                                                            4.9972122737641147,
                                                            5.0039463059454592,
                                                            5.0106352940962555,
                                                            5.0172798368149243,
                                                            5.0238805208462765,
                                                            5.0304379213924353,
                                                            5.0369526024136295,
                                                            5.0434251169192468,
                                                            5.0498560072495371,
                                                            5.0562458053483077,
                                                            5.0625950330269669,
                                                            5.0689042022202315,
                                                            5.0751738152338266,
                                                            5.0814043649844631,
                                                            5.0875963352323836,
                                                            5.0937502008067623,
                                                            5.0998664278241987,
                                                            5.1059454739005803,
                                                            5.1119877883565437,
                                                            5.1179938124167554,
                                                            5.1239639794032588,
                                                            5.1298987149230735,
                                                            5.1357984370502621,
                                                            5.1416635565026603,
                                                            5.1474944768134527,
                                                            5.1532915944977793,
                                                            5.1590552992145291,
                                                            5.1647859739235145,
                                                            5.1704839950381514,
                                                            5.1761497325738288,
                                                            5.181783550292085,
                                                            5.1873858058407549,
                                                            5.1929568508902104,
                                                            5.1984970312658261,
                                                            5.2040066870767951,
                                                            5.2094861528414214,
                                                            5.2149357576089859,
                                                            5.2203558250783244,
                                                            5.2257466737132017,
                                                            5.2311086168545868,
                                                            5.2364419628299492,
                                                            5.2417470150596426,
                                                            5.2470240721604862,
                                                            5.2522734280466299,
                                                            5.2574953720277815,
                                                            5.2626901889048856,
                                                            5.2678581590633282,
                                                            5.2729995585637468,
                                                            5.2781146592305168,
                                                            5.2832037287379885,
                                                            5.2882670306945352,
                                                            5.2933048247244923,
                                                            5.2983173665480363,
                                                            5.3033049080590757,
                                                            5.3082676974012051,
                                                            5.3132059790417872,
                                                            5.3181199938442161,
                                                            5.3230099791384085,
                                                            5.3278761687895813,
                                                            5.3327187932653688,
                                                            5.3375380797013179,
                                                            5.3423342519648109,
                                                            5.3471075307174685,
                                                            5.3518581334760666,
                                                            5.3565862746720123,
                                                            5.3612921657094255,
                                                            5.3659760150218512,
                                                            5.3706380281276624,
                                                            5.3752784076841653,
                                                            5.3798973535404597,
                                                            5.3844950627890888,
                                                            5.389071729816501,
                                                            5.393627546352362,
                                                            5.3981627015177525,
                                                            5.4026773818722793,
                                                            5.4071717714601188,
                                                            5.4116460518550396,
                                                            5.4161004022044201,
                                                            5.4205349992722862,
                                                            5.4249500174814029,
                                                            5.4293456289544411,
                                                            5.43372200355424,
                                                            5.4380793089231956,
                                                            5.4424177105217932,
                                                            5.4467373716663099,
                                                            5.4510384535657002,
                                                            5.4553211153577017,
                                                            5.4595855141441589,
                                                            5.4638318050256105,
                                                            5.4680601411351315,
                                                            5.472270673671475,
                                                            5.476463551931511,
                                                            5.4806389233419912,
                                                            5.4847969334906548,
                                                            5.4889377261566867,
                                                            5.4930614433405482,
                                                            5.4971682252932021,
                                                            5.5012582105447274,
                                                            5.5053315359323625,
                                                            5.5093883366279774,
                                                            5.5134287461649825,
                                                            5.5174528964647074,
                                                            5.521460917862246,
                                                            5.5254529391317835,
                                                            5.5294290875114234,
                                                            5.5333894887275203,
                                                            5.5373342670185366,
                                                            5.5412635451584258,
                                                            5.5451774444795623 };
#endif

__device__ void histogramPrefixScan256(int* hist, int* scan)
{
    int tx = threadIdx.x;
    if (tx < 256)
    {
        scan[tx] = hist[tx];
    }
    __syncthreads();

    for (int offset = 1; offset < 256; offset <<= 1)
    {
        int v = 0;
        if (tx >= offset && tx < 256)
        {
            v = scan[tx - offset];
        }
        __syncthreads();
        if (tx >= offset && tx < 256)
        {
            scan[tx] += v;
        }
        __syncthreads();
    }
}

__device__ void reduceSum256(int* values)
{
    int tx = threadIdx.x;
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tx < stride)
        {
            values[tx] += values[tx + stride];
        }
        __syncthreads();
    }
}

__device__ void histogramWeightedPrefixScan256(int* hist, int* scan)
{
    int tx = threadIdx.x;
    if (tx < 256)
    {
        scan[tx] = hist[tx] * tx;
    }
    __syncthreads();

    for (int offset = 1; offset < 256; offset <<= 1)
    {
        int v = 0;
        if (tx >= offset && tx < 256)
        {
            v = scan[tx - offset];
        }
        __syncthreads();
        if (tx >= offset && tx < 256)
        {
            scan[tx] += v;
        }
        __syncthreads();
    }
}

__device__ RANK_HIST_OUTPUT_T histogramRankValue(int* hist,
                                                 int* scan,
                                                 int* tmp0,
                                                 int* tmp1,
                                                 double* dtmp,
                                                 int op,
                                                 int window_size,
                                                 double p0,
                                                 double p1,
                                                 double s0,
                                                 double s1,
                                                 double dtype_max,
                                                 unsigned char center)
{
    int tx = threadIdx.x;
    __shared__ int result;
    __shared__ int range_start;
    __shared__ int range_end;
    // clang-format off
#if RANK_HIST_OP == OP_MEAN || RANK_HIST_OP == OP_SUM || RANK_HIST_OP == OP_SUBTRACT_MEAN || RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
    // clang-format on
    __shared__ int range_start_sum;
    __shared__ int range_end_sum;
#endif

#if RANK_HIST_OP == OP_ENTROPY
    double ent = 0.0;
    if (tx < 256 && hist[tx] > 0)
    {
        double p = ((double)hist[tx]) / window_size;
        ent = -p * log(p) / 0.6931471805599453;
    }
    dtmp[tx] = ent;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tx < stride)
        {
            dtmp[tx] += dtmp[tx + stride];
        }
        __syncthreads();
    }
    return static_cast<RANK_HIST_OUTPUT_T>(dtmp[0]);
#elif RANK_HIST_OP == OP_MODAL
    tmp0[tx] = hist[tx];
    tmp1[tx] = tx;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tx < stride)
        {
            int other_count = tmp0[tx + stride];
            int other_value = tmp1[tx + stride];
            if (other_count > tmp0[tx] || (other_count == tmp0[tx] && other_value < tmp1[tx]))
            {
                tmp0[tx] = other_count;
                tmp1[tx] = other_value;
            }
        }
        __syncthreads();
    }
    return static_cast<RANK_HIST_OUTPUT_T>(tmp1[0]);
#elif RANK_HIST_OP == OP_GEOMETRIC_MEAN
    double log_sum = 0.0;
    if (tx < 256 && hist[tx] > 0)
    {
        log_sum = ((double)hist[tx]) * geometricMeanLogLut[tx];
    }
    dtmp[tx] = log_sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tx < stride)
        {
            dtmp[tx] += dtmp[tx + stride];
        }
        __syncthreads();
    }
    return static_cast<RANK_HIST_OUTPUT_T>(round(exp(dtmp[0] / window_size) - 1.0));
#else
    op = RANK_HIST_OP;
    histogramPrefixScan256(hist, scan);
    int pop = scan[255];

    if (tx == 0)
    {
        result = 0;
        int start = max(0, (int)ceil(p0 * pop / 100.0) - 1);
        int end = (int)(p1 * pop / 100.0);
        if (end <= start)
        {
            end = start + 1;
        }
        if (end > pop)
        {
            end = pop;
        }
        range_start = start;
        range_end = end;
    }
    __syncthreads();

    if (op == OP_PERCENTILE || op == OP_THRESHOLD)
    {
        int target;
        if (p0 == 100.0)
        {
            target = pop - 1;
        }
        else
        {
            target = (int)(p0 * pop / 100.0);
            if (target >= pop)
            {
                target = pop - 1;
            }
        }

        if (tx < 256 && hist[tx] > 0)
        {
            int bin_start = scan[tx] - hist[tx];
            if (bin_start <= target && scan[tx] > target)
            {
                result = tx;
            }
        }
        __syncthreads();

        if (op == OP_THRESHOLD)
        {
            return (center >= result) ? static_cast<RANK_HIST_OUTPUT_T>(dtype_max) : static_cast<RANK_HIST_OUTPUT_T>(0);
        }
        return static_cast<RANK_HIST_OUTPUT_T>(result);
    }

#    if RANK_HIST_OP == OP_EQUALIZE
    return static_cast<RANK_HIST_OUTPUT_T>(dtype_max * ((double)scan[center]) / pop);
#    endif

    // clang-format off
#if RANK_HIST_OP == OP_MEAN || RANK_HIST_OP == OP_SUM || RANK_HIST_OP == OP_SUBTRACT_MEAN || RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
    // clang-format on
    histogramWeightedPrefixScan256(hist, tmp1);
    if (tx == 0)
    {
        range_start_sum = 0;
        range_end_sum = 0;
    }
    __syncthreads();

#        if RANK_HIST_OP == OP_BILATERAL_MEAN || RANK_HIST_OP == OP_BILATERAL_POP || RANK_HIST_OP == OP_BILATERAL_SUM
    if (tx == 0)
    {
        int start_bin = max(0, (int)floor((double)center - s1) + 1);
        int stop_bin = min(256, (int)ceil((double)center + s0));
        if (stop_bin <= start_bin)
        {
            range_start = 0;
            range_end = 0;
            range_start_sum = 0;
            range_end_sum = 0;
        }
        else
        {
            range_start = (start_bin > 0) ? scan[start_bin - 1] : 0;
            range_end = scan[stop_bin - 1];
            range_start_sum = (start_bin > 0) ? tmp1[start_bin - 1] : 0;
            range_end_sum = tmp1[stop_bin - 1];
        }
    }
    __syncthreads();
#        else
    if (tx < 256 && hist[tx] > 0)
    {
        int bin_end = scan[tx];
        int bin_start = bin_end - hist[tx];
        int weighted_end = tmp1[tx];
        int weighted_start = weighted_end - hist[tx] * tx;

        if (range_start > 0 && bin_start < range_start && bin_end >= range_start)
        {
            range_start_sum = weighted_start + (range_start - bin_start) * tx;
        }
        if (range_end > 0 && bin_start < range_end && bin_end >= range_end)
        {
            range_end_sum = weighted_start + (range_end - bin_start) * tx;
        }
    }
    __syncthreads();
#        endif

    int selected_count_total = range_end - range_start;
    int selected_sum_total = range_end_sum - range_start_sum;
    if (op == OP_BILATERAL_POP)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(selected_count_total);
    }
    if (selected_count_total <= 0)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(0);
    }
    if (op == OP_BILATERAL_MEAN)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(((double)selected_sum_total) / selected_count_total);
    }
    if (op == OP_BILATERAL_SUM)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(selected_sum_total);
    }
    if (op == OP_MEAN)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(((double)selected_sum_total) / selected_count_total);
    }
    if (op == OP_SUBTRACT_MEAN)
    {
        double mean = ((double)selected_sum_total) / selected_count_total;
        return static_cast<RANK_HIST_OUTPUT_T>(((double)center - mean) * 0.5 + floor((dtype_max + 1.0) / 2.0));
    }
    return static_cast<RANK_HIST_OUTPUT_T>(selected_sum_total);
#    endif

    int selected_count = 0;
    int selected_sum = 0;
    if (tx < 256)
    {
        int bin_start = scan[tx] - hist[tx];
        int bin_end = scan[tx];
        selected_count = min(bin_end, range_end) - max(bin_start, range_start);
        if (selected_count < 0)
        {
            selected_count = 0;
        }
        selected_sum = selected_count * tx;
    }

    if (op == OP_POP)
    {
        double low = p0 * pop / 100.0;
        double high = p1 * pop / 100.0;
        int count = 0;
        if (tx < 256 && hist[tx] > 0 && (double)scan[tx] >= low && (double)scan[tx] <= high)
        {
            count = hist[tx];
        }
        tmp0[tx] = count;
        __syncthreads();
        reduceSum256(tmp0);
        return static_cast<RANK_HIST_OUTPUT_T>(tmp0[0]);
    }

    if (op == OP_GRADIENT || op == OP_AUTOLEVEL || op == OP_ENHANCE_CONTRAST)
    {
        tmp0[tx] = selected_count > 0 ? tx : 255;
        tmp1[tx] = selected_count > 0 ? tx : 0;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1)
        {
            if (tx < stride)
            {
                tmp0[tx] = min(tmp0[tx], tmp0[tx + stride]);
                tmp1[tx] = max(tmp1[tx], tmp1[tx + stride]);
            }
            __syncthreads();
        }
        if (op == OP_GRADIENT)
        {
            return static_cast<RANK_HIST_OUTPUT_T>(tmp1[0] - tmp0[0]);
        }

        int min_val = tmp0[0];
        int max_val = tmp1[0];
        if (op == OP_ENHANCE_CONTRAST)
        {
            return (max_val - center < center - min_val) ? static_cast<RANK_HIST_OUTPUT_T>(max_val) :
                                                           static_cast<RANK_HIST_OUTPUT_T>(min_val);
        }

        int clamped = min(max((int)center, min_val), max_val);
        int delta = max_val - min_val;
        if (delta > 0)
        {
            return static_cast<RANK_HIST_OUTPUT_T>(((double)(clamped - min_val) / delta) * dtype_max);
        }
        return static_cast<RANK_HIST_OUTPUT_T>(0);
    }

    // clang-format off
#if RANK_HIST_OP != OP_MEAN && RANK_HIST_OP != OP_SUM && RANK_HIST_OP != OP_SUBTRACT_MEAN && RANK_HIST_OP != OP_BILATERAL_MEAN && RANK_HIST_OP != OP_BILATERAL_POP && RANK_HIST_OP != OP_BILATERAL_SUM
    // clang-format on
    tmp0[tx] = selected_count;
    tmp1[tx] = selected_sum;
    __syncthreads();
    reduceSum256(tmp0);
    reduceSum256(tmp1);

    if (op == OP_MEAN)
    {
        return static_cast<RANK_HIST_OUTPUT_T>(((double)tmp1[0]) / tmp0[0]);
    }
    if (op == OP_SUBTRACT_MEAN)
    {
        double mean = ((double)tmp1[0]) / tmp0[0];
        return static_cast<RANK_HIST_OUTPUT_T>(((double)center - mean) * 0.5 + floor((dtype_max + 1.0) / 2.0));
    }
    return static_cast<RANK_HIST_OUTPUT_T>(tmp1[0]);
#    endif
#endif
}

extern "C" __global__ void cuRankHistogram2DUint8(const unsigned char* src,
                                                  RANK_HIST_OUTPUT_T* dest,
                                                  HIST_COUNTER_T* histPar,
                                                  int r0,
                                                  int r1,
                                                  double p0,
                                                  double p1,
                                                  double s0,
                                                  double s1,
                                                  double dtype_max,
                                                  int op,
                                                  int window_size,
                                                  int rows,
                                                  int cols)
{
    __shared__ int H[256];
    __shared__ int Hscan[256];
    __shared__ int tmp0[256];
    __shared__ int tmp1[256];
    __shared__ double dtmp[256];

    int tx = threadIdx.x;
    int out_rows = rows - 2 * r0;
    int rows_per_block = (out_rows + gridDim.x - 1) / gridDim.x;
    int start_out = blockIdx.x * rows_per_block;
    int stop_out = min(out_rows, start_out + rows_per_block);

    if (start_out >= stop_out)
    {
        return;
    }

    int start_row = r0 + start_out;
    int stop_row = r0 + stop_out;
    HIST_COUNTER_T* hist = histPar + blockIdx.x * cols * 256;

    for (int col = tx; col < cols; col += blockDim.x)
    {
        HIST_COUNTER_T* col_hist = hist + col * 256;
        for (int row = start_row - r0; row <= start_row + r0; row++)
        {
            col_hist[src[row * cols + col]]++;
        }
    }
    __syncthreads();

    for (int row = start_row; row < stop_row; row++)
    {
        if (tx < 256)
        {
            int total = 0;
            for (int col = 0; col <= 2 * r1; col++)
            {
                total += (int)hist[col * 256 + tx];
            }
            H[tx] = total;
        }
        __syncthreads();

        for (int col = r1; col < cols - r1; col++)
        {
            unsigned char center = src[row * cols + col];
            RANK_HIST_OUTPUT_T value =
                histogramRankValue(H, Hscan, tmp0, tmp1, dtmp, op, window_size, p0, p1, s0, s1, dtype_max, center);

            if (tx == 0)
            {
                dest[row * cols + col] = value;
            }
            __syncthreads();

            if (col < cols - r1 - 1 && tx < 256)
            {
                int sub_col = col - r1;
                int add_col = col + r1 + 1;
                H[tx] += (int)hist[add_col * 256 + tx] - (int)hist[sub_col * 256 + tx];
            }
            __syncthreads();
        }

        if (row < stop_row - 1)
        {
            int sub_row = row - r0;
            int add_row = row + r0 + 1;
            for (int col = tx; col < cols; col += blockDim.x)
            {
                HIST_COUNTER_T* col_hist = hist + col * 256;
                col_hist[src[sub_row * cols + col]]--;
                col_hist[src[add_row * cols + col]]++;
            }
            __syncthreads();
        }
    }
}
