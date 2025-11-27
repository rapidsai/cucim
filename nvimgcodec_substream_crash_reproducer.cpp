/**
 * Minimal reproducer for nvImageCodec sub-code stream destruction crash
 * 
 * Issue: Calling nvimgcodecCodeStreamDestroy() on sub-code streams obtained via
 * nvimgcodecCodeStreamGetSubCodeStream() causes "free(): invalid pointer" crash.
 * 
 * Environment:
 *   - nvImageCodec v0.7.0 (internal build 11)
 *   - Linux x86_64
 *   - CUDA 12.x
 * 
 * Build:
 *   g++ -std=c++17 -o nvimgcodec_substream_crash_reproducer \
 *       nvimgcodec_substream_crash_reproducer.cpp \
 *       -I$CONDA_PREFIX/include \
 *       -L$CONDA_PREFIX/lib -lnvimgcodec \
 *       -Wl,-rpath,$CONDA_PREFIX/lib
 * 
 * Run:
 *   ./nvimgcodec_substream_crash_reproducer /path/to/test.tiff
 * 
 * Expected: Clean exit
 * Actual: "free(): invalid pointer" crash
 */

#include <nvimgcodec.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_NVIMGCODEC(call) do { \
    nvimgcodecStatus_t status = (call); \
    if (status != NVIMGCODEC_STATUS_SUCCESS) { \
        fprintf(stderr, "nvImageCodec error at %s:%d - status %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tiff_file>\n", argv[0]);
        fprintf(stderr, "\nThis reproducer demonstrates 'free(): invalid pointer' crash\n");
        fprintf(stderr, "when destroying sub-code streams in nvImageCodec v0.7.0\n");
        return 1;
    }
    
    const char* file_path = argv[1];
    printf("=== nvImageCodec Sub-CodeStream Destruction Crash Reproducer ===\n\n");
    printf("File: %s\n\n", file_path);
    
    // Step 1: Create nvImageCodec instance
    printf("[1] Creating nvImageCodec instance...\n");
    nvimgcodecInstance_t instance = nullptr;
    nvimgcodecInstanceCreateInfo_t create_info{};
    create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
    create_info.struct_next = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecInstanceCreate(&instance, &create_info));
    printf("    ✓ Instance created\n");
    
    // Step 2: Create decoder
    printf("[2] Creating decoder...\n");
    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecExecutionParams_t exec_params{};
    exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
    exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
    exec_params.struct_next = nullptr;
    exec_params.device_id = 0;
    exec_params.max_num_cpu_threads = 4;
    CHECK_NVIMGCODEC(nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    printf("    ✓ Decoder created\n");
    
    // Step 3: Create main code stream from file
    printf("[3] Creating main code stream from file...\n");
    nvimgcodecCodeStream_t main_stream = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromFile(instance, &main_stream, file_path, nullptr));
    printf("    ✓ Main code stream created\n");
    
    // Step 4: Get code stream info
    printf("[4] Getting code stream info...\n");
    nvimgcodecCodeStreamInfo_t stream_info{};
    stream_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
    stream_info.struct_size = sizeof(nvimgcodecCodeStreamInfo_t);
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetCodeStreamInfo(main_stream, &stream_info));
    printf("    ✓ Code stream info: %d images, codec: %s\n", 
           stream_info.num_images, stream_info.codec_name);
    
    // Step 5: Create sub-code stream with ROI view
    printf("[5] Creating sub-code stream with ROI view...\n");
    nvimgcodecCodeStream_t sub_stream = nullptr;
    
    // Create a view for image 0 with a small ROI
    nvimgcodecRegion_t region{};
    region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
    region.struct_size = sizeof(nvimgcodecRegion_t);
    region.ndim = 2;
    region.start[0] = 0;   // y start
    region.start[1] = 0;   // x start
    region.end[0] = 256;   // y end
    region.end[1] = 256;   // x end
    
    nvimgcodecCodeStreamView_t view{};
    view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
    view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
    view.image_idx = 0;
    view.region = region;
    
    CHECK_NVIMGCODEC(nvimgcodecCodeStreamGetSubCodeStream(main_stream, &sub_stream, &view));
    printf("    ✓ Sub-code stream created (ROI: 256x256 at 0,0)\n");
    
    // Step 6: Prepare decode output
    printf("[6] Preparing decode output...\n");
    
    // Allocate GPU buffer
    size_t buffer_size = 256 * 256 * 3;  // RGB
    void* gpu_buffer = nullptr;
    cudaMalloc(&gpu_buffer, buffer_size);
    
    nvimgcodecImageInfo_t image_info{};
    image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
    image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
    image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    image_info.num_planes = 1;
    image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    image_info.buffer = gpu_buffer;
    image_info.plane_info[0].height = 256;
    image_info.plane_info[0].width = 256;
    image_info.plane_info[0].num_channels = 3;
    image_info.plane_info[0].row_stride = 256 * 3;
    image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    image_info.cuda_stream = 0;
    
    nvimgcodecImage_t image = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &image, &image_info));
    printf("    ✓ Image object created\n");
    
    // Step 7: Decode
    printf("[7] Decoding...\n");
    nvimgcodecDecodeParams_t decode_params{};
    decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
    decode_params.apply_exif_orientation = 1;
    
    nvimgcodecFuture_t future = nullptr;
    CHECK_NVIMGCODEC(nvimgcodecDecoderDecode(decoder, &sub_stream, &image, 1, &decode_params, &future));
    
    nvimgcodecProcessingStatus_t proc_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    size_t status_size = 1;
    CHECK_NVIMGCODEC(nvimgcodecFutureGetProcessingStatus(future, &proc_status, &status_size));
    
    cudaDeviceSynchronize();
    
    if (proc_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        printf("    ✓ Decode successful!\n");
    } else {
        printf("    ✗ Decode failed (status: %d)\n", proc_status);
    }
    
    // Step 8: Cleanup - destroy future and image first
    printf("[8] Cleanup: Destroying future and image...\n");
    nvimgcodecFutureDestroy(future);
    printf("    ✓ Future destroyed\n");
    nvimgcodecImageDestroy(image);
    printf("    ✓ Image destroyed\n");
    
    // Step 9: THIS IS WHERE THE CRASH HAPPENS
    printf("[9] Destroying sub-code stream...\n");
    printf("    >>> CRASH EXPECTED HERE <<<\n");
    nvimgcodecCodeStreamDestroy(sub_stream);  // <-- CRASH: "free(): invalid pointer"
    printf("    ✓ Sub-code stream destroyed (if you see this, no crash!)\n");
    
    // Step 10: Cleanup main stream, decoder, instance
    printf("[10] Destroying main code stream...\n");
    nvimgcodecCodeStreamDestroy(main_stream);
    printf("    ✓ Main code stream destroyed\n");
    
    printf("[11] Destroying decoder...\n");
    nvimgcodecDecoderDestroy(decoder);
    printf("    ✓ Decoder destroyed\n");
    
    printf("[12] Destroying instance...\n");
    nvimgcodecInstanceDestroy(instance);
    printf("    ✓ Instance destroyed\n");
    
    // GPU cleanup
    cudaFree(gpu_buffer);
    
    printf("\n=== Test completed successfully (no crash) ===\n");
    return 0;
}

