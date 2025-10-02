/*
 * Diagnostic tool to check nvImageCodec available codecs and backends
 * 
 * Compile:
 *   g++ -o check_nvimgcodec_codecs check_nvimgcodec_codecs.cpp \
 *       -I/usr/local/cuda/include \
 *       -L/usr/local/cuda/lib64 \
 *       -lnvimgcodec -lfmt
 * 
 * Run:
 *   LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ./check_nvimgcodec_codecs
 */

#include <nvimgcodec.h>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::cout << "=" << std::string(70, '=') << std::endl;
    std::cout << "nvImageCodec Codec & Backend Diagnostic Tool" << std::endl;
    std::cout << "=" << std::string(70, '=') << std::endl;
    
    // Step 1: Get nvImageCodec version
    std::cout << "\n📋 nvImageCodec Version Information:" << std::endl;
    nvimgcodecProperties_t props{};
    props.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
    props.struct_size = sizeof(nvimgcodecProperties_t);
    props.struct_next = nullptr;
    
    if (nvimgcodecGetProperties(&props) == NVIMGCODEC_STATUS_SUCCESS)
    {
        uint32_t version = props.version;
        uint32_t major = (version >> 16) & 0xFF;
        uint32_t minor = (version >> 8) & 0xFF;
        uint32_t patch = version & 0xFF;
        
        std::cout << "   Version: " << major << "." << minor << "." << patch << std::endl;
        std::cout << "   CUDA Runtime Version: " << props.cudart_version << std::endl;
        std::cout << "   Extension: " << (props.ext_api_version ? "Available" : "Not Available") << std::endl;
    }
    else
    {
        std::cerr << "   ❌ Failed to get nvImageCodec properties" << std::endl;
        return 1;
    }
    
    // Step 2: Create instance and decoder
    std::cout << "\n🔧 Creating nvImageCodec instance..." << std::endl;
    nvimgcodecInstance_t instance = nullptr;
    
    nvimgcodecInstanceCreateInfo_t create_info{};
    create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
    create_info.struct_next = nullptr;
    create_info.load_builtin_modules = 1;      // Load built-in modules
    create_info.load_extension_modules = 1;    // Load extension modules
    create_info.extension_modules_path = nullptr;  // Use default path
    create_info.create_debug_messenger = 0;
    create_info.debug_messenger_desc = nullptr;
    create_info.message_severity = 0;
    create_info.message_category = 0;
    
    if (nvimgcodecInstanceCreate(&instance, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
    {
        std::cerr << "   ❌ Failed to create instance" << std::endl;
        return 1;
    }
    std::cout << "   ✅ Instance created" << std::endl;
    
    // Step 3: Create decoder with all backends enabled
    std::cout << "\n🔧 Creating decoder (all backends enabled)..." << std::endl;
    nvimgcodecDecoder_t decoder = nullptr;
    
    nvimgcodecExecutionParams_t exec_params{};
    exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
    exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
    exec_params.struct_next = nullptr;
    exec_params.device_allocator = nullptr;
    exec_params.pinned_allocator = nullptr;
    exec_params.max_num_cpu_threads = 0;  // Use default
    exec_params.executor = nullptr;
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    exec_params.pre_init = 0;
    exec_params.skip_pre_sync = 0;
    exec_params.num_backends = 0;      // 0 = all available backends
    exec_params.backends = nullptr;    // nullptr = auto-select
    
    if (nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr) != NVIMGCODEC_STATUS_SUCCESS)
    {
        std::cerr << "   ❌ Failed to create decoder" << std::endl;
        nvimgcodecInstanceDestroy(instance);
        return 1;
    }
    std::cout << "   ✅ Decoder created" << std::endl;
    
    // Step 4: Query available codecs
    std::cout << "\n📊 Querying available codecs..." << std::endl;
    std::cout << "   Note: nvImageCodec 0.6.0 doesn't have a direct API to list codecs" << std::endl;
    std::cout << "   Common codecs: jpeg, jpeg2000, tiff, png, bmp, pnm, webp" << std::endl;
    
    // Step 5: Test CPU vs GPU backend availability
    std::cout << "\n🔍 Backend Detection (CPU vs GPU):" << std::endl;
    std::cout << "   Testing with a sample to determine available backends..." << std::endl;
    std::cout << "   This requires a sample image file to test properly." << std::endl;
    
    std::cout << "\n💡 Backend Configuration:" << std::endl;
    std::cout << "   • num_backends = " << exec_params.num_backends 
              << " (0 = all available)" << std::endl;
    std::cout << "   • backends = " << (exec_params.backends ? "specified" : "auto-select") 
              << std::endl;
    std::cout << "   • load_builtin_modules = " << create_info.load_builtin_modules << std::endl;
    std::cout << "   • load_extension_modules = " << create_info.load_extension_modules << std::endl;
    
    std::cout << "\n📝 Known Backend Plugins:" << std::endl;
    std::cout << "   GPU Backends:" << std::endl;
    std::cout << "   • nvjpeg_decoder    - NVIDIA JPEG decoder (GPU)" << std::endl;
    std::cout << "   • nvjpeg2k_decoder  - NVIDIA JPEG2000 decoder (GPU)" << std::endl;
    std::cout << "   • nvtiff_decoder    - NVIDIA TIFF decoder (GPU)" << std::endl;
    std::cout << "   • libnvjpeg         - CUDA JPEG library" << std::endl;
    std::cout << "\n   CPU Backends (if installed):" << std::endl;
    std::cout << "   • libjpeg_turbo     - CPU JPEG decoder" << std::endl;
    std::cout << "   • opencv_decoder    - OpenCV-based CPU decoder" << std::endl;
    std::cout << "   • pnm_decoder       - CPU PNM decoder" << std::endl;
    
    std::cout << "\n💡 How to Check Installed Backends:" << std::endl;
    std::cout << "   1. Check nvImageCodec installation directory:" << std::endl;
    std::cout << "      ls /usr/local/cuda/lib64/ | grep nvimgcodec" << std::endl;
    std::cout << "      ls /usr/local/cuda/lib64/ | grep libjpeg" << std::endl;
    std::cout << "\n   2. Check for extension modules:" << std::endl;
    std::cout << "      ls /usr/local/cuda/lib64/nvimgcodec_extensions/" << std::endl;
    std::cout << "\n   3. Set NVIMGCODEC_DEBUG=1 for verbose logging:" << std::endl;
    std::cout << "      export NVIMGCODEC_DEBUG=1" << std::endl;
    std::cout << "      # Then run your application" << std::endl;
    
    // Cleanup
    nvimgcodecDecoderDestroy(decoder);
    nvimgcodecInstanceDestroy(instance);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✅ Diagnostic complete" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}

