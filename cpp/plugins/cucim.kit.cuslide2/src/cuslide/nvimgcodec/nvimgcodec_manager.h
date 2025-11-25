/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <string>
#include <mutex>
#include <fmt/format.h>

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

/**
 * @brief Singleton manager for nvImageCodec instance and decoder
 * 
 * Provides centralized access to nvImageCodec resources with thread-safe initialization.
 */
class NvImageCodecManager
{
public:
    static NvImageCodecManager& instance()
    {
        static NvImageCodecManager instance;
        return instance;
    }

    nvimgcodecInstance_t get_instance() const { return instance_; }
    nvimgcodecDecoder_t get_decoder() const { return decoder_; }
    nvimgcodecDecoder_t get_cpu_decoder() const { return cpu_decoder_; }  // CPU-only decoder
    bool has_cpu_decoder() const { return cpu_decoder_ != nullptr; }
    std::mutex& get_mutex() { return decoder_mutex_; }
    bool is_initialized() const { return initialized_; }
    const std::string& get_status() const { return status_message_; }

    // Quick API validation test
    bool test_nvimagecodec_api()
    {
        if (!initialized_) return false;
        
        try {
            // Test 1: Get nvImageCodec properties
            nvimgcodecProperties_t props{};
            props.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
            props.struct_size = sizeof(nvimgcodecProperties_t);
            props.struct_next = nullptr;
            
            if (nvimgcodecGetProperties(&props) == NVIMGCODEC_STATUS_SUCCESS)
            {
                uint32_t version = props.version;
                // Use official nvImageCodec version macros (version format: major*1000 + minor*100 + patch)
                
                [[maybe_unused]] uint32_t major = version / 1000;
                [[maybe_unused]] uint32_t minor = (version % 1000) / 100;
                [[maybe_unused]] uint32_t patch = version % 100;
                
                #ifdef DEBUG
                fmt::print("✅ nvImageCodec API Test: Version {}.{}.{}\n", major, minor, patch);
                #endif
                
                // Test 2: Check decoder capabilities
                if (decoder_)
                {
                    #ifdef DEBUG
                    fmt::print("✅ nvImageCodec Decoder: Ready\n");
                    #endif
                    return true;
                }
            }
        }
        catch (const std::exception& e)
        {
            #ifdef DEBUG
            fmt::print("⚠️  nvImageCodec API Test failed: {}\n", e.what());
            #endif
            (void)e;  // Suppress unused warning in release builds
        }
        
        return false;
    }

    // Disable copy/move
    NvImageCodecManager(const NvImageCodecManager&) = delete;
    NvImageCodecManager& operator=(const NvImageCodecManager&) = delete;
    NvImageCodecManager(NvImageCodecManager&&) = delete;
    NvImageCodecManager& operator=(NvImageCodecManager&&) = delete;

private:
    NvImageCodecManager() : initialized_(false)
    {
        try {
            // Create nvImageCodec instance following official API pattern
            nvimgcodecInstanceCreateInfo_t create_info{};
            create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
            create_info.struct_next = nullptr;
            create_info.load_builtin_modules = 1;
            create_info.load_extension_modules = 1;
            create_info.extension_modules_path = nullptr;
            create_info.create_debug_messenger = 1;
            create_info.debug_messenger_desc = nullptr;
            create_info.message_severity = 0;
            create_info.message_category = 0;
        
            if (nvimgcodecInstanceCreate(&instance_, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
            {
                status_message_ = "Failed to create nvImageCodec instance";
                #ifdef DEBUG
                fmt::print("❌ {}\n", status_message_);
                #endif
                return;
            }

            // Create decoder with execution parameters following official API pattern
            nvimgcodecExecutionParams_t exec_params{};
            exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
            exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
            exec_params.struct_next = nullptr;
            exec_params.device_allocator = nullptr;
            exec_params.pinned_allocator = nullptr;
            exec_params.max_num_cpu_threads = 0; // Use default
            exec_params.executor = nullptr;
            exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
            exec_params.pre_init = 0;
            exec_params.skip_pre_sync = 0;
            exec_params.num_backends = 0;
            exec_params.backends = nullptr;
        
            if (nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr) != NVIMGCODEC_STATUS_SUCCESS)
            {
                nvimgcodecInstanceDestroy(instance_);
                instance_ = nullptr;
                status_message_ = "Failed to create nvImageCodec decoder";
                #ifdef DEBUG
                fmt::print("❌ {}\n", status_message_);
                #endif
                return;
            }
            
            // Create CPU-only decoder for native CPU decoding
            nvimgcodecBackendKind_t cpu_backend_kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
            nvimgcodecBackendParams_t cpu_backend_params{};
            cpu_backend_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS;
            cpu_backend_params.struct_size = sizeof(nvimgcodecBackendParams_t);
            cpu_backend_params.struct_next = nullptr;
            
            nvimgcodecBackend_t cpu_backend{};
            cpu_backend.struct_type = NVIMGCODEC_STRUCTURE_TYPE_BACKEND;
            cpu_backend.struct_size = sizeof(nvimgcodecBackend_t);
            cpu_backend.struct_next = nullptr;
            cpu_backend.kind = cpu_backend_kind;
            cpu_backend.params = cpu_backend_params;
            
            nvimgcodecExecutionParams_t cpu_exec_params = exec_params;
            cpu_exec_params.num_backends = 1;
            cpu_exec_params.backends = &cpu_backend;
            
            if (nvimgcodecDecoderCreate(instance_, &cpu_decoder_, &cpu_exec_params, nullptr) == NVIMGCODEC_STATUS_SUCCESS)
            {
                #ifdef DEBUG
                fmt::print("✅ CPU-only decoder created successfully\n");
                #endif
            }
            else
            {
                #ifdef DEBUG
                fmt::print("⚠️  Failed to create CPU-only decoder (CPU decoding will use fallback)\n");
                #endif
                cpu_decoder_ = nullptr;
            }
            
            initialized_ = true;
            status_message_ = "nvImageCodec initialized successfully";
            #ifdef DEBUG
            fmt::print("✅ {}\n", status_message_);
            #endif
            
            // Run quick API test
            test_nvimagecodec_api();
        }
        catch (const std::exception& e)
        {
            status_message_ = fmt::format("nvImageCodec initialization exception: {}", e.what());
            #ifdef DEBUG
            fmt::print("❌ {}\n", status_message_);
            #endif
            initialized_ = false;
        }
    }

    ~NvImageCodecManager()
    {
        if (cpu_decoder_)
        {
            nvimgcodecDecoderDestroy(cpu_decoder_);
            cpu_decoder_ = nullptr;
        }
        
        if (decoder_)
        {
            nvimgcodecDecoderDestroy(decoder_);
            decoder_ = nullptr;
        }
        
        if (instance_)
        {
            nvimgcodecInstanceDestroy(instance_);
            instance_ = nullptr;
        }
    }

    nvimgcodecInstance_t instance_{nullptr};
    nvimgcodecDecoder_t decoder_{nullptr};
    nvimgcodecDecoder_t cpu_decoder_{nullptr};  // CPU-only decoder (uses libjpeg-turbo, etc.)
    bool initialized_{false};
    std::string status_message_;
    std::mutex decoder_mutex_;  // Protect decoder operations from concurrent access
};

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

