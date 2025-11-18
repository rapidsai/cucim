/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE_CPU_DECODER_STUBS_H
#define CUSLIDE_CPU_DECODER_STUBS_H

/**
 * STUB DEFINITIONS FOR CPU DECODERS
 * 
 * This file provides minimal stub definitions for the old CPU decoder namespaces
 * to allow compilation of legacy fallback code paths. These functions throw runtime
 * errors if called, since this is a pure nvImageCodec build.
 * 
 * In a pure nvImageCodec build, all decoding should go through the nvImageCodec
 * path in IFD::read(), so these stubs should never be executed.
 */

#include <stdexcept>
#include <cstdint>
#include <fmt/format.h>
#include <cucim/io/device.h>

namespace cuslide {

namespace raw {
    inline bool decode_raw([[maybe_unused]] int fd, [[maybe_unused]] void* buf, 
                          [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
                          [[maybe_unused]] uint8_t** out, [[maybe_unused]] uint64_t out_size, 
                          [[maybe_unused]] const cucim::io::Device& dev) {
        throw std::runtime_error(
            "CPU decoder (raw) called in pure nvImageCodec build! "
            "This should not happen - check that nvImageCodec path is being used.");
        return false;
    }
}

namespace jpeg {
    inline bool decode_libjpeg([[maybe_unused]] int fd, [[maybe_unused]] unsigned char* buf, 
                              [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
                              [[maybe_unused]] uint8_t* jpegtable_data, [[maybe_unused]] uint32_t jpegtable_count,
                              [[maybe_unused]] uint8_t** out, [[maybe_unused]] const cucim::io::Device& dev, 
                              [[maybe_unused]] int32_t color_space) {
        throw std::runtime_error(
            "CPU decoder (jpeg) called in pure nvImageCodec build! "
            "This should not happen - check that nvImageCodec path is being used.");
        return false;
    }
    
    inline bool get_dimension([[maybe_unused]] void* buf, [[maybe_unused]] uint64_t offset, 
                             [[maybe_unused]] uint64_t size,
                             [[maybe_unused]] uint32_t* width, [[maybe_unused]] uint32_t* height) {
        throw std::runtime_error(
            "CPU decoder (jpeg::get_dimension) called in pure nvImageCodec build!");
        return false;
    }
}

namespace deflate {
    inline bool decode_deflate([[maybe_unused]] int fd, [[maybe_unused]] void* buf, 
                              [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
                              [[maybe_unused]] uint8_t** out, [[maybe_unused]] uint64_t out_size, 
                              [[maybe_unused]] const cucim::io::Device& dev) {
        throw std::runtime_error(
            "CPU decoder (deflate) called in pure nvImageCodec build! "
            "This should not happen - check that nvImageCodec path is being used.");
        return false;
    }
}

namespace lzw {
    inline bool decode_lzw([[maybe_unused]] int fd, [[maybe_unused]] void* buf, 
                          [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
                          [[maybe_unused]] uint8_t** out, [[maybe_unused]] uint64_t out_size, 
                          [[maybe_unused]] const cucim::io::Device& dev) {
        throw std::runtime_error(
            "CPU decoder (lzw) called in pure nvImageCodec build! "
            "This should not happen - check that nvImageCodec path is being used.");
        return false;
    }
    
    inline void horAcc8([[maybe_unused]] uint8_t* buf, [[maybe_unused]] uint64_t size, 
                       [[maybe_unused]] uint32_t stride) {
        throw std::runtime_error(
            "CPU decoder (lzw::horAcc8) called in pure nvImageCodec build!");
    }
}

namespace jpeg2k {
    enum class ColorSpace {
        kSYCC = 0,
        kRGB = 1
    };
    
    inline bool decode_libopenjpeg([[maybe_unused]] int fd, [[maybe_unused]] void* buf, 
                                  [[maybe_unused]] uint64_t offset, [[maybe_unused]] uint64_t size,
                                  [[maybe_unused]] uint8_t** out, [[maybe_unused]] uint64_t out_size,
                                  [[maybe_unused]] const cucim::io::Device& dev, 
                                  [[maybe_unused]] ColorSpace cs) {
        throw std::runtime_error(
            "CPU decoder (jpeg2k/openjpeg) called in pure nvImageCodec build! "
            "This should not happen - check that nvImageCodec path is being used.");
        return false;
    }
}

} // namespace cuslide

#endif // CUSLIDE_CPU_DECODER_STUBS_H

