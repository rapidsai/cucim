/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_PROFILER_NVTX3_H
#define CUCIM_PROFILER_NVTX3_H

#include "cucim/core/framework.h"

#if CUCIM_SUPPORT_NVTX

#    include <cucim/cuimage.h>
#    include <nvtx3/nvtx3.hpp>


// Override nvtx3::v1::scoped_range_in to check whether if "trace" is enabled or not
namespace nvtx3::v1
{

template <class D = domain::global>
class cucim_scoped_range_in
{
    bool enabled_ = false;

public:
    explicit cucim_scoped_range_in(event_attributes const& attr) noexcept
    {
#    ifndef NVTX_DISABLE
        enabled_ = ::cucim::CuImage::is_trace_enabled();
        if (enabled_)
        {
            nvtxDomainRangePushEx(domain::get<D>(), attr.get());
        }
#    else
        (void)attr;
#    endif
    }

    template <typename... Args>
    explicit cucim_scoped_range_in(Args const&... args) noexcept : cucim_scoped_range_in{ event_attributes{ args... } }
    {
    }

    cucim_scoped_range_in() noexcept : cucim_scoped_range_in{ event_attributes{} }
    {
    }

    void* operator new(std::size_t) = delete;

    cucim_scoped_range_in(cucim_scoped_range_in const&) = delete;
    cucim_scoped_range_in& operator=(cucim_scoped_range_in const&) = delete;
    cucim_scoped_range_in(cucim_scoped_range_in&&) = delete;
    cucim_scoped_range_in& operator=(cucim_scoped_range_in&&) = delete;

    ~cucim_scoped_range_in() noexcept
    {
#    ifndef NVTX_DISABLE
        if (enabled_)
        {
            nvtxDomainRangePop(domain::get<D>());
        }
#    endif
    }
};

} // namespace nvtx3::v1


#    define PROF_SCOPED_RANGE(...)                                                                                     \
        cucim::profiler::scoped_range _p                                                                               \
        {                                                                                                              \
            __VA_ARGS__                                                                                                \
        }

#    define PROF_EVENT(name)                                                                                           \
        cucim::profiler::message::get<cucim::profiler::message_##name>(), cucim::profiler::message_##name::category(), \
            cucim::profiler::message_##name::color

#    define PROF_EVENT_P(name, p)                                                                                      \
        cucim::profiler::message::get<cucim::profiler::message_##name>(), cucim::profiler::message_##name::category(), \
            cucim::profiler::message_##name::color, nvtx3::payload                                                     \
        {                                                                                                              \
            p                                                                                                          \
        }
#    define PROF_MESSAGE(name) cucim::profiler::message::get<cucim::profiler::message_##name>()
#    define PROF_CATEGORY(name) cucim::profiler::category::get<cucim::profiler::category_##name>()

#    define PROF_RGB(r, g, b)                                                                                          \
        nvtx3::rgb                                                                                                     \
        {                                                                                                              \
            r, g, b                                                                                                    \
        }
#    define PROF_ARGB(a, r, g, b)                                                                                      \
        nvtx3::argb                                                                                                    \
        {                                                                                                              \
            a, r, g, b                                                                                                 \
        }

#    define DEFINE_MESSAGE(id, msg)                                                                                    \
        struct message_##id                                                                                            \
        {                                                                                                              \
            static constexpr char const* message{ msg };                                                               \
        }

#    define DEFINE_EVENT(id, msg, c, a, r, g, b)                                                                       \
        struct message_##id                                                                                            \
        {                                                                                                              \
            static constexpr char const* message{ msg };                                                               \
            static constexpr auto category = []() {                                                                    \
                return cucim::profiler::category::get<cucim::profiler::category_##c>();                                \
            };                                                                                                         \
            static constexpr nvtx3::color color{ nvtx3::argb{ a, r, g, b } };                                          \
        }


namespace cucim::profiler
{

// Domain
struct domain
{
    static constexpr char const* name{ "cuCIM" };
};

// Aliases
using scoped_range = nvtx3::cucim_scoped_range_in<domain>;
using category = nvtx3::named_category<domain>;
using message = nvtx3::registered_string<domain>;

// Category
struct category_io
{
    static constexpr char const* name{ "IO" };
    static constexpr uint32_t id{ 10 };
};

struct category_memory
{
    static constexpr char const* name{ "Memory" };
    static constexpr uint32_t id{ 20 };
};

struct category_compute
{
    static constexpr char const* name{ "Compute" };
    static constexpr uint32_t id{ 30 };
};

// Message/Event
DEFINE_EVENT(cucim_malloc, "cucim_malloc()", memory, 255, 63, 72, 204);
DEFINE_EVENT(cucim_free, "cucim_free()", memory, 255, 211, 213, 245);

DEFINE_EVENT(cucim_plugin_detect_image_format, "ImageFormat::detect_image_format()", io, 255, 255, 0, 0);

DEFINE_EVENT(cuimage_cuimage, "CuImage::CuImage()", io, 255, 255, 0, 0);
DEFINE_EVENT(cuimage__cuimage, "CuImage::~CuImage()", io, 255, 251, 207, 208);
DEFINE_EVENT(cuimage_ensure_init, "CuImage::ensure_init()", io, 255, 255, 0, 0);
DEFINE_EVENT(cuimage_ensure_init_plugin_iter, "CuImage::ensure_init::plugin_iter", io, 255, 255, 0, 0);
DEFINE_EVENT(cuimage_cuimage_open, "CuImage::CuImage::open", io, 255, 255, 0, 0);

DEFINE_EVENT(cuimage_read_region, "CuImage::read_region()", io, 255, 255, 0, 0);
DEFINE_EVENT(cuimage_associated_image, "CuImage::associated_image()", io, 255, 255, 0, 0);
DEFINE_EVENT(cuimage_crop_image, "CuImage::crop_image()", io, 255, 255, 0, 0);

DEFINE_EVENT(image_cache_create_cache, "ImageCacheManager::create_cache()", memory, 255, 63, 72, 204);

DEFINE_EVENT(tiff_tiff, "TIFF::TIFF()", io, 255, 255, 0, 0);
DEFINE_EVENT(tiff__tiff, "TIFF::~TIFF()", io, 255, 251, 207, 208);
DEFINE_EVENT(tiff_construct_ifds, "TIFF::construct_ifds()", io, 255, 255, 0, 0);
DEFINE_EVENT(tiff_resolve_vendor_format, "TIFF::resolve_vendor_format()", io, 255, 255, 0, 0);
DEFINE_EVENT(tiff_read, "TIFF::read()", io, 255, 255, 0, 0);
DEFINE_EVENT(tiff_read_associated_image, "TIFF::read_associated_image()", io, 255, 255, 0, 0);

DEFINE_EVENT(ifd_ifd, "IFD::IFD()", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read, "IFD::read()", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_slowpath, "IFD::read::slow_path", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles, "IFD::read_region_tiles()", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles_iter, "IFD::read_region_tiles::iter", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles_task, "IFD::read_region_tiles::task", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles_boundary, "IFD::read_region_tiles_boundary()", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles_boundary_iter, "IFD::read_region_tiles_boundary::iter", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_read_region_tiles_boundary_task, "IFD::read_region_tiles_boundary::task", io, 255, 255, 0, 0);
DEFINE_EVENT(ifd_decompression, "IFD::decompression", compute, 255, 0, 255, 0);

DEFINE_EVENT(decoder_libjpeg_turbo_tjAlloc, "libjpeg-turbo::tjAlloc()", memory, 255, 63, 72, 204);
DEFINE_EVENT(decoder_libjpeg_turbo_tjFree, "libjpeg-turbo::tjFree()", memory, 255, 211, 213, 245);
DEFINE_EVENT(decoder_libjpeg_turbo_tjInitDecompress, "libjpeg-turbo::tjInitDecompress()", compute, 255, 0, 255, 0);
DEFINE_EVENT(decoder_libjpeg_turbo_tjDecompressHeader3, "libjpeg-turbo::tjDecompressHeader3()", compute, 255, 0, 255, 0);
DEFINE_EVENT(decoder_libjpeg_turbo_tjDestroy, "libjpeg-turbo::tjDestroy()", compute, 255, 0, 255, 0);
DEFINE_EVENT(
    decoder_libjpeg_turbo_read_jpeg_header_tables, "cuslide::jpeg::read_jpeg_header_tables()", compute, 255, 0, 255, 0);
DEFINE_EVENT(decoder_libjpeg_turbo_jpeg_decode_buffer, "cuslide::jpeg::jpeg_decode_buffer()", compute, 255, 0, 255, 0);

DEFINE_EVENT(libdeflate_alloc_decompressor, "libdeflate::libdeflate_alloc_decompressor()", memory, 255, 63, 72, 204);
DEFINE_EVENT(libdeflate_zlib_decompress, "libdeflate::libdeflate_zlib_decompress()", compute, 255, 0, 255, 0);
DEFINE_EVENT(libdeflate_free_decompressor, "libdeflate::libdeflate_free_decompressor()", memory, 255, 211, 213, 245);

DEFINE_EVENT(opj_stream_create, "libopenjpeg::opj_stream_create()", compute, 255, 0, 255, 0);
DEFINE_EVENT(opj_create_decompress, "libopenjpeg::opj_create_decompress()", compute, 255, 0, 255, 0);

DEFINE_EVENT(opj_read_header, "libopenjpeg::opj_read_header()", compute, 255, 0, 255, 0);
DEFINE_EVENT(opj_decode, "libopenjpeg::opj_decode()", compute, 255, 0, 255, 0);
DEFINE_EVENT(color_sycc_to_rgb, "libopenjpeg::color_sycc_to_rgb()", compute, 255, 0, 255, 0);
DEFINE_EVENT(color_apply_icc_profile, "libopenjpeg::color_apply_icc_profile()", compute, 255, 0, 255, 0);
DEFINE_EVENT(opj_destructions, "libopenjpeg::opj_destructions", compute, 255, 0, 255, 0);

DEFINE_EVENT(jpeg2k_fast_sycc422_to_rgb, "jpeg2k::fast_sycc422_to_rgb()", compute, 255, 0, 255, 0);
DEFINE_EVENT(jpeg2k_fast_sycc420_to_rgb, "jpeg2k::fast_sycc420_to_rgb()", compute, 255, 0, 255, 0);
DEFINE_EVENT(jpeg2k_fast_sycc444_to_rgb, "jpeg2k::fast_sycc444_to_rgb()", compute, 255, 0, 255, 0);
DEFINE_EVENT(jpeg2k_fast_image_to_rgb, "jpeg2k::fast_image_to_rgb()", compute, 255, 0, 255, 0);

DEFINE_EVENT(lzw_TIFFInitLZW, "lzw::TIFFInitLZW()", compute, 255, 0, 255, 0);
DEFINE_EVENT(lzw_LZWSetupDecode, "lzw::LZWSetupDecode()", compute, 255, 0, 255, 0);
DEFINE_EVENT(lzw_LZWPreDecode, "lzw::LZWPreDecode()", compute, 255, 0, 255, 0);
DEFINE_EVENT(lzw_LZWDecode, "lzw::LZWDecode()", compute, 255, 0, 255, 0);
DEFINE_EVENT(lzw_LZWCleanup, "lzw::LZWCleanup()", compute, 255, 0, 255, 0);
DEFINE_EVENT(lzw_horAcc8, "lzw::LZWCleanup()", compute, 255, 0, 255, 0);

} // namespace cucim::profiler

#else

#    define PROF_SCOPED_RANGE(...) ((void)0)
#    define PROF_EVENT(name) ((void)0)
#    define PROF_EVENT_P(name, p) ((void)0)
#    define PROF_MESSAGE(name) ((void)0)
#    define PROF_CATEGORY(name) ((void)0)
#    define PROF_RGB(r, g, b) ((void)0)
#    define PROF_ARGB(a, r, g, b) ((void)0)
#    define DEFINE_MESSAGE(id, msg) ((void)0)
#    define DEFINE_EVENT(id, msg, c, p, a, r, g, b) ((void)0)

#endif // CUCIM_SUPPORT_NVTX

#endif // CUCIM_PROFILER_NVTX3_H
