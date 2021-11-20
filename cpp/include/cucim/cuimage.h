/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_CUIMAGE_H
#define CUCIM_CUIMAGE_H

#include "cucim/core/framework.h"
#include "cucim/cache/image_cache_manager.h"
#include "cucim/config/config.h"
#include "cucim/filesystem/file_path.h"
#include "cucim/io/device.h"
#include "cucim/io/format/image_format.h"
#include "cucim/memory/dlpack.h"
#include "cucim/plugin/image_format.h"

#include <array>
#include <set>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

namespace cucim
{

using DetectedFormat = std::pair<std::string, std::vector<std::string>>;
using Metadata = std::string;
using Shape = std::vector<int64_t>;

constexpr int64_t kWholeRange = -1;

/**
 *
 * This class is used in both cases:
 *   1. Specifying index for dimension string (e.g., "YXC" => Y:0, X:1, C:2)
 *   2. Specifying index for read_region() (e.g., {{'C', -1}, {'T', 0}} => C:(whole range), T:0)
 */
class EXPORT_VISIBLE DimIndices
{
public:
    DimIndices(const char* dims = nullptr);
    DimIndices(std::vector<std::pair<char, int64_t>> init_list);
    int64_t index(char dim_char) const;

private:
    io::format::DimIndicesDesc dim_indices_{ { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } };
};

class EXPORT_VISIBLE ResolutionInfo
{
public:
    ResolutionInfo(io::format::ResolutionInfoDesc desc);

    uint16_t level_count() const;
    const std::vector<int64_t>& level_dimensions() const;
    std::vector<int64_t> level_dimension(uint16_t level) const;
    const std::vector<float>& level_downsamples() const;
    float level_downsample(uint16_t level) const;
    const std::vector<uint32_t>& level_tile_sizes() const;
    std::vector<uint32_t> level_tile_size(uint16_t level) const;

private:
    uint16_t level_count_;
    uint16_t level_ndim_;
    std::vector<int64_t> level_dimensions_;
    std::vector<float> level_downsamples_;
    std::vector<uint32_t> level_tile_sizes_;
};

/**
 * Detect available formats (plugins) from the input path.
 *
 * The plugin name can be used later to specify the plugin to load the image file explicitly.
 *
 * @param path An input path to detect available formats
 * @return A tuple that describes the format (file format or format vendor) and the list of plugin names that
 * supports the file
 */
DetectedFormat detect_format(filesystem::Path path);

class EXPORT_VISIBLE CuImage : public std::enable_shared_from_this<CuImage>
{
public:
    CuImage(const filesystem::Path& path);
    CuImage(const filesystem::Path& path, const std::string& plugin_name);
    CuImage(const CuImage& cuimg) = delete;
    CuImage(CuImage&& cuimg);
    CuImage(const CuImage* cuimg,
            io::format::ImageMetadataDesc* image_metadata,
            cucim::io::format::ImageDataDesc* image_data);

    ~CuImage();

    operator bool() const
    {
        return !!image_format_ && !is_loaded_;
    }

    static Framework* get_framework();
    static config::Config* get_config();
    static cache::ImageCacheManager& cache_manager();
    static std::shared_ptr<cache::ImageCache> cache();
    static std::shared_ptr<cache::ImageCache> cache(cache::ImageCacheConfig& config);

    filesystem::Path path() const;

    bool is_loaded() const;

    io::Device device() const;

    Metadata raw_metadata() const;

    Metadata metadata() const;

    uint16_t ndim() const;

    std::string dims() const;

    Shape shape() const;

    std::vector<int64_t> size(std::string dim_order = std::string{}) const;

    DLDataType dtype() const;

    std::vector<std::string> channel_names() const;

    std::vector<float> spacing(std::string dim_order = std::string{}) const;

    std::vector<std::string> spacing_units(std::string dim_order = std::string{}) const;

    std::array<float, 3> origin() const;

    std::array<std::array<float, 3>, 3> direction() const;

    std::string coord_sys() const;

    ResolutionInfo resolutions() const;

    memory::DLTContainer container() const;

    CuImage read_region(std::vector<int64_t>&& location,
                        std::vector<int64_t>&& size,
                        uint16_t level = 0,
                        const DimIndices& region_dim_indices = {},
                        const io::Device& device = "cpu",
                        DLTensor* buf = nullptr,
                        const std::string& shm_name = std::string{}) const;

    std::set<std::string> associated_images() const;
    CuImage associated_image(const std::string& name, const io::Device& device = "cpu") const;

    void save(std::string file_path) const;

    void close();

private:
    using Mutex = std::mutex;
    using ScopedLock = std::scoped_lock<Mutex>;

    explicit CuImage();

    void ensure_init();
    bool crop_image(const io::format::ImageReaderRegionRequestDesc& request,
                    io::format::ImageDataDesc& out_image_data) const;


    static Framework* framework_;
    // Note: config_ should be placed before cache_manager_ (cache_manager_ depends on config_)
    static std::unique_ptr<config::Config> config_;
    static std::unique_ptr<cache::ImageCacheManager> cache_manager_;
    static std::unique_ptr<cucim::plugin::ImageFormat> image_format_plugins_;

    mutable Mutex mutex_;
    cucim::io::format::ImageFormatDesc* image_format_ = nullptr;
    CuCIMFileHandle file_handle_{};
    io::format::ImageMetadataDesc* image_metadata_ = nullptr;
    io::format::ImageDataDesc* image_data_ = nullptr;
    bool is_loaded_ = false;
    DimIndices dim_indices_{};
    std::set<std::string> associated_images_;
};

} // namespace cucim

#endif // CUCIM_CUIMAGE_H
