/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/cache/image_cache.h"

#include "cucim/cuimage.h"

namespace cucim::cache
{

ImageCacheKey::ImageCacheKey(uint64_t file_hash, uint64_t index) : file_hash(file_hash), location_hash(index)
{
}

ImageCacheValue::ImageCacheValue(void* data, uint64_t size, void* user_obj, const cucim::io::DeviceType device_type)
    : data(data), size(size), user_obj(user_obj), device_type(device_type)
{
}

ImageCacheValue::operator bool() const
{
    return data != nullptr;
}


ImageCache::ImageCache(const ImageCacheConfig& config, CacheType type, const cucim::io::DeviceType device_type)
    : type_(type), device_type_(device_type), config_(config){};

CacheType ImageCache::type() const
{
    return type_;
}

const char* ImageCache::type_str() const
{
    return "nocache";
}

cucim::io::DeviceType ImageCache::device_type() const
{
    return device_type_;
}

ImageCacheConfig& ImageCache::config()
{
    return config_;
}

ImageCacheConfig ImageCache::get_config() const
{
    return config_;
}


} // namespace cucim::cache
