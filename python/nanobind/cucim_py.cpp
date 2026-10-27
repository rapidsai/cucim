/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim_py.h"
#include "cucim_pydoc.h"

#include <cstring>
#include <memory>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nlohmann/json.hpp>

#include <cucim/cuimage.h>

#include "cache/cache_py.h"
#include "compat.h"
#include "filesystem/filesystem_py.h"
#include "io/io_py.h"
#include "profiler/profiler_py.h"

using json = nlohmann::json;
using namespace nanobind::literals;

namespace cucim
{

static const std::unordered_map<uint8_t, const char*> g_dldata_typecode{
    { kDLInt, "DLInt" },
    { kDLUInt, "DLUInt" },
    { kDLFloat, "DLFloat" },
    { kDLBfloat, "DLBfloat" },
};

NB_MODULE(_cucim, m)
{

    // clang-format off
#ifdef CUCIM_VERSION
#  define XSTR(x) STR(x)
#  define STR(x) #x
    // Set version
    m.attr("__version__") = XSTR(CUCIM_VERSION);
#endif
    // clang-format on
    // Get/set plugin root path
    m.def("_get_plugin_root", &get_plugin_root);
    m.def("_set_plugin_root", &set_plugin_root);

    // Submodule: io
    io::init_io(m.def_submodule("io"));

    // Submodule: filesystem
    filesystem::init_filesystem(m.def_submodule("filesystem"));

    // Submodule: cache
    cache::init_cache(m.def_submodule("cache"));

    // Submodule: profiler
    profiler::init_profiler(m.def_submodule("profiler"));

    // Data structures
    bind_enum<DLDataTypeCode>(m, "DLDataTypeCode") //
        .value("DLInt", kDLInt) //
        .value("DLUInt", kDLUInt) //
        .value("DLFloat", kDLFloat) //
        .value("DLBfloat", kDLBfloat);

    nb::class_<DLDataType>(m, "DLDataType", nb::is_weak_referenceable()) //
        .def(
            "__init__",
            [](DLDataType* self, DLDataTypeCode code, uint8_t bits, uint16_t lanes) {
                new (self) DLDataType{ static_cast<uint8_t>(code), bits, lanes };
            },
            doc::DLDataType::doc_DLDataType)
        .def_ro("code", &DLDataType::code, doc::DLDataType::doc_code) //
        .def_ro("bits", &DLDataType::bits, doc::DLDataType::doc_bits) //
        .def_ro("lanes", &DLDataType::lanes, doc::DLDataType::doc_lanes) //
        .def(nb::self == nb::self) //
        .def(nb::self != nb::self) //
        .def("__repr__",
             [](const DLDataType& dtype) {
                 return fmt::format("<cucim.clara.DLDataType code:{}({}) bits:{} lanes:{}>",
                                    g_dldata_typecode.at(dtype.code), dtype.code, dtype.bits, dtype.lanes);
             })
        // nanobind, unlike pybind11, keeps the inherited __hash__ when __eq__ is bound.
        .attr("__hash__") = nb::none();

    nb::class_<CuImageIterator<CuImage>>(m, "CuImageIterator", nb::is_weak_referenceable()) //
        .def(nb::init<std::shared_ptr<CuImage>, Bool>(), doc::CuImageIterator::doc_CuImageIterator,
             "cuimg"_a.none(), //
             "ending"_a.none() = false, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__len__",
            [](const CuImageIterator<CuImage>& it) { //
                return it.size(); //
            }, //
            nb::call_guard<nb::gil_scoped_release>())
        .def("__iter__", //
             [](nb::handle self) { //
                 return self; //
             })
        .def("__next__", &py_cuimage_iterator_next)
        .def(
            "__repr__", //
            [](CuImageIterator<CuImage>& it) { return fmt::format("<cucim.CuImageIterator index:{}>", it.index()); },
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<CuImage>(m, "CuImage", nb::dynamic_attr(), nb::is_weak_referenceable()) //
        .def(
            "__init__", [](CuImage* self, const std::filesystem::path& path) { new (self) CuImage(path.native()); },
            "path"_a, doc::CuImage::doc_CuImage, nb::call_guard<nb::gil_scoped_release>()) //
        .def_static("cache", &py_cache, doc::CuImage::doc_cache, //
                    "type"_a = nb::none(), "kwargs"_a) //
        .def_static("device_cache", &CuImage::device_cache, doc::CuImage::doc_device_cache,
                    nb::call_guard<nb::gil_scoped_release>()) //
        .def_static("profiler", &py_profiler, doc::CuImage::doc_profiler, "kwargs"_a) //
        .def_prop_ro_static(
            "is_trace_enabled", [](nb::handle /* cls */) { return CuImage::is_trace_enabled(); },
            doc::CuImage::doc_is_trace_enabled) //
        // Do not release GIL
        .def_static("_set_array_interface", &_set_array_interface, doc::CuImage::doc__set_array_interface, //
                    "cuimg"_a = nb::none()) //
        .def_prop_ro("path", &CuImage::path, doc::CuImage::doc_path, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro(
            "is_loaded", &CuImage::is_loaded, doc::CuImage::doc_is_loaded, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("device", &CuImage::device, doc::CuImage::doc_device, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("raw_metadata", &CuImage::raw_metadata, doc::CuImage::doc_raw_metadata,
                     nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("metadata", &py_metadata, doc::CuImage::doc_metadata) //
        .def_prop_ro("ndim", &CuImage::ndim, doc::CuImage::doc_ndim, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("dims", &CuImage::dims, doc::CuImage::doc_dims, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("shape", &CuImage::shape, doc::CuImage::doc_shape, nb::call_guard<nb::gil_scoped_release>()) //
        .def("size", &CuImage::size, doc::CuImage::doc_size, nb::call_guard<nb::gil_scoped_release>(), //
             "dim_order"_a = "" //
             ) //
        .def_prop_ro("dtype", &CuImage::dtype, doc::CuImage::doc_dtype, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("typestr", &CuImage::typestr, doc::CuImage::doc_typestr, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("channel_names", &CuImage::channel_names, doc::CuImage::doc_channel_names,
                     nb::call_guard<nb::gil_scoped_release>()) //
        .def("spacing", &CuImage::spacing, doc::CuImage::doc_spacing, nb::call_guard<nb::gil_scoped_release>(), //
             "dim_order"_a = "" //
             ) //
        .def("spacing_units", &CuImage::spacing_units, doc::CuImage::doc_spacing_units,
             nb::call_guard<nb::gil_scoped_release>(), //
             "dim_order"_a = "" //
             ) //
        .def_prop_ro("origin", &CuImage::origin, doc::CuImage::doc_origin, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro(
            "direction", &CuImage::direction, doc::CuImage::doc_direction, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro(
            "coord_sys", &CuImage::coord_sys, doc::CuImage::doc_coord_sys, nb::call_guard<nb::gil_scoped_release>()) //
        .def_prop_ro("resolutions", &py_resolutions, doc::CuImage::doc_resolutions) //
        .def("read_region", &py_read_region, doc::CuImage::doc_read_region, //
             "location"_a = nb::tuple(), //
             "size"_a = nb::tuple(), //
             "level"_a = 0, //
             "num_workers"_a = 0, //
             "batch_size"_a = 1, //
             "drop_last"_a.none() = false, //
             "prefetch_factor"_a = 2, //
             "shuffle"_a.none() = false, //
             "seed"_a = 0, //
             "device"_a = io::Device(), //
             "buf"_a = nb::none(), //
             "shm_name"_a = "", //
             "kwargs"_a) //
        .def_prop_ro("associated_images", &CuImage::associated_images, doc::CuImage::doc_associated_images,
                     nb::call_guard<nb::gil_scoped_release>()) //
        .def("associated_image", &py_associated_image, doc::CuImage::doc_associated_image, //
             "name"_a = "", //
             "device"_a = io::Device()) //
        .def(
            "save", [](const CuImage& cuimg, const std::filesystem::path& file_path) { cuimg.save(file_path.native()); },
            doc::CuImage::doc_save, nb::call_guard<nb::gil_scoped_release>()) //
        .def("close", &CuImage::close, doc::CuImage::doc_close, nb::call_guard<nb::gil_scoped_release>()) //
        .def("__bool__", &CuImage::operator bool, nb::call_guard<nb::gil_scoped_release>()) //
        .def(
            "__iter__", //
            [](const std::shared_ptr<CuImage>& cuimg) { //
                return cuimg->begin(); //
            }, //
            nb::call_guard<nb::gil_scoped_release>())
        .def("__enter__",
             [](nb::handle self) { //
                 return self; //
             })
        .def(
            "__exit__",
            [](CuImage& cuimg, nb::handle /* type */, nb::handle /* value */, nb::handle /* traceback */) { //
                nb::gil_scoped_release release;
                cuimg.close(); //
            },
            "type"_a.none(), "value"_a.none(), "traceback"_a.none())
        .def(
            "__repr__", //
            [](const CuImage& cuimg) { //
                return fmt::format("<cucim.CuImage path:{}>", cuimg.path());
            },
            nb::call_guard<nb::gil_scoped_release>());
}

std::string get_plugin_root()
{
    return CuImage::get_framework()->get_plugin_root();
}

void set_plugin_root(const std::filesystem::path& path)
{
    CuImage::get_framework()->set_plugin_root(path.c_str());
}

template <typename T>
static nb::tuple vector2pytuple(const T& vec)
{
    nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(vec.size())));
    if (!result.is_valid())
    {
        throw nb::python_error();
    }
    Py_ssize_t index = 0;
    for (const auto& value : vec)
    {
        PyTuple_SetItem(result.ptr(), index++, nb::cast(value).release().ptr());
    }
    return result;
}

std::shared_ptr<cucim::cache::ImageCache> py_cache(nb::handle type, nb::kwargs kwargs)
{
    if (nb::isinstance<nb::str>(type))
    {
        std::string ctype = nb::cast<std::string>(type);

        cucim::cache::CacheType cache_type = cucim::cache::lookup_cache_type(ctype);
        // Copy default cache config to local
        cucim::cache::ImageCacheConfig config = cucim::CuImage::get_config()->cache();
        config.type = cache_type;

        if (kwargs.contains("memory_capacity"))
        {
            config.memory_capacity = cast_or_runtime_error<uint32_t>(kwargs["memory_capacity"]);
        }
        if (kwargs.contains("capacity"))
        {
            config.capacity = cast_or_runtime_error<uint32_t>(kwargs["capacity"]);
        }
        else
        {
            // Update capacity depends on memory_capacity.
            config.capacity = cucim::cache::calc_default_cache_capacity(cucim::cache::kOneMiB * config.memory_capacity);
        }
        if (kwargs.contains("mutex_pool_capacity"))
        {
            config.mutex_pool_capacity = cast_or_runtime_error<uint32_t>(kwargs["mutex_pool_capacity"]);
        }
        if (kwargs.contains("list_padding"))
        {
            config.list_padding = cast_or_runtime_error<uint32_t>(kwargs["list_padding"]);
        }
        if (kwargs.contains("extra_shared_memory_size"))
        {
            config.extra_shared_memory_size = cast_or_runtime_error<uint32_t>(kwargs["extra_shared_memory_size"]);
        }
        if (kwargs.contains("record_stat"))
        {
            config.record_stat = cast_or_runtime_error<Bool>(kwargs["record_stat"]);
        }
        return CuImage::cache(config);
    }
    else if (type.is_none())
    {
        return CuImage::cache();
    }

    throw std::invalid_argument(
        fmt::format("The first argument should be one of ['nocache', 'per_process', 'shared_memory']."));
}

std::shared_ptr<cucim::profiler::Profiler> py_profiler(nb::kwargs kwargs)
{
    if (nb::len(kwargs) == 0)
    {
        nb::gil_scoped_release release;
        return CuImage::profiler();
    }
    else
    {
        // Copy default profiler config to local
        cucim::profiler::ProfilerConfig config = cucim::CuImage::get_config()->profiler();

        if (kwargs.contains("trace"))
        {
            config.trace = cast_or_runtime_error<Bool>(kwargs["trace"]);
        }
        nb::gil_scoped_release release;
        return CuImage::profiler(config);
    }
}

static nb::object json2py(const json& j)
{
    switch (j.type())
    {
    case json::value_t::null:
        return nb::none();
    case json::value_t::boolean:
        return nb::bool_(j.get<bool>());
    case json::value_t::number_unsigned:
        return nb::int_(j.get<json::number_unsigned_t>());
    case json::value_t::number_integer:
        return nb::int_(j.get<json::number_integer_t>());
    case json::value_t::number_float:
        return nb::float_(j.get<double>());
    case json::value_t::string: {
        const auto& str = j.get_ref<const std::string&>();
        return nb::str(str.data(), str.size());
    }
    case json::value_t::array: {
        nb::list list;
        for (const auto& item : j)
        {
            list.append(json2py(item));
        }
        return list;
    }
    case json::value_t::object: {
        nb::dict dict;
        for (const auto& [key, value] : j.items())
        {
            dict[nb::str(key.data(), key.size())] = json2py(value);
        }
        return dict;
    }
    default:
        throw std::runtime_error(fmt::format("Unsupported JSON value type '{}' in metadata", j.type_name()));
    }
}

nb::object py_metadata(const CuImage& cuimg)
{
    json json_obj;
    {
        nb::gil_scoped_release release;

        auto metadata = cuimg.metadata();
        json_obj = json::parse(metadata.empty() ? "{}" : metadata);

        // Append basic metadata for the image
        auto item_iter = json_obj.emplace("cucim", json::object());
        json& cucim_metadata = *(item_iter.first);

        cucim_metadata.emplace("path", cuimg.path());
        cucim_metadata.emplace("ndim", cuimg.ndim());
        cucim_metadata.emplace("dims", cuimg.dims());
        cucim_metadata.emplace("shape", cuimg.shape());
        {
            const auto& dtype = cuimg.dtype();
            cucim_metadata.emplace(
                "dtype", json::object({ { "code", dtype.code }, { "bits", dtype.bits }, { "lanes", dtype.lanes } }));
        }
        cucim_metadata.emplace("typestr", cuimg.typestr());
        cucim_metadata.emplace("channel_names", cuimg.channel_names());
        cucim_metadata.emplace("spacing", cuimg.spacing());
        cucim_metadata.emplace("spacing_units", cuimg.spacing_units());
        cucim_metadata.emplace("origin", cuimg.origin());
        cucim_metadata.emplace("direction", cuimg.direction());
        cucim_metadata.emplace("coord_sys", cuimg.coord_sys());
        {
            const auto& resolutions = cuimg.resolutions();
            auto resolutions_iter = cucim_metadata.emplace("resolutions", json::object());
            json& resolutions_metadata = *(resolutions_iter.first);
            auto level_count = resolutions.level_count();
            resolutions_metadata.emplace("level_count", level_count);
            std::vector<std::vector<int64_t>> level_dimensions_vec;
            level_dimensions_vec.reserve(level_count);
            for (int level = 0; level < level_count; ++level)
            {
                level_dimensions_vec.emplace_back(resolutions.level_dimension(level));
            }
            resolutions_metadata.emplace("level_dimensions", level_dimensions_vec);
            resolutions_metadata.emplace("level_downsamples", resolutions.level_downsamples());
            std::vector<std::vector<uint32_t>> level_tile_sizes_vec;
            level_tile_sizes_vec.reserve(level_count);
            for (int level = 0; level < level_count; ++level)
            {
                level_tile_sizes_vec.emplace_back(resolutions.level_tile_size(level));
            }
            resolutions_metadata.emplace("level_tile_sizes", level_tile_sizes_vec);
        }
        cucim_metadata.emplace("associated_images", cuimg.associated_images());
    }
    return json2py(json_obj);
}

nb::dict py_resolutions(const CuImage& cuimg)
{
    const auto& resolutions = cuimg.resolutions();
    auto level_count = resolutions.level_count();
    nb::dict result;
    if (resolutions.level_count() == 0)
    {
        result["level_count"] = 0;
        result["level_dimensions"] = nb::tuple();
        result["level_downsamples"] = nb::tuple();
        result["level_tile_sizes"] = nb::tuple();
        return result;
    }

    std::vector<nb::tuple> level_dimensions_vec;
    level_dimensions_vec.reserve(level_count);
    std::vector<nb::tuple> level_tile_sizes_vec;
    level_tile_sizes_vec.reserve(level_count);
    for (int level = 0; level < level_count; ++level)
    {
        level_dimensions_vec.emplace_back(vector2pytuple(resolutions.level_dimension(level)));
        level_tile_sizes_vec.emplace_back(vector2pytuple(resolutions.level_tile_size(level)));
    }

    result["level_count"] = level_count;
    result["level_dimensions"] = vector2pytuple(level_dimensions_vec);
    result["level_downsamples"] = vector2pytuple(resolutions.level_downsamples());
    result["level_tile_sizes"] = vector2pytuple(level_tile_sizes_vec);
    return result;
}

// struct-module format codes of a native int64 ("@" is the default, native byte order and native sizes).
static bool is_int64_format(const char* format)
{
    if (format == nullptr)
    {
        return false;
    }
    if (*format == '@')
    {
        ++format;
    }
    return std::strcmp(format, "q") == 0 || (sizeof(long) == sizeof(int64_t) && std::strcmp(format, "l") == 0);
}

static std::vector<int64_t> parse_locations(nb::iterable location)
{
    Py_buffer view;
    if (PyObject_GetBuffer(location.ptr(), &view, PyBUF_FULL_RO) == 0)
    {
        std::unique_ptr<Py_buffer, decltype(&PyBuffer_Release)> release(&view, PyBuffer_Release);
        if (!is_int64_format(view.format))
        {
            throw std::invalid_argument("Expected int64 array-like");
        }
        if (PyBuffer_IsContiguous(&view, 'C') == 0)
        {
            throw std::invalid_argument("Expected C-contiguous array-like");
        }
        const auto* data_array = static_cast<const int64_t*>(view.buf);
        return std::vector<int64_t>(data_array, data_array + view.len / view.itemsize);
    }
    PyErr_Clear();

    std::vector<int64_t> locations;
    for (nb::handle item : location)
    {
        if (nb::isinstance<nb::iterable>(item))
        {
            for (nb::handle value : item)
            {
                locations.emplace_back(cast_or_runtime_error<int64_t>(value));
            }
        }
        else
        {
            locations.emplace_back(cast_or_runtime_error<int64_t>(item));
        }
    }
    return locations;
}

nb::object py_read_region(const CuImage& cuimg,
                          nb::iterable location,
                          std::vector<int64_t> size,
                          int16_t level,
                          uint32_t num_workers,
                          uint32_t batch_size,
                          Bool drop_last,
                          uint32_t prefetch_factor,
                          Bool shuffle,
                          uint64_t seed,
                          const io::Device& device,
                          nb::handle buf,
                          const std::string& shm_name,
                          nb::kwargs kwargs)
{
    if (!size.empty() && size.size() != 2)
    {
        throw std::runtime_error("size (patch size) should be 2!");
    }

    std::vector<int64_t> locations = parse_locations(location);

    cucim::DimIndices indices;
    if (nb::len(kwargs) > 0)
    {
        std::vector<std::pair<char, int64_t>> indices_args;

        for (auto [key_obj, value_obj] : kwargs)
        {
            auto key = nb::cast<std::string>(nb::str(key_obj));
            auto value = cast_or_runtime_error<int>(value_obj);

            if (key.size() != 1)
            {
                throw std::invalid_argument(
                    fmt::format("Argument name for Dimension should be a single character but '{}' is used.", key));
            }
            char key_char = key[0] & ~32;
            if (key_char < 'A' || key_char > 'Z')
            {
                throw std::invalid_argument(
                    fmt::format("Dimension character should be an alphabet but '{}' is used.", key));
            }

            indices_args.emplace_back(std::make_pair(key_char, value));
        }
        indices = cucim::DimIndices(indices_args);
    }
    else
    {
        indices = cucim::DimIndices{};
    }

    std::shared_ptr<CuImage> region_ptr;
    bool is_batch = false;
    {
        nb::gil_scoped_release release;

        region_ptr = std::make_shared<cucim::CuImage>(
            cuimg.read_region(std::move(locations), std::move(size), level, num_workers, batch_size, drop_last,
                              prefetch_factor, shuffle, seed, indices, device, nullptr, ""));
        auto loader = region_ptr->loader();
        is_batch = batch_size > 1 || (loader && loader->size() > 1);
    }

    if (is_batch)
    {
        return nb::cast(std::make_shared<CuImageIterator<CuImage>>(region_ptr));
    }

    nb::object region = nb::cast(region_ptr);

    // Add `__array_interface__` or `__cuda_array_interface__` in runtime.
    _set_array_interface(region);

    return region;
}

nb::object py_associated_image(const CuImage& cuimg, const std::string& name, const io::Device& device)
{
    std::shared_ptr<CuImage> image_ptr;
    {
        nb::gil_scoped_release release;
        image_ptr = std::make_shared<cucim::CuImage>(cuimg.associated_image(name, device));
    }

    nb::object image = nb::cast(image_ptr);

    // Add `__array_interface__` or `__cuda_array_interface__` in runtime.
    _set_array_interface(image);

    return image;
}

nb::object py_cuimage_iterator_next(CuImageIterator<CuImage>& it)
{
    std::shared_ptr<CuImage> cuimg;
    {
        nb::gil_scoped_release release;

        if (it.index() == it.size())
        {
            throw nb::stop_iteration();
        }

        // Get the next batch of images.
        ++it;

        cuimg = *it;
    }

    nb::object cuimg_obj = nb::cast(cuimg);
    if (cuimg->loader())
    {
        _set_array_interface(cuimg_obj);
    }
    return cuimg_obj;
}

void _set_array_interface(nb::handle cuimg_obj)
{
    const auto& cuimg = nb::cast<const CuImage&>(cuimg_obj);

    // TODO: using __array_struct__, access to array interface could be faster
    //       (https://numpy.org/doc/stable/reference/arrays.interface.html#c-struct-access)
    // TODO: check the performance difference between python int vs python long later.

    loader::ThreadBatchDataLoader* loader = cuimg.loader();
    memory::DLTContainer container = cuimg.container();

    DLTensor* tensor = static_cast<DLTensor*>(container);
    if (!tensor)
    {
        return;
    }
    if (loader)
    {
        // Get the last available (batch) image.
        tensor->data = loader->data();
    }

    if (tensor->data)
    {
        nb::str typestr(container.numpy_dtype());

        nb::tuple data = nb::make_tuple(reinterpret_cast<uint64_t>(tensor->data), false);
        nb::list descr;
        descr.append(nb::make_tuple("", typestr));

        nb::tuple shape = vector2pytuple(cuimg.shape());

        nb::dict interface;
        interface["data"] = data;
        interface["strides"] = nb::none();
        interface["descr"] = descr;
        interface["typestr"] = typestr;
        interface["shape"] = shape;
        interface["version"] = 3;

        // Depending on container's memory type, expose either array_interface or cuda_array_interface
        switch (tensor->device.device_type)
        {
        case kDLCPU: {
            // Reference: https://numpy.org/doc/stable/reference/arrays.interface.html
            nb::setattr(cuimg_obj, "__array_interface__", interface);
        }
        break;
        case kDLCUDA: {
            // Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
            interface["mask"] = nb::none();
            interface["stream"] = 1;
            nb::setattr(cuimg_obj, "__cuda_array_interface__", interface);
        }
        break;
        default:
            break;
        }
    }
    else
    {
        switch (tensor->device.device_type)
        {
        case kDLCPU: {
            if (nb::hasattr(cuimg_obj, "__array_interface__"))
            {
                nb::delattr(cuimg_obj, "__array_interface__");
            }
        }
        break;
        case kDLCUDA: {
            if (nb::hasattr(cuimg_obj, "__cuda_array_interface__"))
            {
                nb::delattr(cuimg_obj, "__cuda_array_interface__");
            }
        }
        break;
        default:
            break;
        }
    }
}

} // namespace cucim
