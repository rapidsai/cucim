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

#include "cucim_py.h"
#include "cucim_pydoc.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cucim/cuimage.h>

#include "cache/cache_py.h"
#include "filesystem/filesystem_py.h"
#include "io/io_py.h"
#include "profiler/profiler_py.h"

using namespace pybind11::literals;
namespace py = pybind11;

namespace cucim
{

static const std::unordered_map<uint8_t, const char*> g_dldata_typecode{
    { kDLInt, "DLInt" },
    { kDLUInt, "DLUInt" },
    { kDLFloat, "DLFloat" },
    { kDLBfloat, "DLBfloat" },
};

PYBIND11_MODULE(_cucim, m)
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
    auto m_io = m.def_submodule("io");
    io::init_io(m_io);

    // Submodule: filesystem
    auto m_fs = m.def_submodule("filesystem");
    filesystem::init_filesystem(m_fs);

    // Submodule: cache
    auto m_cache = m.def_submodule("cache");
    cache::init_cache(m_cache);

    // Submodule: profiler
    auto m_profiler = m.def_submodule("profiler");
    profiler::init_profiler(m_profiler);

    // Data structures
    py::enum_<DLDataTypeCode>(m, "DLDataTypeCode") //
        .value("DLInt", kDLInt) //
        .value("DLUInt", kDLUInt) //
        .value("DLFloat", kDLFloat) //
        .value("DLBfloat", kDLBfloat);

    py::class_<DLDataType>(m, "DLDataType") //
        .def(py::init([](DLDataTypeCode code, uint8_t bits, uint16_t lanes) {
                 auto ctr = std::make_unique<DLDataType>();
                 ctr->code = static_cast<uint8_t>(code);
                 ctr->bits = bits;
                 ctr->lanes = lanes;
                 return ctr;
             }),
             doc::DLDataType::doc_DLDataType, py::call_guard<py::gil_scoped_release>())
        .def_readonly("code", &DLDataType::code, doc::DLDataType::doc_code, py::call_guard<py::gil_scoped_release>()) //
        .def_readonly("bits", &DLDataType::bits, doc::DLDataType::doc_bits, py::call_guard<py::gil_scoped_release>()) //
        .def_readonly("lanes", &DLDataType::lanes, doc::DLDataType::doc_lanes, py::call_guard<py::gil_scoped_release>()) //
        .def(
            "__repr__",
            [](const DLDataType& dtype) {
                return fmt::format("<cucim.DLDataType code:{}({}) bits:{} lanes:{}>", g_dldata_typecode.at(dtype.code),
                                   dtype.code, dtype.bits, dtype.lanes);
            },
            py::call_guard<py::gil_scoped_release>());

    py::class_<CuImage, std::shared_ptr<CuImage>>(m, "CuImage", py::dynamic_attr()) //
        .def(py::init<const std::string&>(), doc::CuImage::doc_CuImage, py::call_guard<py::gil_scoped_release>(), //
             py::arg("path")) //
        .def_static("cache", &py_cache, doc::CuImage::doc_cache, py::call_guard<py::gil_scoped_release>(), //
                    py::arg("type") = py::none()) //
        .def_static("profiler", &py_profiler, doc::CuImage::doc_profiler, py::call_guard<py::gil_scoped_release>()) //
        .def_property_readonly_static("is_trace_enabled", &py_is_trace_enabled, doc::CuImage::doc_is_trace_enabled,
                                      py::call_guard<py::gil_scoped_release>()) //);
        // Do not release GIL
        .def_static("_set_array_interface", &_set_array_interface, doc::CuImage::doc__set_array_interface, //
                    py::arg("cuimg") = py::none()) //
        .def_property("path", &CuImage::path, nullptr, doc::CuImage::doc_path, py::call_guard<py::gil_scoped_release>()) //
        .def_property("is_loaded", &CuImage::is_loaded, nullptr, doc::CuImage::doc_is_loaded,
                      py::call_guard<py::gil_scoped_release>()) //
        .def_property(
            "device", &CuImage::device, nullptr, doc::CuImage::doc_device, py::call_guard<py::gil_scoped_release>()) //
        .def_property("raw_metadata", &CuImage::raw_metadata, nullptr, doc::CuImage::doc_raw_metadata,
                      py::call_guard<py::gil_scoped_release>()) //
        .def_property(
            "metadata", &py_metadata, nullptr, doc::CuImage::doc_metadata, py::call_guard<py::gil_scoped_release>()) //
        .def_property("ndim", &CuImage::ndim, nullptr, doc::CuImage::doc_ndim, py::call_guard<py::gil_scoped_release>()) //
        .def_property("dims", &CuImage::dims, nullptr, doc::CuImage::doc_dims, py::call_guard<py::gil_scoped_release>()) //
        .def_property(
            "shape", &CuImage::shape, nullptr, doc::CuImage::doc_shape, py::call_guard<py::gil_scoped_release>()) //
        .def("size", &CuImage::size, doc::CuImage::doc_size, py::call_guard<py::gil_scoped_release>(), //
             py::arg("dim_order") = "" //
             ) //
        .def_property(
            "dtype", &CuImage::dtype, nullptr, doc::CuImage::doc_dtype, py::call_guard<py::gil_scoped_release>()) //
        .def_property("channel_names", &CuImage::channel_names, nullptr, doc::CuImage::doc_channel_names,
                      py::call_guard<py::gil_scoped_release>()) //
        .def("spacing", &CuImage::spacing, doc::CuImage::doc_spacing, py::call_guard<py::gil_scoped_release>(), //
             py::arg("dim_order") = "" //
             ) //
        .def("spacing_units", &CuImage::spacing_units, doc::CuImage::doc_spacing_units,
             py::call_guard<py::gil_scoped_release>(), //
             py::arg("dim_order") = "" //
             ) //
        .def_property(
            "origin", &CuImage::origin, nullptr, doc::CuImage::doc_origin, py::call_guard<py::gil_scoped_release>()) //
        .def_property("direction", &CuImage::direction, nullptr, doc::CuImage::doc_direction,
                      py::call_guard<py::gil_scoped_release>()) //
        .def_property("coord_sys", &CuImage::coord_sys, nullptr, doc::CuImage::doc_coord_sys,
                      py::call_guard<py::gil_scoped_release>()) //
        .def_property("resolutions", &py_resolutions, nullptr, doc::CuImage::doc_resolutions,
                      py::call_guard<py::gil_scoped_release>()) //
        .def("read_region", &py_read_region, doc::CuImage::doc_read_region, py::call_guard<py::gil_scoped_release>(), //
             py::arg("location") = py::tuple{}, //
             py::arg("size") = py::tuple{}, //
             py::arg("level") = 0, //
             py::arg("num_workers") = 0, //
             py::arg("batch_size") = 1, //
             py::arg("drop_last") = py::bool_(false), //
             py::arg("prefetch_factor") = 2, //
             py::arg("shuffle") = py::bool_(false), //
             py::arg("seed") = py::int_(0), //
             py::arg("device") = io::Device(), //
             py::arg("buf") = py::none(), //
             py::arg("shm_name") = "") //
        .def_property("associated_images", &CuImage::associated_images, nullptr, doc::CuImage::doc_associated_images,
                      py::call_guard<py::gil_scoped_release>()) //
        .def("associated_image", &py_associated_image, doc::CuImage::doc_associated_image,
             py::call_guard<py::gil_scoped_release>(), //
             py::arg("name") = "", //
             py::arg("device") = io::Device()) //
        .def("save", &CuImage::save, doc::CuImage::doc_save, py::call_guard<py::gil_scoped_release>()) //
        .def("close", &CuImage::close, doc::CuImage::doc_close, py::call_guard<py::gil_scoped_release>()) //
        .def("__bool__", &CuImage::operator bool, py::call_guard<py::gil_scoped_release>()) //
        .def(
            "__iter__", //
            [](const std::shared_ptr<CuImage>& cuimg) { //
                return cuimg->begin(); //
            }, //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__enter__",
            [](const std::shared_ptr<CuImage>& cuimg) { //
                return cuimg; //
            }, //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__exit__",
            [](const std::shared_ptr<CuImage>& cuimg, const py::object& type, const py::object& value,
               const py::object& traceback) { //
                cuimg->close(); //
            }, //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__repr__", //
            [](const CuImage& cuimg) { //
                return fmt::format("<cucim.CuImage path:{}>", cuimg.path());
            },
            py::call_guard<py::gil_scoped_release>());

    py::class_<CuImageIterator<CuImage>>(m, "CuImageIterator") //
        .def(py::init<std::shared_ptr<CuImage>, bool>(), doc::CuImageIterator::doc_CuImageIterator,
             py::arg("cuimg"), //
             py::arg("ending") = false, py::call_guard<py::gil_scoped_release>())
        .def(
            "__len__",
            [](const CuImageIterator<CuImage>& it) { //
                return it.size(); //
            }, //
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__iter__", //
            [](CuImageIterator<CuImage>& it) { //
                return CuImageIterator<CuImage>(it); //
            }, //
            py::call_guard<py::gil_scoped_release>())
        .def("__next__", &py_cuimage_iterator_next, py::call_guard<py::gil_scoped_release>())
        .def(
            "__repr__", //
            [](CuImageIterator<CuImage>& it) { //
                return fmt::format("<cucim.CuImageIterator index:{}>", it.index());
            },
            py::call_guard<py::gil_scoped_release>());

    // We can use `"cpu"` instead of `Device("cpu")`
    py::implicitly_convertible<const char*, io::Device>();
}

std::string get_plugin_root()
{
    return CuImage::get_framework()->get_plugin_root();
}

void set_plugin_root(std::string path)
{
    CuImage::get_framework()->set_plugin_root(path.c_str());
}

template <typename PT, typename T>
pybind11::tuple vector2pytuple(const std::vector<T>& vec)
{
    py::tuple result(vec.size());
    std::vector<pybind11::object> args;
    args.reserve(vec.size());
    for (auto& arg_value : vec)
    {
        args.emplace_back(pybind11::reinterpret_steal<pybind11::object>(pybind11::detail::make_caster<PT>::cast(
            std::forward<PT>(arg_value), pybind11::return_value_policy::automatic_reference, nullptr)));
    }
    int counter = 0;
    for (auto& arg_value : args)
    {
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    }
    return result;
}

std::shared_ptr<cucim::cache::ImageCache> py_cache(const py::object& type, const py::kwargs& kwargs)
{
    if (py::isinstance<py::str>(type))
    {
        std::string ctype = std::string(py::cast<py::str>(type));

        cucim::cache::CacheType cache_type = cucim::cache::lookup_cache_type(ctype);
        // Copy default cache config to local
        cucim::cache::ImageCacheConfig config = cucim::CuImage::get_config()->cache();
        config.type = cache_type;

        if (kwargs.contains("memory_capacity"))
        {
            config.memory_capacity = py::cast<uint32_t>(kwargs["memory_capacity"]);
        }
        if (kwargs.contains("capacity"))
        {
            config.capacity = py::cast<uint32_t>(kwargs["capacity"]);
        }
        else
        {
            // Update capacity depends on memory_capacity.
            config.capacity = cucim::cache::calc_default_cache_capacity(cucim::cache::kOneMiB * config.memory_capacity);
        }
        if (kwargs.contains("mutex_pool_capacity"))
        {
            config.mutex_pool_capacity = py::cast<uint32_t>(kwargs["mutex_pool_capacity"]);
        }
        if (kwargs.contains("list_padding"))
        {
            config.list_padding = py::cast<uint32_t>(kwargs["list_padding"]);
        }
        if (kwargs.contains("extra_shared_memory_size"))
        {
            config.extra_shared_memory_size = py::cast<uint32_t>(kwargs["extra_shared_memory_size"]);
        }
        if (kwargs.contains("record_stat"))
        {
            config.record_stat = py::cast<bool>(kwargs["record_stat"]);
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

std::shared_ptr<cucim::profiler::Profiler> py_profiler(const py::kwargs& kwargs)
{
    if (kwargs.empty())
    {
        return CuImage::profiler();
    }
    else
    {
        // Copy default profiler config to local
        cucim::profiler::ProfilerConfig config = cucim::CuImage::get_config()->profiler();

        if (kwargs.contains("trace"))
        {
            config.trace = py::cast<bool>(kwargs["trace"]);
        }
        return CuImage::profiler(config);
    }
}

bool py_is_trace_enabled(py::object /* self */)
{
    return CuImage::is_trace_enabled();
}

json py_metadata(const CuImage& cuimg)
{
    auto metadata = cuimg.metadata();
    auto json_obj = json::parse(metadata.empty() ? "{}" : metadata);

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
    return json_obj;
}

py::dict py_resolutions(const CuImage& cuimg)
{
    const auto& resolutions = cuimg.resolutions();
    auto level_count = resolutions.level_count();
    if (resolutions.level_count() == 0)
    {
        return py::dict{
            "level_count"_a = pybind11::int_(0), //
            "level_dimensions"_a = pybind11::tuple(), //
            "level_downsamples"_a = pybind11::tuple(), //
            "level_tile_sizes"_a = pybind11::tuple() //
        };
    }

    std::vector<py::tuple> level_dimensions_vec;
    level_dimensions_vec.reserve(level_count);
    std::vector<py::tuple> level_tile_sizes_vec;
    level_tile_sizes_vec.reserve(level_count);
    for (int level = 0; level < level_count; ++level)
    {
        level_dimensions_vec.emplace_back(vector2pytuple<pybind11::int_>(resolutions.level_dimension(level)));
        level_tile_sizes_vec.emplace_back(vector2pytuple<pybind11::int_>(resolutions.level_tile_size(level)));
    }

    py::tuple level_dimensions = vector2pytuple<const pybind11::tuple&>(level_dimensions_vec);
    py::tuple level_downsamples = vector2pytuple<pybind11::float_>(resolutions.level_downsamples());
    py::tuple level_tile_sizes = vector2pytuple<const pybind11::tuple&>(level_tile_sizes_vec);

    return py::dict{
        "level_count"_a = pybind11::int_(level_count), //
        "level_dimensions"_a = level_dimensions, //
        "level_downsamples"_a = level_downsamples, //
        "level_tile_sizes"_a = level_tile_sizes //
    };
}


py::object py_read_region(const CuImage& cuimg,
                          const py::iterable& location,
                          std::vector<int64_t>&& size,
                          int16_t level,
                          uint32_t num_workers,
                          uint32_t batch_size,
                          bool drop_last,
                          uint32_t prefetch_factor,
                          bool shuffle,
                          uint64_t seed,
                          const io::Device& device,
                          const py::object& buf,
                          const std::string& shm_name,
                          const py::kwargs& kwargs)
{
    if (!size.empty() && size.size() != 2)
    {
        throw std::runtime_error("size (patch size) should be 2!");
    }

    cucim::DimIndices indices;
    std::vector<int64_t> locations;
    {
        py::gil_scoped_acquire scope_guard;

        auto arr = pybind11::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(location);
        if (arr) // fast copy
        {
            py::buffer_info buf = arr.request();
            int64_t* data_array = static_cast<int64_t*>(buf.ptr);
            ssize_t data_size = buf.size;
            locations.reserve(data_size);
            locations.insert(locations.end(), &data_array[0], &data_array[data_size]);
        }
        else
        {
            auto iter = py::iter(location);
            while (iter != py::iterator::sentinel())
            {
                if (py::isinstance<py::iterable>(*iter))
                {
                    auto iter2 = py::iter(*iter);
                    while (iter2 != py::iterator::sentinel())
                    {
                        locations.emplace_back(py::cast<int64_t>(*iter2));
                        ++iter2;
                    }
                }
                else
                {
                    locations.emplace_back(py::cast<int64_t>(*iter));
                }
                ++iter;
            }
        }
    }

    if (kwargs)
    {
        std::vector<std::pair<char, int64_t>> indices_args;

        {
            py::gil_scoped_acquire scope_guard;

            for (auto item : kwargs)
            {
                auto key = std::string(py::str(item.first));
                auto value = py::cast<int>(item.second);

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
        }
        indices = cucim::DimIndices(indices_args);
    }
    else
    {
        indices = cucim::DimIndices{};
    }

    auto region_ptr = std::make_shared<cucim::CuImage>(
        std::move(cuimg.read_region(std::move(locations), std::move(size), level, num_workers, batch_size, drop_last,
                                    prefetch_factor, shuffle, seed, indices, device, nullptr, "")));
    auto loader = region_ptr->loader();
    if (batch_size > 1 || (loader && loader->size() > 1))
    {
        auto iter_ptr = region_ptr->begin();

        py::gil_scoped_acquire scope_guard;

        py::object iter = py::cast(iter_ptr);

        return iter;
    }
    else
    {
        py::gil_scoped_acquire scope_guard;

        py::object region = py::cast(region_ptr);

        // Add `__array_inteface__` or `__cuda_array_interface__` in runtime.
        _set_array_interface(region);

        return region;
    }
}

py::object py_associated_image(const CuImage& cuimg, const std::string& name, const io::Device& device)
{
    auto image_ptr = std::make_shared<cucim::CuImage>(cuimg.associated_image(name, device));

    {
        py::gil_scoped_acquire scope_guard;

        py::object image = py::cast(image_ptr);

        // Add `__array_interace__` or `__cuda_array_interface__` in runtime.
        _set_array_interface(image);

        return image;
    }
}

py::object py_cuimage_iterator_next(CuImageIterator<CuImage>& it)
{
    bool stop_iteration = (it.index() == it.size());

    // Get the next batch of images.
    ++it;

    auto cuimg = *it;
    memory::DLTContainer container = cuimg->container();
    DLTensor* tensor = static_cast<DLTensor*>(container);
    cucim::loader::ThreadBatchDataLoader* loader = cuimg->loader();

    {
        py::gil_scoped_acquire scope_guard;
        py::object cuimg_obj = py::cast(cuimg);
        if (loader)
        {
            _set_array_interface(cuimg_obj);
        }
        if (stop_iteration)
        {
            throw py::stop_iteration();
        }
        return cuimg_obj;
    }
}

void _set_array_interface(const py::object& cuimg_obj)
{
    const auto& cuimg = cuimg_obj.cast<const CuImage&>();

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
        const char* type_str = container.numpy_dtype();
        py::str typestr = py::str(type_str);

        py::tuple data = pybind11::make_tuple(py::int_(reinterpret_cast<uint64_t>(tensor->data)), py::bool_(false));
        py::list descr;
        descr.append(py::make_tuple(""_s, typestr));

        py::tuple shape = vector2pytuple<pybind11::int_>(cuimg.shape());

        // Depending on container's memory type, expose either array_interface or cuda_array_interface
        switch (tensor->ctx.device_type)
        {
        case kDLCPU: {
            // Reference: https://numpy.org/doc/stable/reference/arrays.interface.html
            cuimg_obj.attr("__array_interface__") =
                py::dict{ "data"_a = data,       "strides"_a = py::none(), "descr"_a = descr,
                          "typestr"_a = typestr, "shape"_a = shape,        "version"_a = py::int_(3) };
        }
        break;
        case kDLGPU: {
            // Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
            cuimg_obj.attr("__cuda_array_interface__") =
                py::dict{ "data"_a = data,   "strides"_a = py::none(),  "descr"_a = descr,     "typestr"_a = typestr,
                          "shape"_a = shape, "version"_a = py::int_(3), "mask"_a = py::none(), "stream"_a = 1 };
        }
        break;
        default:
            break;
        }
    }
    else
    {
        switch (tensor->ctx.device_type)
        {
        case kDLCPU: {
            if (py::hasattr(cuimg_obj, "__array_interface__"))
            {
                py::delattr(cuimg_obj, "__array_interface__");
            }
        }
        break;
        case kDLGPU: {
            if (py::hasattr(cuimg_obj, "__cuda_array_interface__"))
            {
                py::delattr(cuimg_obj, "__cuda_array_interface__");
            }
        }
        break;
        default:
            break;
        }
    }
}

} // namespace cucim
