# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cython: always_allow_keywords=True
# cython: binding=True
# cython: embedsignature=True
# cython: language_level=3

from enum import IntEnum
import json
import sys
import types

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "binding.hpp" namespace "cucim::python":
    string cpp_library_version "cucim::python::library_version"() except + nogil

    cdef cppclass NativeCuImage "cucim::CuImage":
        pass

    cdef cppclass NativeImageCache "cucim::cache::ImageCache":
        pass

    cdef cppclass NativeProfiler "cucim::profiler::Profiler":
        pass

    cdef cppclass NativeCuFileDriver "cucim::filesystem::CuFileDriver":
        pass

    cdef cppclass DeviceData:
        int16_t type
        int16_t index
        string name

    cdef cppclass DTypeData:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    cdef cppclass ResolutionData:
        uint16_t level_count
        vector[vector[int64_t]] level_dimensions
        vector[float] level_downsamples
        vector[vector[uint32_t]] level_tile_sizes

    cdef cppclass ArrayInterfaceData:
        cbool has_tensor
        cbool has_data
        uintptr_t data
        int device_type
        string typestr
        vector[int64_t] shape

    cdef cppclass CacheConfigData:
        int type
        string type_name
        uint32_t memory_capacity
        uint32_t capacity
        uint32_t mutex_pool_capacity
        uint32_t list_padding
        uint32_t extra_shared_memory_size
        cbool record_stat

    cdef cppclass NativeImageIterator "cucim::python::ImageIterator":
        NativeImageIterator(shared_ptr[NativeCuImage], cbool) except +
        int64_t index() except + nogil
        uint64_t size() except + nogil
        shared_ptr[NativeCuImage] next() except + nogil

    string cpp_get_plugin_root "cucim::python::get_plugin_root"() except + nogil
    void cpp_set_plugin_root "cucim::python::set_plugin_root"(const string&) except + nogil
    DeviceData cpp_parse_device "cucim::python::parse_device"(const string&) except + nogil

    shared_ptr[NativeCuImage] cpp_make_image "cucim::python::make_image"(const string&) except + nogil
    string cpp_image_path "cucim::python::image_path"(const shared_ptr[NativeCuImage]&) except + nogil
    cbool cpp_image_is_loaded "cucim::python::image_is_loaded"(const shared_ptr[NativeCuImage]&) except + nogil
    DeviceData cpp_image_device "cucim::python::image_device"(const shared_ptr[NativeCuImage]&) except + nogil
    string cpp_image_raw_metadata "cucim::python::image_raw_metadata"(const shared_ptr[NativeCuImage]&) except + nogil
    string cpp_image_metadata "cucim::python::image_metadata"(const shared_ptr[NativeCuImage]&) except + nogil
    uint16_t cpp_image_ndim "cucim::python::image_ndim"(const shared_ptr[NativeCuImage]&) except + nogil
    string cpp_image_dims "cucim::python::image_dims"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[int64_t] cpp_image_shape "cucim::python::image_shape"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[int64_t] cpp_image_size "cucim::python::image_size"(const shared_ptr[NativeCuImage]&, const string&) except + nogil
    DTypeData cpp_image_dtype "cucim::python::image_dtype"(const shared_ptr[NativeCuImage]&) except + nogil
    string cpp_image_typestr "cucim::python::image_typestr"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[string] cpp_image_channel_names "cucim::python::image_channel_names"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[float] cpp_image_spacing "cucim::python::image_spacing"(const shared_ptr[NativeCuImage]&, const string&) except + nogil
    vector[string] cpp_image_spacing_units "cucim::python::image_spacing_units"(const shared_ptr[NativeCuImage]&, const string&) except + nogil
    vector[float] cpp_image_origin "cucim::python::image_origin"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[vector[float]] cpp_image_direction "cucim::python::image_direction"(const shared_ptr[NativeCuImage]&) except + nogil
    string cpp_image_coord_sys "cucim::python::image_coord_sys"(const shared_ptr[NativeCuImage]&) except + nogil
    ResolutionData cpp_image_resolutions "cucim::python::image_resolutions"(const shared_ptr[NativeCuImage]&) except + nogil
    vector[string] cpp_image_associated_images "cucim::python::image_associated_images"(const shared_ptr[NativeCuImage]&) except + nogil
    shared_ptr[NativeCuImage] cpp_image_associated_image "cucim::python::image_associated_image"(
        const shared_ptr[NativeCuImage]&, const string&, const string&
    ) except + nogil
    shared_ptr[NativeCuImage] cpp_image_read_region "cucim::python::image_read_region"(
        const shared_ptr[NativeCuImage]&,
        vector[int64_t],
        vector[int64_t],
        int16_t,
        uint32_t,
        uint32_t,
        cbool,
        uint32_t,
        cbool,
        uint64_t,
        const vector[int]&,
        const vector[int64_t]&,
        const string&,
    ) except + nogil
    cbool cpp_image_has_multiple_batches "cucim::python::image_has_multiple_batches"(
        const shared_ptr[NativeCuImage]&, uint32_t
    ) except + nogil
    ArrayInterfaceData cpp_image_array_interface "cucim::python::image_array_interface"(
        const shared_ptr[NativeCuImage]&
    ) except + nogil
    void cpp_image_save "cucim::python::image_save"(const shared_ptr[NativeCuImage]&, const string&) except + nogil
    void cpp_image_close "cucim::python::image_close"(const shared_ptr[NativeCuImage]&) except + nogil
    cbool cpp_image_bool "cucim::python::image_bool"(const shared_ptr[NativeCuImage]&) except + nogil
    cbool cpp_image_is_trace_enabled "cucim::python::image_is_trace_enabled"() except + nogil

    CacheConfigData cpp_cache_default_config "cucim::python::cache_default_config"() except + nogil
    int cpp_cache_type_from_name "cucim::python::cache_type_from_name"(const string&) except + nogil
    shared_ptr[NativeImageCache] cpp_cache_default "cucim::python::cache_default"() except + nogil
    shared_ptr[NativeImageCache] cpp_cache_create "cucim::python::cache_create"(const CacheConfigData&) except + nogil
    CacheConfigData cpp_cache_config "cucim::python::cache_config"(const shared_ptr[NativeImageCache]&) except + nogil
    int cpp_cache_type "cucim::python::cache_type"(const shared_ptr[NativeImageCache]&) except + nogil
    uint32_t cpp_cache_size "cucim::python::cache_size"(const shared_ptr[NativeImageCache]&) except + nogil
    uint64_t cpp_cache_memory_size "cucim::python::cache_memory_size"(const shared_ptr[NativeImageCache]&) except + nogil
    uint32_t cpp_cache_capacity "cucim::python::cache_capacity"(const shared_ptr[NativeImageCache]&) except + nogil
    uint64_t cpp_cache_memory_capacity "cucim::python::cache_memory_capacity"(const shared_ptr[NativeImageCache]&) except + nogil
    uint64_t cpp_cache_free_memory "cucim::python::cache_free_memory"(const shared_ptr[NativeImageCache]&) except + nogil
    cbool cpp_cache_record_get "cucim::python::cache_record_get"(const shared_ptr[NativeImageCache]&) except + nogil
    void cpp_cache_record_set "cucim::python::cache_record_set"(const shared_ptr[NativeImageCache]&, cbool) except + nogil
    uint64_t cpp_cache_hit_count "cucim::python::cache_hit_count"(const shared_ptr[NativeImageCache]&) except + nogil
    uint64_t cpp_cache_miss_count "cucim::python::cache_miss_count"(const shared_ptr[NativeImageCache]&) except + nogil
    void cpp_cache_reserve "cucim::python::cache_reserve"(
        const shared_ptr[NativeImageCache]&, uint32_t, cbool, uint32_t
    ) except + nogil
    uint32_t cpp_preferred_memory_capacity_for_image "cucim::python::preferred_memory_capacity_for_image"(
        const shared_ptr[NativeCuImage]&, const vector[uint32_t]&, uint32_t
    ) except + nogil
    uint32_t cpp_preferred_memory_capacity_explicit "cucim::python::preferred_memory_capacity_explicit"(
        const vector[uint64_t]&,
        const vector[uint32_t]&,
        const vector[uint32_t]&,
        uint32_t,
    ) except + nogil

    shared_ptr[NativeProfiler] cpp_profiler_default "cucim::python::profiler_default"() except + nogil
    shared_ptr[NativeProfiler] cpp_profiler_create "cucim::python::profiler_create"(cbool) except + nogil
    cbool cpp_profiler_trace_get "cucim::python::profiler_trace_get"(const shared_ptr[NativeProfiler]&) except + nogil
    void cpp_profiler_trace_set "cucim::python::profiler_trace_set"(
        const shared_ptr[NativeProfiler]&, cbool
    ) except + nogil

    shared_ptr[NativeCuFileDriver] cpp_file_driver_from_fd "cucim::python::file_driver_from_fd"(
        int, cbool, cbool, const string&
    ) except + nogil
    shared_ptr[NativeCuFileDriver] cpp_filesystem_open "cucim::python::filesystem_open"(
        const string&, const string&, uint32_t
    ) except + nogil
    cbool cpp_filesystem_is_gds_available "cucim::python::filesystem_is_gds_available"() except + nogil
    cbool cpp_filesystem_close "cucim::python::filesystem_close"(
        const shared_ptr[NativeCuFileDriver]&
    ) except + nogil
    cbool cpp_filesystem_discard_page_cache "cucim::python::filesystem_discard_page_cache"(
        const string&
    ) except + nogil
    string cpp_file_driver_path "cucim::python::file_driver_path"(
        const shared_ptr[NativeCuFileDriver]&
    ) except + nogil
    int64_t cpp_file_driver_pread "cucim::python::file_driver_pread"(
        const shared_ptr[NativeCuFileDriver]&,
        uintptr_t,
        uint64_t,
        int64_t,
        int64_t,
    ) except + nogil
    int64_t cpp_file_driver_pwrite "cucim::python::file_driver_pwrite"(
        const shared_ptr[NativeCuFileDriver]&,
        uintptr_t,
        uint64_t,
        int64_t,
        int64_t,
    ) except + nogil


cdef class _CuImageHandle:
    cdef shared_ptr[NativeCuImage] value


cdef class _ImageCacheHandle:
    cdef shared_ptr[NativeImageCache] value


cdef class _ProfilerHandle:
    cdef shared_ptr[NativeProfiler] value


cdef class _CuFileDriverHandle:
    cdef shared_ptr[NativeCuFileDriver] value


cdef class _ImageIteratorHandle:
    cdef NativeImageIterator* value

    def __cinit__(self):
        self.value = NULL

    def __dealloc__(self):
        if self.value != NULL:
            del self.value


cdef string _as_cpp_string(object value):
    cdef bytes encoded = str(value).encode()
    return encoded


cdef str _as_python_string(string value):
    return (<bytes>value).decode()


cdef list _int64_vector_to_list(vector[int64_t] values):
    cdef size_t index
    cdef list result = []
    for index in range(values.size()):
        result.append(values[index])
    return result


cdef list _uint32_vector_to_list(vector[uint32_t] values):
    cdef size_t index
    cdef list result = []
    for index in range(values.size()):
        result.append(values[index])
    return result


cdef list _float_vector_to_list(vector[float] values):
    cdef size_t index
    cdef list result = []
    for index in range(values.size()):
        result.append(values[index])
    return result


cdef list _string_vector_to_list(vector[string] values):
    cdef size_t index
    cdef list result = []
    for index in range(values.size()):
        result.append(_as_python_string(values[index]))
    return result


cdef vector[int64_t] _to_int64_vector(object values):
    cdef vector[int64_t] result
    for value in values:
        result.push_back(value)
    return result


cdef vector[uint32_t] _to_uint32_vector(object values):
    cdef vector[uint32_t] result
    if values is not None:
        for value in values:
            result.push_back(value)
    return result


cdef vector[uint64_t] _to_uint64_vector(object values):
    cdef vector[uint64_t] result
    if values is not None:
        for value in values:
            result.push_back(value)
    return result


cdef list _flatten_location(object location):
    cdef list result = []
    for value in location:
        try:
            nested = iter(value)
        except TypeError:
            result.append(value)
        else:
            result.extend(nested)
    return result


class DLDataTypeCode(IntEnum):
    DLInt = 0
    DLUInt = 1
    DLFloat = 2
    DLBfloat = 4


class DLDataType:
    def __init__(self, code, bits, lanes):
        self.code = DLDataTypeCode(code)
        self.bits = bits
        self.lanes = lanes

    def __eq__(self, other):
        if not isinstance(other, DLDataType):
            return NotImplemented
        return (
            self.code == other.code
            and self.bits == other.bits
            and self.lanes == other.lanes
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __repr__(self):
        return (
            f"<cucim.clara.DLDataType code:{self.code.name}({int(self.code)}) "
            f"bits:{self.bits} lanes:{self.lanes}>"
        )


class DeviceType(IntEnum):
    CPU = 1
    CUDA = 2
    CUDAHost = 3
    CUDAManaged = 13
    CPUShared = 101
    CUDAShared = 102


class Device:
    def __init__(self, device_name="cpu"):
        cdef string name = _as_cpp_string(device_name)
        cdef DeviceData data
        with nogil:
            data = cpp_parse_device(name)
        self._name = _as_python_string(data.name)
        self._type = DeviceType(data.type)
        self._index = data.index

    @staticmethod
    def parse_type(device_name):
        return Device(device_name).type

    @property
    def type(self):
        return self._type

    @property
    def index(self):
        return self._index

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name


class CacheType(IntEnum):
    NoCache = 0
    PerProcess = 1
    SharedMemory = 2


class FileHandleType(IntEnum):
    Unknown = 0
    Posix = 1
    PosixODirect = 2
    MemoryMapped = 4
    GPUDirect = 8


cdef _CuImageHandle _new_image_handle(shared_ptr[NativeCuImage] value):
    cdef _CuImageHandle handle = _CuImageHandle()
    handle.value = value
    return handle


cdef _ImageCacheHandle _new_cache_handle(shared_ptr[NativeImageCache] value):
    cdef _ImageCacheHandle handle = _ImageCacheHandle()
    handle.value = value
    return handle


cdef _ProfilerHandle _new_profiler_handle(shared_ptr[NativeProfiler] value):
    cdef _ProfilerHandle handle = _ProfilerHandle()
    handle.value = value
    return handle


cdef _CuFileDriverHandle _new_file_handle(shared_ptr[NativeCuFileDriver] value):
    cdef _CuFileDriverHandle handle = _CuFileDriverHandle()
    handle.value = value
    return handle


cdef object _wrap_image(shared_ptr[NativeCuImage] value):
    result = CuImage.__new__(CuImage)
    result._handle = _new_image_handle(value)
    return result


cdef object _wrap_cache(shared_ptr[NativeImageCache] value):
    result = ImageCache.__new__(ImageCache)
    result._handle = _new_cache_handle(value)
    return result


cdef object _wrap_profiler(shared_ptr[NativeProfiler] value):
    result = Profiler.__new__(Profiler)
    result._handle = _new_profiler_handle(value)
    return result


cdef object _wrap_file(shared_ptr[NativeCuFileDriver] value):
    result = CuFileDriver.__new__(CuFileDriver)
    result._handle = _new_file_handle(value)
    return result


class _CuImageMeta(type):
    @property
    def is_trace_enabled(cls):
        cdef cbool enabled
        with nogil:
            enabled = cpp_image_is_trace_enabled()
        return bool(enabled)


class CuImage(metaclass=_CuImageMeta):
    def __init__(self, path):
        cdef string cpp_path = _as_cpp_string(path)
        cdef shared_ptr[NativeCuImage] value
        with nogil:
            value = cpp_make_image(cpp_path)
        self._handle = _new_image_handle(value)

    @staticmethod
    def cache(type=None, **kwargs):
        cdef CacheConfigData config
        cdef string type_name
        cdef shared_ptr[NativeImageCache] value
        if type is None:
            with nogil:
                value = cpp_cache_default()
            return _wrap_cache(value)
        if not isinstance(type, str):
            raise ValueError(
                "The first argument should be one of "
                "['nocache', 'per_process', 'shared_memory']."
            )

        type_name = _as_cpp_string(type)
        with nogil:
            config = cpp_cache_default_config()
            config.type = cpp_cache_type_from_name(type_name)
        if "memory_capacity" in kwargs:
            config.memory_capacity = kwargs["memory_capacity"]
        if "capacity" in kwargs:
            config.capacity = kwargs["capacity"]
        else:
            config.capacity = (
                <uint64_t>config.memory_capacity * 1024 * 1024
            ) // (256 * 256 * 3)
        if "mutex_pool_capacity" in kwargs:
            config.mutex_pool_capacity = kwargs["mutex_pool_capacity"]
        if "list_padding" in kwargs:
            config.list_padding = kwargs["list_padding"]
        if "extra_shared_memory_size" in kwargs:
            config.extra_shared_memory_size = kwargs[
                "extra_shared_memory_size"
            ]
        if "record_stat" in kwargs:
            config.record_stat = kwargs["record_stat"]
        with nogil:
            value = cpp_cache_create(config)
        return _wrap_cache(value)

    @staticmethod
    def profiler(**kwargs):
        cdef shared_ptr[NativeProfiler] value
        cdef cbool trace
        if not kwargs:
            with nogil:
                value = cpp_profiler_default()
        else:
            trace = bool(kwargs.get("trace", False))
            with nogil:
                value = cpp_profiler_create(trace)
        return _wrap_profiler(value)

    @staticmethod
    def _set_array_interface(cuimg=None):
        if not isinstance(cuimg, CuImage):
            raise TypeError("cuimg must be a CuImage")
        cuimg._refresh_array_interface()

    @property
    def path(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_path(handle.value)
        return _as_python_string(value)

    @property
    def is_loaded(self):
        cdef _CuImageHandle handle = self._handle
        cdef cbool value
        with nogil:
            value = cpp_image_is_loaded(handle.value)
        return bool(value)

    @property
    def device(self):
        cdef _CuImageHandle handle = self._handle
        cdef DeviceData value
        with nogil:
            value = cpp_image_device(handle.value)
        return Device(_as_python_string(value.name))

    @property
    def raw_metadata(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_raw_metadata(handle.value)
        return _as_python_string(value)

    @property
    def metadata(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_metadata(handle.value)
        metadata = json.loads(_as_python_string(value) or "{}")
        metadata["cucim"] = {
            "path": self.path,
            "ndim": self.ndim,
            "dims": self.dims,
            "shape": self.shape,
            "dtype": {
                "code": int(self.dtype.code),
                "bits": self.dtype.bits,
                "lanes": self.dtype.lanes,
            },
            "typestr": self.typestr,
            "channel_names": self.channel_names,
            "spacing": self.spacing(),
            "spacing_units": self.spacing_units(),
            "origin": self.origin,
            "direction": self.direction,
            "coord_sys": self.coord_sys,
            "resolutions": self.resolutions,
            "associated_images": self.associated_images,
        }
        return metadata

    @property
    def ndim(self):
        cdef _CuImageHandle handle = self._handle
        cdef uint16_t value
        with nogil:
            value = cpp_image_ndim(handle.value)
        return value

    @property
    def dims(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_dims(handle.value)
        return _as_python_string(value)

    @property
    def shape(self):
        cdef _CuImageHandle handle = self._handle
        cdef vector[int64_t] value
        with nogil:
            value = cpp_image_shape(handle.value)
        return _int64_vector_to_list(value)

    def size(self, dim_order=""):
        cdef _CuImageHandle handle = self._handle
        cdef string order = _as_cpp_string(dim_order)
        cdef vector[int64_t] value
        with nogil:
            value = cpp_image_size(handle.value, order)
        return _int64_vector_to_list(value)

    @property
    def dtype(self):
        cdef _CuImageHandle handle = self._handle
        cdef DTypeData value
        with nogil:
            value = cpp_image_dtype(handle.value)
        return DLDataType(value.code, value.bits, value.lanes)

    @property
    def typestr(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_typestr(handle.value)
        return _as_python_string(value)

    @property
    def channel_names(self):
        cdef _CuImageHandle handle = self._handle
        cdef vector[string] value
        with nogil:
            value = cpp_image_channel_names(handle.value)
        return _string_vector_to_list(value)

    def spacing(self, dim_order=""):
        cdef _CuImageHandle handle = self._handle
        cdef string order = _as_cpp_string(dim_order)
        cdef vector[float] value
        with nogil:
            value = cpp_image_spacing(handle.value, order)
        return _float_vector_to_list(value)

    def spacing_units(self, dim_order=""):
        cdef _CuImageHandle handle = self._handle
        cdef string order = _as_cpp_string(dim_order)
        cdef vector[string] value
        with nogil:
            value = cpp_image_spacing_units(handle.value, order)
        return _string_vector_to_list(value)

    @property
    def origin(self):
        cdef _CuImageHandle handle = self._handle
        cdef vector[float] value
        with nogil:
            value = cpp_image_origin(handle.value)
        return _float_vector_to_list(value)

    @property
    def direction(self):
        cdef _CuImageHandle handle = self._handle
        cdef vector[vector[float]] value
        cdef size_t index
        with nogil:
            value = cpp_image_direction(handle.value)
        return [
            _float_vector_to_list(value[index])
            for index in range(value.size())
        ]

    @property
    def coord_sys(self):
        cdef _CuImageHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_image_coord_sys(handle.value)
        return _as_python_string(value)

    @property
    def resolutions(self):
        cdef _CuImageHandle handle = self._handle
        cdef ResolutionData value
        cdef size_t index
        cdef list dimensions = []
        cdef list tile_sizes = []
        with nogil:
            value = cpp_image_resolutions(handle.value)
        for index in range(value.level_dimensions.size()):
            dimensions.append(
                tuple(_int64_vector_to_list(value.level_dimensions[index]))
            )
        for index in range(value.level_tile_sizes.size()):
            tile_sizes.append(
                tuple(_uint32_vector_to_list(value.level_tile_sizes[index]))
            )
        return {
            "level_count": value.level_count,
            "level_dimensions": tuple(dimensions),
            "level_downsamples": tuple(
                _float_vector_to_list(value.level_downsamples)
            ),
            "level_tile_sizes": tuple(tile_sizes),
        }

    def read_region(
        self,
        location=(),
        size=(),
        level=0,
        num_workers=0,
        batch_size=1,
        drop_last=False,
        prefetch_factor=2,
        shuffle=False,
        seed=0,
        device="cpu",
        buf=None,
        shm_name="",
        **kwargs,
    ):
        cdef _CuImageHandle handle = self._handle
        cdef vector[int64_t] cpp_location = _to_int64_vector(
            _flatten_location(location)
        )
        cdef vector[int64_t] cpp_size = _to_int64_vector(size)
        cdef vector[int] dim_chars
        cdef vector[int64_t] dim_values
        cdef string device_name = _as_cpp_string(device)
        cdef shared_ptr[NativeCuImage] value
        cdef cbool multiple
        cdef int16_t cpp_level = level
        cdef uint32_t cpp_num_workers = num_workers
        cdef uint32_t cpp_batch_size = batch_size
        cdef cbool cpp_drop_last = drop_last
        cdef uint32_t cpp_prefetch_factor = prefetch_factor
        cdef cbool cpp_shuffle = shuffle
        cdef uint64_t cpp_seed = seed
        del buf, shm_name
        for key, dim_value in kwargs.items():
            key = str(key)
            if len(key) != 1:
                raise ValueError(
                    "Argument name for Dimension should be a single "
                    f"character but '{key}' is used."
                )
            key = key.upper()
            if key < "A" or key > "Z":
                raise ValueError(
                    "Dimension character should be an alphabet but "
                    f"'{key}' is used."
                )
            dim_chars.push_back(ord(key))
            dim_values.push_back(dim_value)
        with nogil:
            value = cpp_image_read_region(
                handle.value,
                cpp_location,
                cpp_size,
                cpp_level,
                cpp_num_workers,
                cpp_batch_size,
                cpp_drop_last,
                cpp_prefetch_factor,
                cpp_shuffle,
                cpp_seed,
                dim_chars,
                dim_values,
                device_name,
            )
            multiple = cpp_image_has_multiple_batches(value, cpp_batch_size)
        image = _wrap_image(value)
        if multiple:
            return CuImageIterator(image)
        image._refresh_array_interface()
        return image

    @property
    def associated_images(self):
        cdef _CuImageHandle handle = self._handle
        cdef vector[string] value
        with nogil:
            value = cpp_image_associated_images(handle.value)
        return set(_string_vector_to_list(value))

    def associated_image(self, name="", device="cpu"):
        cdef _CuImageHandle handle = self._handle
        cdef string cpp_name = _as_cpp_string(name)
        cdef string cpp_device = _as_cpp_string(device)
        cdef shared_ptr[NativeCuImage] value
        with nogil:
            value = cpp_image_associated_image(
                handle.value, cpp_name, cpp_device
            )
        image = _wrap_image(value)
        image._refresh_array_interface()
        return image

    def save(self, file_path):
        cdef _CuImageHandle handle = self._handle
        cdef string cpp_path = _as_cpp_string(file_path)
        with nogil:
            cpp_image_save(handle.value, cpp_path)

    def close(self):
        cdef _CuImageHandle handle = self._handle
        with nogil:
            cpp_image_close(handle.value)

    def __bool__(self):
        cdef _CuImageHandle handle = self._handle
        cdef cbool value
        with nogil:
            value = cpp_image_bool(handle.value)
        return bool(value)

    def __iter__(self):
        return CuImageIterator(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        return f"<cucim.CuImage path:{self.path}>"

    def _refresh_array_interface(self):
        cdef _CuImageHandle handle = self._handle
        cdef ArrayInterfaceData value
        with nogil:
            value = cpp_image_array_interface(handle.value)
        if not value.has_tensor:
            return
        if value.has_data:
            typestr = _as_python_string(value.typestr)
            interface = {
                "data": (value.data, False),
                "strides": None,
                "descr": [("", typestr)],
                "typestr": typestr,
                "shape": tuple(_int64_vector_to_list(value.shape)),
                "version": 3,
            }
            if value.device_type == 1:
                self.__array_interface__ = interface
                self.__dict__.pop("__cuda_array_interface__", None)
            elif value.device_type == 2:
                interface["mask"] = None
                interface["stream"] = 1
                self.__cuda_array_interface__ = interface
                self.__dict__.pop("__array_interface__", None)
        elif value.device_type == 1:
            self.__dict__.pop("__array_interface__", None)
        elif value.device_type == 2:
            self.__dict__.pop("__cuda_array_interface__", None)


class CuImageIterator:
    def __init__(self, cuimg, ending=False):
        if not isinstance(cuimg, CuImage):
            raise TypeError("cuimg must be a CuImage")
        cdef _CuImageHandle image_handle = cuimg._handle
        cdef _ImageIteratorHandle handle = _ImageIteratorHandle()
        handle.value = new NativeImageIterator(image_handle.value, ending)
        self._handle = handle

    def __len__(self):
        cdef _ImageIteratorHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = handle.value.size()
        return value

    def __iter__(self):
        return self

    def __next__(self):
        cdef _ImageIteratorHandle handle = self._handle
        cdef int64_t index
        cdef uint64_t size
        cdef shared_ptr[NativeCuImage] value
        with nogil:
            index = handle.value.index()
            size = handle.value.size()
        if index == size:
            raise StopIteration
        with nogil:
            value = handle.value.next()
        image = _wrap_image(value)
        image._refresh_array_interface()
        return image

    def __repr__(self):
        cdef _ImageIteratorHandle handle = self._handle
        cdef int64_t value
        with nogil:
            value = handle.value.index()
        return f"<cucim.CuImageIterator index:{value}>"


class ImageCache:
    @property
    def type(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef int value
        with nogil:
            value = cpp_cache_type(handle.value)
        return CacheType(value)

    @property
    def config(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef CacheConfigData value
        with nogil:
            value = cpp_cache_config(handle.value)
        return {
            "type": _as_python_string(value.type_name),
            "memory_capacity": value.memory_capacity,
            "capacity": value.capacity,
            "mutex_pool_capacity": value.mutex_pool_capacity,
            "list_padding": value.list_padding,
            "extra_shared_memory_size": value.extra_shared_memory_size,
            "record_stat": bool(value.record_stat),
        }

    @property
    def size(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint32_t value
        with nogil:
            value = cpp_cache_size(handle.value)
        return value

    @property
    def memory_size(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = cpp_cache_memory_size(handle.value)
        return value

    @property
    def capacity(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint32_t value
        with nogil:
            value = cpp_cache_capacity(handle.value)
        return value

    @property
    def memory_capacity(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = cpp_cache_memory_capacity(handle.value)
        return value

    @property
    def free_memory(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = cpp_cache_free_memory(handle.value)
        return value

    def record(self, value=None):
        cdef _ImageCacheHandle handle = self._handle
        cdef cbool result
        cdef cbool cpp_value
        if value is None:
            with nogil:
                result = cpp_cache_record_get(handle.value)
            return bool(result)
        if not isinstance(value, bool):
            raise ValueError(
                "Only 'NoneType' or 'bool' is available for the argument"
            )
        cpp_value = value
        with nogil:
            cpp_cache_record_set(handle.value, cpp_value)
        return value

    @property
    def hit_count(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = cpp_cache_hit_count(handle.value)
        return value

    @property
    def miss_count(self):
        cdef _ImageCacheHandle handle = self._handle
        cdef uint64_t value
        with nogil:
            value = cpp_cache_miss_count(handle.value)
        return value

    def reserve(self, memory_capacity, **kwargs):
        cdef _ImageCacheHandle handle = self._handle
        cdef cbool has_capacity = "capacity" in kwargs
        cdef uint32_t capacity = kwargs.get("capacity", 0)
        cdef uint32_t cpp_memory_capacity = memory_capacity
        with nogil:
            cpp_cache_reserve(
                handle.value, cpp_memory_capacity, has_capacity, capacity
            )


class Profiler:
    @property
    def config(self):
        return {"trace": self.trace()}

    def trace(self, value=None):
        cdef _ProfilerHandle handle = self._handle
        cdef cbool result
        cdef cbool cpp_value
        if value is None:
            with nogil:
                result = cpp_profiler_trace_get(handle.value)
            return bool(result)
        if not isinstance(value, bool):
            raise ValueError(
                "Only 'NoneType' or 'bool' is available for the argument"
            )
        cpp_value = value
        with nogil:
            cpp_profiler_trace_set(handle.value, cpp_value)
        return value


cdef tuple _memory_info(object buffer):
    interface = None
    if hasattr(buffer, "__array_interface__"):
        interface = buffer.__array_interface__
    elif hasattr(buffer, "__cuda_array_interface__"):
        interface = buffer.__cuda_array_interface__
    elif isinstance(buffer, int):
        return buffer, 0, False

    if isinstance(interface, dict):
        data = interface.get("data")
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], getattr(buffer, "nbytes", 0), bool(data[1])
    raise RuntimeError("Cannot Recognize the array object!")


class CuFileDriver:
    def __init__(self, fd, no_gds=False, use_mmap=False, file_path=""):
        cdef string cpp_path = _as_cpp_string(file_path)
        cdef shared_ptr[NativeCuFileDriver] value
        cdef int cpp_fd = fd
        cdef cbool cpp_no_gds = no_gds
        cdef cbool cpp_use_mmap = use_mmap
        with nogil:
            value = cpp_file_driver_from_fd(
                cpp_fd, cpp_no_gds, cpp_use_mmap, cpp_path
            )
        self._handle = _new_file_handle(value)

    def pread(self, buf, count, file_offset, buf_offset=0):
        cdef _CuFileDriverHandle handle = self._handle
        pointer, memory_size, readonly = _memory_info(buf)
        cdef uintptr_t cpp_pointer = pointer
        cdef uint64_t cpp_count = count
        cdef int64_t cpp_file_offset = file_offset
        cdef int64_t cpp_buf_offset = buf_offset
        if readonly:
            raise RuntimeError(
                "The buffer is readonly so cannot be used for pread!"
            )
        if memory_size and count > memory_size:
            raise RuntimeError(
                f"[Error] 'count' ({count}) is larger than the size of the "
                f"array object ({memory_size})!"
            )
        cdef int64_t result
        with nogil:
            result = cpp_file_driver_pread(
                handle.value,
                cpp_pointer,
                cpp_count,
                cpp_file_offset,
                cpp_buf_offset,
            )
        return result

    def pwrite(self, buf, count, file_offset, buf_offset=0):
        cdef _CuFileDriverHandle handle = self._handle
        pointer, memory_size, _ = _memory_info(buf)
        cdef uintptr_t cpp_pointer = pointer
        cdef uint64_t cpp_count = count
        cdef int64_t cpp_file_offset = file_offset
        cdef int64_t cpp_buf_offset = buf_offset
        if memory_size and count > memory_size:
            raise RuntimeError(
                f"[Error] 'count' ({count}) is larger than the size of the "
                f"array object ({memory_size})!"
            )
        cdef int64_t result
        with nogil:
            result = cpp_file_driver_pwrite(
                handle.value,
                cpp_pointer,
                cpp_count,
                cpp_file_offset,
                cpp_buf_offset,
            )
        return result

    def close(self):
        cdef _CuFileDriverHandle handle = self._handle
        cdef cbool result
        with nogil:
            result = cpp_filesystem_close(handle.value)
        return bool(result)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        cdef _CuFileDriverHandle handle = self._handle
        cdef string value
        with nogil:
            value = cpp_file_driver_path(handle.value)
        return (
            "<cucim.clara.filesystem.CuFileDriver path:"
            f"{_as_python_string(value)}>"
        )


def preferred_memory_capacity(
    img=None,
    image_size=None,
    tile_size=None,
    patch_size=None,
    bytes_per_pixel=3,
):
    cdef vector[uint32_t] cpp_patch_size = _to_uint32_vector(patch_size)
    cdef vector[uint64_t] cpp_image_size
    cdef vector[uint32_t] cpp_tile_size
    cdef uint32_t result
    cdef uint32_t cpp_bytes_per_pixel = bytes_per_pixel
    cdef _CuImageHandle image_handle
    if img is not None:
        if not isinstance(img, CuImage):
            raise TypeError("img must be a CuImage or None")
        image_handle = img._handle
        with nogil:
            result = cpp_preferred_memory_capacity_for_image(
                image_handle.value, cpp_patch_size, cpp_bytes_per_pixel
            )
        return result

    cpp_image_size = _to_uint64_vector(image_size)
    cpp_tile_size = _to_uint32_vector(tile_size)
    with nogil:
        result = cpp_preferred_memory_capacity_explicit(
            cpp_image_size,
            cpp_tile_size,
            cpp_patch_size,
            cpp_bytes_per_pixel,
        )
    return result


def filesystem_open(file_path, flags, mode=0o644):
    cdef string cpp_path = _as_cpp_string(file_path)
    cdef string cpp_flags = _as_cpp_string(flags)
    cdef shared_ptr[NativeCuFileDriver] value
    cdef uint32_t cpp_mode = mode
    with nogil:
        value = cpp_filesystem_open(cpp_path, cpp_flags, cpp_mode)
    return _wrap_file(value)


def filesystem_pread(fd, buf, count, file_offset, buf_offset=0):
    if fd is None:
        print("fd (CuFileDriver) is None!", file=sys.stderr)
        return -1
    if not isinstance(fd, CuFileDriver):
        raise TypeError("fd must be a CuFileDriver")
    return fd.pread(buf, count, file_offset, buf_offset)


def filesystem_pwrite(fd, buf, count, file_offset, buf_offset=0):
    if fd is None:
        print("fd (CuFileDriver) is None!", file=sys.stderr)
        return -1
    if not isinstance(fd, CuFileDriver):
        raise TypeError("fd must be a CuFileDriver")
    return fd.pwrite(buf, count, file_offset, buf_offset)


def filesystem_close(fd):
    if fd is None:
        return False
    if not isinstance(fd, CuFileDriver):
        raise TypeError("fd must be a CuFileDriver")
    return fd.close()


def filesystem_is_gds_available():
    cdef cbool result
    with nogil:
        result = cpp_filesystem_is_gds_available()
    return bool(result)


def filesystem_discard_page_cache(file_path):
    cdef string cpp_path = _as_cpp_string(file_path)
    cdef cbool result
    with nogil:
        result = cpp_filesystem_discard_page_cache(cpp_path)
    return bool(result)


def _get_plugin_root():
    cdef string value
    with nogil:
        value = cpp_get_plugin_root()
    return _as_python_string(value)


def _set_plugin_root(path):
    cdef string value = _as_cpp_string(path)
    with nogil:
        cpp_set_plugin_root(value)


io = types.ModuleType(f"{__name__}.io")
io.DeviceType = DeviceType
io.Device = Device
sys.modules[io.__name__] = io

cache = types.ModuleType(f"{__name__}.cache")
cache.CacheType = CacheType
cache.ImageCache = ImageCache
cache.preferred_memory_capacity = preferred_memory_capacity
sys.modules[cache.__name__] = cache

filesystem = types.ModuleType(f"{__name__}.filesystem")
filesystem.FileHandleType = FileHandleType
filesystem.CuFileDriver = CuFileDriver
filesystem.is_gds_available = filesystem_is_gds_available
filesystem.open = filesystem_open
filesystem.pread = filesystem_pread
filesystem.pwrite = filesystem_pwrite
filesystem.close = filesystem_close
filesystem.discard_page_cache = filesystem_discard_page_cache
sys.modules[filesystem.__name__] = filesystem

profiler = types.ModuleType(f"{__name__}.profiler")
profiler.Profiler = Profiler
sys.modules[profiler.__name__] = profiler

__version__ = _as_python_string(cpp_library_version())
