#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import enum
import importlib
import os
import pathlib

import numpy as np
import pytest

import cucim
from cucim import CuImage
from cucim.clara import DLDataType, DLDataTypeCode, cache, filesystem, io
from cucim.clara._cucim import profiler

INT_ENUMS = {DLDataTypeCode}
ENUM_MEMBERS = {
    DLDataTypeCode: {"DLInt": 0, "DLUInt": 1, "DLFloat": 2, "DLBfloat": 4},
    cache.CacheType: {"NoCache": 0, "PerProcess": 1, "SharedMemory": 2},
    io.DeviceType: {
        "CPU": 1,
        "CUDA": 2,
        "CUDAHost": 3,
        "CUDAManaged": 13,
        "CPUShared": 101,
        "CUDAShared": 102,
    },
    filesystem.FileHandleType: {
        "Unknown": 0,
        "Posix": 1,
        "PosixODirect": 2,
        "MemoryMapped": 4,
        "GPUDirect": 8,
    },
}


def test_public_names_are_importable():
    assert cucim.CuImage is CuImage
    for name in ("clara", "clara.cache", "clara.io", "clara.filesystem"):
        module = importlib.import_module(f"cucim.{name}")
        for attr in module.__all__:
            assert hasattr(module, attr), f"{module.__name__}.{attr}"
    assert callable(filesystem.is_gds_available)
    assert isinstance(cucim.clara._cucim.__version__, str)


@pytest.mark.parametrize("enum_type", ENUM_MEMBERS, ids=lambda e: e.__name__)
def test_enum_members(enum_type):
    expected = ENUM_MEMBERS[enum_type]
    is_int_enum = enum_type in INT_ENUMS
    assert issubclass(enum_type, enum.Enum)
    assert issubclass(enum_type, int) is is_int_enum
    assert list(enum_type.__members__) == list(expected)
    members = []
    for name, value in expected.items():
        member = getattr(enum_type, name)
        members.append(member)
        assert member.name == name
        assert member.value == value
        assert enum_type(value) is member
        assert str(member) == f"{enum_type.__name__}.{name}"
        assert repr(member) == str(member)
        assert format(member) == f"{member}" == str(member)
        assert (member == value) is is_int_enum
        if is_int_enum:
            assert int(member) == value
        else:
            with pytest.raises(TypeError):
                int(member)
    assert len(set(members)) == len(members)
    assert members[0] != members[1]
    with pytest.raises(ValueError):
        enum_type(99)


def test_dldatatype():
    dtype = DLDataType(DLDataTypeCode.DLUInt, 8, 1)
    assert (dtype.code, dtype.bits, dtype.lanes) == (1, 8, 1)
    assert type(dtype.code) is int
    assert dtype.code == DLDataTypeCode.DLUInt
    assert dtype == DLDataType(DLDataTypeCode.DLUInt, 8, 1)
    assert dtype == DLDataType(1, 8, 1)
    assert dtype != DLDataType(DLDataTypeCode.DLFloat, 32, 1)
    assert (
        repr(dtype) == "<cucim.clara.DLDataType code:DLUInt(1) bits:8 lanes:1>"
    )
    with pytest.raises(TypeError):
        hash(dtype)


def test_device():
    cpu = io.Device("cpu")
    assert cpu.type == io.DeviceType.CPU
    assert io.Device("cuda").type == io.DeviceType.CUDA
    assert io.Device.parse_type("cuda") == io.DeviceType.CUDA
    assert isinstance(cpu.index, int)
    assert isinstance(repr(cpu), str)


def test_open_errors():
    with pytest.raises(ValueError, match="Cannot open"):
        CuImage("/tmp/cucim-test-binding-nonexisting.tif")
    with pytest.raises(TypeError):
        CuImage()


def test_cache_api(restore_cache):
    per_process = CuImage.cache(
        "per_process", memory_capacity=16, record_stat=True
    )
    assert isinstance(per_process, cache.ImageCache)
    assert per_process.type == cache.CacheType.PerProcess
    config = per_process.config
    assert list(config) == [
        "type",
        "memory_capacity",
        "capacity",
        "mutex_pool_capacity",
        "list_padding",
        "extra_shared_memory_size",
        "record_stat",
    ]
    assert config["type"] == "per_process"
    assert config["memory_capacity"] == 16
    assert config["record_stat"] is True
    assert per_process.record() is True
    assert per_process.record(False) is False
    assert per_process.record() is False
    assert per_process.record(True) is True
    with pytest.raises(ValueError):
        per_process.record("yes")

    for name in (
        "size",
        "memory_size",
        "capacity",
        "memory_capacity",
        "free_memory",
        "hit_count",
        "miss_count",
    ):
        assert type(getattr(per_process, name)) is int, name
    assert per_process.memory_capacity == 16 * 1024 * 1024
    assert per_process.free_memory == (
        per_process.memory_capacity - per_process.memory_size
    )

    per_process.reserve(32)
    assert per_process.memory_capacity == 32 * 1024 * 1024
    per_process.reserve(memory_capacity=64, capacity=4000)
    assert per_process.capacity == 4000

    assert CuImage.cache().type == cache.CacheType.PerProcess
    assert CuImage.cache("nocache").type == cache.CacheType.NoCache
    with pytest.raises(ValueError, match="first argument"):
        CuImage.cache(1)
    with pytest.raises(RuntimeError):
        CuImage.cache("per_process", capacity="1")


def test_profiler_api(restore_trace):
    prof = CuImage.profiler()
    assert isinstance(prof, profiler.Profiler)
    assert prof.trace(True) is True
    assert prof.config == {"trace": True}
    assert prof.trace() is True
    assert prof.trace(False) is False
    assert isinstance(CuImage.is_trace_enabled, bool)
    with pytest.raises(ValueError):
        prof.trace(1)


def test_profiler_config_keyword(restore_trace):
    assert CuImage.profiler(trace=True).config == {"trace": True}
    assert CuImage.profiler(trace=False).config == {"trace": False}


def test_filesystem_roundtrip(tmp_path):
    path = str(tmp_path / "data.bin")
    payload = np.arange(4096, dtype=np.uint8)

    fd = filesystem.open(path, "w")
    assert isinstance(fd, filesystem.CuFileDriver)
    assert repr(fd) == f"<cucim.clara.filesystem.CuFileDriver path:{path}>"
    assert filesystem.pwrite(fd, payload, payload.nbytes, 0) == payload.nbytes
    assert fd.pwrite(payload, 16, 4096, 16) == 16
    assert filesystem.close(fd) is True

    with filesystem.open(path, "r") as reader:
        out = np.zeros(4096, dtype=np.uint8)
        assert reader.pread(out, out.nbytes, 0) == out.nbytes
        np.testing.assert_array_equal(out, payload)
        tail = np.zeros(16, dtype=np.uint8)
        assert filesystem.pread(reader, tail, 16, 4096) == 16
        np.testing.assert_array_equal(tail, payload[16:32])
        by_pointer = np.zeros(16, dtype=np.uint8)
        assert reader.pread(by_pointer.ctypes.data, 16, 0) == 16
        np.testing.assert_array_equal(by_pointer, payload[:16])

        with pytest.raises(RuntimeError, match="readonly"):
            reader.pread(np.frombuffer(bytes(16), np.uint8), 16, 0)
        with pytest.raises(RuntimeError, match="larger than"):
            reader.pread(np.zeros(8, np.uint8), 16, 0)
        with pytest.raises(RuntimeError, match="Recognize"):
            reader.pread([0] * 16, 16, 0)
        assert filesystem.pread(None, out, 16, 0) == -1

    with filesystem.CuFileDriver(
        os.open(path, os.O_RDONLY), no_gds=True, file_path=path
    ) as driver:
        assert repr(driver).endswith(f"path:{path}>")
        out = np.zeros(16, dtype=np.uint8)
        assert driver.pread(out, 16, 0) == 16
        np.testing.assert_array_equal(out, payload[:16])

    assert isinstance(filesystem.is_gds_available(), bool)
    assert isinstance(filesystem.discard_page_cache(path), bool)


def test_device_cache():
    assert isinstance(CuImage.device_cache(), cache.ImageCache)


def test_bool_arguments_are_strict(tmp_path, restore_cache, restore_trace):
    for flag in (1, 0, np.True_, np.False_, None):
        with pytest.raises(RuntimeError):
            CuImage.cache("per_process", record_stat=flag)
        with pytest.raises(RuntimeError):
            CuImage.profiler(trace=flag)

    path = tmp_path / "data.bin"
    path.write_bytes(bytes(16))
    fd = os.open(path, os.O_RDONLY)
    try:
        for flag in (1, np.True_, None):
            with pytest.raises(TypeError):
                filesystem.CuFileDriver(fd, no_gds=flag)
            with pytest.raises(TypeError):
                filesystem.CuFileDriver(fd, use_mmap=flag)
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "as_path", [str, os.fsencode, pathlib.Path], ids=["str", "bytes", "pathlib"]
)
def test_filesystem_path_arguments(tmp_path, as_path):
    path = tmp_path / "data.bin"
    payload = np.arange(16, dtype=np.uint8)
    path.write_bytes(payload.tobytes())

    with filesystem.open(as_path(path), "r") as reader:
        out = np.zeros(16, np.uint8)
        assert reader.pread(out, 16, 0) == 16
        np.testing.assert_array_equal(out, payload)
    with filesystem.CuFileDriver(
        os.open(path, os.O_RDONLY), no_gds=True, file_path=as_path(path)
    ) as driver:
        assert repr(driver).endswith(f"path:{path}>")
    assert filesystem.discard_page_cache(as_path(path)) is True


def test_filesystem_opens_non_utf8_file_name(tmp_path):
    path = os.path.join(os.fsencode(tmp_path), b"data-\xff.bin")
    payload = np.arange(16, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(payload.tobytes())

    for path_arg in (path, os.fsdecode(path)):
        with filesystem.open(path_arg, "r") as reader:
            out = np.zeros(16, np.uint8)
            assert reader.pread(out, 16, 0) == 16
            np.testing.assert_array_equal(out, payload)


def test_filesystem_open_rejects_invalid_paths(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(bytes(16))
    for path_arg in (
        bytearray(os.fsencode(path)),
        f"{path}\0.bin",
        os.fsencode(path) + b"\0.bin",
    ):
        with pytest.raises(TypeError):
            filesystem.open(path_arg, "r")


def test_text_arguments_reject_bytes(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(bytes(16))
    with pytest.raises(TypeError):
        io.Device(b"cpu")
    with pytest.raises(TypeError):
        filesystem.open(path, b"r")
