# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_native_extension_uses_limited_abi():
    from cucim.clara import _cucim

    assert Path(_cucim.__file__).name.endswith(".abi3.so")
    assert _cucim.__version__


def test_binding_module_compatibility_surface():
    from cucim.clara import CuImage, DLDataType, DLDataTypeCode
    from cucim.clara._cucim.cache import CacheType, ImageCache
    from cucim.clara._cucim.filesystem import CuFileDriver, FileHandleType
    from cucim.clara._cucim.io import Device, DeviceType
    from cucim.clara._cucim.profiler import Profiler

    assert CuImage is not None
    assert DLDataType(DLDataTypeCode.DLUInt, 8, 1) == DLDataType(
        DLDataTypeCode.DLUInt, 8, 1
    )
    assert Device("cpu").type == DeviceType.CPU
    assert int(CacheType.PerProcess) == 1
    assert int(FileHandleType.GPUDirect) == 8
    assert ImageCache is not None
    assert CuFileDriver is not None
    assert Profiler is not None
