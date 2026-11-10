/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_py.h"

#include <stdexcept>

#include <cucim/memory/memory_manager.h>

namespace cucim::memory
{

void get_memory_info(
    nb::handle buf_obj, void** out_buf, cucim::io::Device* out_device, size_t* out_memory_size, bool* out_readonly)
{
    if (out_buf == nullptr)
    {
        throw std::runtime_error("[Error] out_buf shouldn't be nullptr!");
    }

    void* buf = nullptr;
    size_t memory_size = 0;

    const char* interface_name = nb::hasattr(buf_obj, "__array_interface__")      ? "__array_interface__" :
                                 nb::hasattr(buf_obj, "__cuda_array_interface__") ? "__cuda_array_interface__" :
                                                                                    nullptr;
    if (interface_name)
    {
        nb::object attr = nb::getattr(buf_obj, interface_name);
        nb::object data =
            nb::isinstance<nb::dict>(attr) ? nb::borrow<nb::dict>(attr).get("data", nb::none()) : nb::none();
        if (nb::isinstance<nb::tuple>(data) && nb::len(data) == 2)
        {
            auto data_tuple = nb::borrow<nb::tuple>(data);
            if (out_readonly)
            {
                *out_readonly = nb::cast<bool>(data_tuple[1]);
            }
            buf = reinterpret_cast<void*>(nb::cast<uint64_t>(data_tuple[0]));
            if (nb::hasattr(buf_obj, "nbytes"))
            {
                memory_size = nb::cast<size_t>(nb::getattr(buf_obj, "nbytes"));
            }
        }
    }
    else if (nb::isinstance<nb::int_>(buf_obj))
    {
        buf = reinterpret_cast<void*>(nb::cast<uint64_t>(buf_obj));
    }

    *out_buf = buf;
    if (out_memory_size)
    {
        *out_memory_size = memory_size;
    }

    if (buf == nullptr)
    {
        return;
    }

    if (out_device)
    {
        cucim::memory::PointerAttributes attributes;
        cucim::memory::get_pointer_attributes(attributes, buf);
        *out_device = attributes.device;
    }
}

} // namespace cucim::memory
