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

#include "memory_py.h"
#include "memory_pydoc.h"

#include <pybind11/pybind11.h>

#include <cucim/memory/memory_manager.h>

namespace py = pybind11;

namespace cucim::memory
{

void init_memory(py::module& memory)
{
}

static size_t calculate_memory_size(const std::vector<size_t>& shape, const char* dtype_str)
{
    // TODO: implement method to calculate size
    //       https://github.com/pytorch/pytorch/blob/master/torch/tensor.py#L733  (we can take digit part)
    return 0;
}

void get_memory_info(const py::object& buf_obj,
                     void** out_buf,
                     cucim::io::Device* out_device,
                     size_t* out_memory_size,
                     bool* out_readonly)
{
    if (out_buf == nullptr)
    {
        throw std::runtime_error("[Error] out_buf shouldn't be nullptr!");
    }

    void* buf = nullptr;
    size_t memory_size = 0;

    if (py::hasattr(buf_obj, "__array_interface__"))
    {
        auto attr = py::getattr(buf_obj, "__array_interface__");
        if (py::isinstance<py::dict>(attr))
        {
            auto dict = py::cast<py::dict>(attr);
            if (dict.contains("data"))
            {
                auto data = dict["data"];
                if (py::isinstance<py::tuple>(data))
                {
                    auto data_tuple = data.cast<py::tuple>();
                    if (data_tuple.size() == 2)
                    {
                        if (out_readonly)
                        {
                            *out_readonly = data_tuple[1].cast<bool>();
                        }
                        buf = reinterpret_cast<void*>(data_tuple[0].cast<uint64_t>());
                        if (py::hasattr(buf_obj, "nbytes"))
                        {
                            memory_size = py::getattr(buf_obj, "nbytes").cast<size_t>();
                        }
                        else
                        {
                            // TODO: implement method to calculate size
                        }
                    }
                }
            }
        }
    }
    else if (py::hasattr(buf_obj, "__cuda_array_interface__"))
    {
        auto attr = py::getattr(buf_obj, "__cuda_array_interface__");
        if (py::isinstance<py::dict>(attr))
        {
            auto dict = py::cast<py::dict>(attr);
            if (dict.contains("data"))
            {
                auto data = dict["data"];
                if (py::isinstance<py::tuple>(data))
                {
                    auto data_tuple = data.cast<py::tuple>();
                    if (data_tuple.size() == 2)
                    {
                        if (out_readonly)
                        {
                            *out_readonly = data_tuple[1].cast<bool>();
                        }
                        buf = reinterpret_cast<void*>(data_tuple[0].cast<uint64_t>());
                        if (py::hasattr(buf_obj, "nbytes"))
                        {
                            memory_size = py::getattr(buf_obj, "nbytes").cast<size_t>();
                        }
                        else
                        {
                            // TODO: implement method to calculate size
                        }
                    }
                }
            }
        }
    }
    else if (py::isinstance<py::int_>(buf_obj))
    {
        buf = reinterpret_cast<void*>(buf_obj.cast<uint64_t>());
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