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
#ifndef PYCUCIM_MEMORY_INIT_H
#define PYCUCIM_MEMORY_INIT_H

#include <pybind11/pybind11.h>

#include <cucim/io/device.h>

namespace py = pybind11;

namespace cucim::memory
{

void init_memory(py::module& m);

void get_memory_info(const py::object& buf_obj,
                     void** out_buf,
                     cucim::io::Device* out_device = nullptr,
                     size_t* out_memory_size = 0,
                     bool* out_readonly = nullptr);

} // namespace cucim::memory


#endif // PYCUCIM_MEMORY_INIT_H
