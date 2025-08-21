/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#ifndef CUCIM_CUFILE_STUB_H
#define CUCIM_CUFILE_STUB_H

// Try to include the real cufile.h, fall back to minimal types if not available
#if __has_include(<cufile.h>)
    #include <cufile.h>
#else
    #include "cufile_stub_types.h"
#endif

#include "cucim/dynlib/helper.h"

class CuFileStub
{
public:
    void load();
    void unload();
    ~CuFileStub();

private:
    cucim::dynlib::LibraryHandle handle_ = nullptr;
};

#endif // CUCIM_CUFILE_STUB_H
