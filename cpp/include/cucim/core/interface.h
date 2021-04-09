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

#ifndef CUCIM_INTERFACE_H
#define CUCIM_INTERFACE_H


#include "cucim/core/version.h"

namespace cucim
{

struct InterfaceDesc
{
    const char* name = nullptr;
    InterfaceVersion version = { 0, 1 };
};

/**
 * Macro to declare a plugin interface.
 */
#define CUCIM_PLUGIN_INTERFACE(name, major, minor)                                                                   \
    static cucim::InterfaceDesc get_interface_desc()                                                                     \
    {                                                                                                                  \
        return cucim::InterfaceDesc{ name, { major, minor } };                                                           \
    }

} // namespace cucim

#include "../macros/defines.h"

#endif // CUCIM_INTERFACE_H
