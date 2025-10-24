/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
