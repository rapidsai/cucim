/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_MACRO_H
#define PYCUCIM_MACRO_H

#include <string>

#include <fmt/format.h>

constexpr const char* remove_leading_spaces(const char* str)
{
    return *str == '\0' ? str : ((*str == ' ' || *str == '\n') ? remove_leading_spaces(str + 1) : str);
}

#define PYDOC(method, doc) static constexpr const char* doc_##method = remove_leading_spaces(doc);

#endif // PYCUCIM_MACRO_H
