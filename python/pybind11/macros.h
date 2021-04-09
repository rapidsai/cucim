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
