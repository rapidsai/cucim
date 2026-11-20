/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cufile.h>

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
