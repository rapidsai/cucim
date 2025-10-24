/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/profiler/profiler.h"

#include "cucim/cuimage.h"

namespace cucim::profiler
{

Profiler::Profiler(ProfilerConfig& config) : config_(config){};

ProfilerConfig& Profiler::config()
{
    return config_;
}

ProfilerConfig Profiler::get_config() const
{
    return config_;
}

void Profiler::trace(bool value)
{
    config_.trace = value;
}

/**
 * @brief Return whether if trace is enabled or not
 *
 * @return true if profiler is enabled. false otherwise
 */
bool Profiler::trace() const
{
    return config_.trace;
}

} // namespace cucim::profiler
