/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_PROFILER_PROFILER_H
#define CUCIM_PROFILER_PROFILER_H

#include "cucim/core/framework.h"

#include <memory>

#include "cucim/profiler/profiler_config.h"


namespace cucim::profiler
{

/**
 * @brief Profiler class
 *
 * Holds the profiler state and provides the interface to configure it.
 *
 */

class EXPORT_VISIBLE Profiler : public std::enable_shared_from_this<Profiler>
{
public:
    Profiler() = delete;
    Profiler(ProfilerConfig& config);
    virtual ~Profiler(){};

    ProfilerConfig& config();
    ProfilerConfig get_config() const;

    void trace(bool value);

    /**
     * @brief Return whether if trace is enabled or not
     *
     * @return true if profiler is enabled. false otherwise
     */
    bool trace() const;

protected:
    ProfilerConfig& config_;
};

} // namespace cucim::profiler

#endif // CUCIM_PROFILER_PROFILER_H
