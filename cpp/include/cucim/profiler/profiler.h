/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
