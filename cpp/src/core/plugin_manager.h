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

#ifndef CUCIM_PLUGIN_MANAGER_H
#define CUCIM_PLUGIN_MANAGER_H

#include <cstddef>
#include <limits>
#include <vector>
#include <memory>
#include <unordered_set>

#include "cucim/macros/defines.h"
namespace cucim
{

class Plugin;

const size_t kInvalidPluginIndex = std::numeric_limits<size_t>::max();

class PluginManager
{
public:
    size_t add_plugin(std::shared_ptr<Plugin> plugin)
    {
        size_t index = plugin_list_.size();
        plugin_list_.push_back(std::move(plugin));
        plugin_indices_.insert(index);
        return index;
    }

    void remove_plugin(size_t index)
    {
        CUCIM_ASSERT(plugin_indices_.find(index) != plugin_indices_.end());
        CUCIM_ASSERT(index < plugin_list_.size());
        plugin_indices_.erase(index);
        plugin_list_[index] = nullptr;
    }

    Plugin* get_plugin(size_t index) const
    {
        CUCIM_ASSERT(index < plugin_list_.size());
        return plugin_list_[index].get();
    }

    const std::unordered_set<size_t>& get_plugin_indices() const
    {
        return plugin_indices_;
    }

private:
    std::vector<std::shared_ptr<Plugin>> plugin_list_;
    std::unordered_set<size_t> plugin_indices_;
};
} // namespace cucim
#endif // CUCIM_PLUGIN_MANAGER_H
