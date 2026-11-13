// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <nvimgcodec_version.h>
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#define STR_IMPL_(x) #x
#define STR(x) STR_IMPL_(x)
#define FULL_VER_STR               \
  STR(NVIMGCODEC_VER_MAJOR) \
  "." STR(NVIMGCODEC_VER_MINOR) "." STR(NVIMGCODEC_VER_PATCH)
#define MAJOR_VER_STR STR(NVIMGCODEC_VER_MAJOR)

namespace {

typedef void *NVIMGCODECDRIVER;

const char nvimgcodecLibNameFullVer[] = "libnvimgcodec.so." FULL_VER_STR;
const char nvimgcodecLibNameMajorVer[] = "libnvimgcodec.so." MAJOR_VER_STR;
const char nvimgcodecLibName[] = "libnvimgcodec.so";

// Directory of the shared object that contains this code (the plugin), or an
// empty string if it cannot be determined.
std::string pluginDirectory() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&pluginDirectory), &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  std::string file(info.dli_fname);
  auto pos = file.rfind('/');
  return pos == std::string::npos ? std::string(".") : file.substr(0, pos);
}

NVIMGCODECDRIVER loadNvimgcodecLibrary() {
  static const char *names[] = {nvimgcodecLibNameFullVer,
                                nvimgcodecLibNameMajorVer,
                                nvimgcodecLibName};
  std::vector<std::string> candidates(std::begin(names), std::end(names));

  // The nvidia-nvimgcodec-cu* pip packages install the library in
  // site-packages/nvidia/nvimgcodec/, which is not on the default search path.
  // The plugin is installed in site-packages/cucim/clara/, so look in the
  // directory of the nvImageCodec package that is next to cucim as well.
  // Libraries found by the default search (system installs, conda,
  // LD_LIBRARY_PATH) take precedence.
  const std::string plugin_dir = pluginDirectory();
  if (!plugin_dir.empty()) {
    for (const char *name : names) {
      candidates.push_back(plugin_dir + "/../../nvidia/nvimgcodec/" + name);
    }
  }

  NVIMGCODECDRIVER ret = nullptr;
  std::string errors;
  for (const std::string &path : candidates) {
    ret = dlopen(path.c_str(), RTLD_NOW);
    if (ret) {
      fprintf(stderr, "[nvimgcodec_dynlink] Successfully loaded: %s\n", path.c_str());
      break;
    }
    const char *err = dlerror();
    errors += "\n  " + path + ": " + (err ? err : "unknown error");
  }

  if (!ret) {
    std::string error_msg = "dlopen libnvimgcodec.so failed! Tried:" + errors +
        "\nPlease install nvImageCodec (pip: nvidia-nvimgcodec-cu12 or nvidia-nvimgcodec-cu13, "
        "conda: libnvimgcodec): See https://developer.nvidia.com/nvimgcodec-downloads.";
    throw std::runtime_error(error_msg);
  }
  return ret;
}

}  // namespace

void *NvimgcodecLoadSymbol(const char *name) {
  static NVIMGCODECDRIVER nvimgcodecDrvLib = loadNvimgcodecLibrary();
  void *ret = nvimgcodecDrvLib ? dlsym(nvimgcodecDrvLib, name) : nullptr;
  if (!ret) {
    fprintf(stderr, "[nvimgcodec_dynlink] WARNING: Symbol '%s' not found\n", name);
  }
  return ret;
}

bool nvimgcodecIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void *> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvimgcodecLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
