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
#include <mutex>
#include <string>
#include <unordered_map>
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

NVIMGCODECDRIVER loadNvimgcodecLibrary() {
  static const char *paths[] = {nvimgcodecLibNameFullVer,
                                nvimgcodecLibNameMajorVer,
                                nvimgcodecLibName};
  NVIMGCODECDRIVER ret = nullptr;
  std::string last_error;
  for (const char *path : paths) {
    ret = dlopen(path, RTLD_NOW);
    if (ret) {
      fprintf(stderr, "[nvimgcodec_dynlink] Successfully loaded: %s\n", path);
      break;
    } else {
      const char* err = dlerror();
      fprintf(stderr, "[nvimgcodec_dynlink] Failed to load %s: %s\n", path, err ? err : "unknown error");
      if (err) last_error = err;
    }
  }

  if (!ret) {
    std::string error_msg = "dlopen libnvimgcodec.so failed! Last error: " + last_error +
        "\nPlease install nvimagecodec: See https://developer.nvidia.com/nvimgcodec-downloads.";
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
