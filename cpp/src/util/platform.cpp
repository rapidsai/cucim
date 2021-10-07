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

#include "cucim/util/platform.h"

#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>


namespace cucim::util
{

bool is_in_wsl()
{
    struct utsname buf;
    int err = uname(&buf);
    if (err == 0)
    {
        char* pos = strstr(buf.release, "icrosoft");
        if (pos)
        {
            // 'Microsoft' for WSL1 and 'microsoft' for WSL2
            if (buf.release < pos && (pos[-1] == 'm' || pos[-1] == 'M'))
            {
                return true;
            }
        }
    }
    return false;
}

} // namespace cucim::util
