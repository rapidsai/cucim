/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
