#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# https://github.com/uclouvain/openjpeg/blob/v2.4.0/src/bin/common/color.c#L60

###############################################################################
# Matrix for sYCC, Amendment 1 to IEC 61966-2-1
#
# Y :   0.299   0.587    0.114   :R
# Cb:  -0.1687 -0.3312   0.5     :G
# Cr:   0.5    -0.4187  -0.0812  :B
#
# Inverse:
#
# R: 1        -3.68213e-05    1.40199      :Y
# G: 1.00003  -0.344125      -0.714128     :Cb - 2^(prec - 1)
# B: 0.999823  1.77204       -8.04142e-06  :Cr - 2^(prec - 1)
###############################################################################

"""
Color Conversion table generator.

Instrumented on the following image to see which pre-calculation can minimize
errors, compared with openjpeg's slow-color-conversion logic.

We chose to use the approach of `count_cr0.5.txt`

>>> input_file = "notebooks/input/TUPAC-TR-467.svs"
>>> img = CuImage(input_file)
>>> region = img.read_region(level=2)

Configuration for `gen_g_cb()` and `gen_g_cr()` was changed.

```c++
if (*buf != comp0[i])
{
    fprintf(stdout, "%u(0):  %d != %d  #%d\n ", i, *buf, comp0[i],
        ((uint8_t)(*buf) - (uint8_t)comp0[i]));
}
*(buf++) = comp0[i];
if (*buf != comp1[i])
{
    fprintf(stdout, "%u(1):  %d != %d  #%d\n", i, *buf, comp1[i],
        ((uint8_t)(*buf) - (uint8_t)comp1[i]));
}
*(buf++) = comp1[i];
if (*buf != comp2[i])
{
    fprintf(stdout, "%u(2):  %d != %d  #%d\n", i, *buf, comp2[i],
        ((uint8_t)(*buf) - (uint8_t)comp2[i]));
}
*(buf++) = comp2[i];
```

❯ grep -c "#-1" count_both0.5.txt
1286
❯ grep -c "#1" count_both0.5.txt
1275184

❯ grep -c "#1" count_both0.txt
0
❯ grep -c "#-1" count_both0.txt
1125962

❯ grep -c "#-1" count_cb0.5.txt
511399
❯ grep -c "#1" count_cb0.5.txt
248788

❯ grep -c "#-1" count_round_cb0.5.txt
511399
❯ grep -c "#1" count_round_cb0.5.txt
248788

❯ grep -c "#-1" count_cr0.5.txt
511399
❯ grep -c "#1" count_cr0.5.txt
248788

❯ grep -c "#-1" count_round_cr0.5.txt
511399
❯ grep -c "#1" count_round_cr0.5.txt
248788

❯ grep -c "#-1" count_round_short_cb0.5.txt
508465
❯ grep -c "#1" count_round_short_cb0.5.txt
248808

"""


def gen_r_cr():
    """
    Generate the R-Cr table.
    """
    r_cr = [0] * 256
    for i in range(256):
        r_cr[i] = int(1.40199 * (i - 128))
    return r_cr


def gen_g_cb():
    """
    Generate the G-Cb table.
    """
    g_cb = [0] * 256
    for i in range(256):
        g_cb[i] = int((-0.344125 * (i - 128)) * (1 << 16))
    return g_cb


def gen_g_cr():
    """
    Generate the G-Cr table.
    """
    g_cr = [0] * 256
    for i in range(256):
        g_cr[i] = int((-0.714128 * (i - 128) + 0.5) * (1 << 16))
    return g_cr


def gen_b_cb():
    """
    Generate the B-Cb table.
    """
    b_cb = [0] * 256
    for i in range(256):
        b_cb[i] = int(1.77204 * (i - 128))
    return b_cb


TEMPLATE = """// This file is generated by gen_color_table.py

// clang-format off
#ifndef CUSLIDE_JPEG2K_COLOR_TABLE_H
#define CUSLIDE_JPEG2K_COLOR_TABLE_H

namespace cuslide::jpeg2k
{

static constexpr int16_t R_Cr[256] = {
    %(r_cr)s
};

static constexpr int32_t G_Cb[256]  = {
    %(g_cb)s
};

static constexpr int32_t G_Cr[256]  = {
    %(g_cr)s
};

static constexpr int16_t B_Cb[256]  = {
    %(b_cb)s
};

} // namespace cuslide::jpeg2k

#endif // CUSLIDE_JPEG2K_COLOR_TABLE_H
// clang-format on
"""


def gen_list(values: list, width: int, align: int = 8):
    text = []
    for i in range(0, len(values), width):
        text.append(
            ", ".join(
                ("{:>" + str(align) + "}").format(item)
                for item in values[i : i + width]
            )
        )
    return ",\n    ".join(text)


def main(output_file_name: str) -> int:
    r_cr = gen_list(list(gen_r_cr()), 10, 4)
    g_cb = gen_list(list(gen_g_cb()), 10)
    g_cr = gen_list(list(gen_g_cr()), 10)
    b_cb = gen_list(list(gen_b_cb()), 10, 4)

    with open(output_file_name, "w") as f:
        f.write(
            TEMPLATE % {"r_cr": r_cr, "g_cb": g_cb, "g_cr": g_cr, "b_cb": b_cb}
        )

    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: gen_color_table.py <output_file_name>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
