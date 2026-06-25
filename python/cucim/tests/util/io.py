#
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


def open_image_cucim(file_path):
    from cucim import CuImage

    img = CuImage(file_path)
    return img
