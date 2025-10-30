# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Prefer FFTs via the new scipy.fft module when available (SciPy 1.4+)

Otherwise fall back to numpy.fft.

Like numpy 1.15+ scipy 1.3+ is also using pocketfft, but a newer
C++/pybind11 version called pypocketfft
"""
import cupyx.scipy.fft
from cupyx.scipy.fft import next_fast_len

fftmodule = cupyx.scipy.fft

__all__ = ["fftmodule", "next_fast_len"]
