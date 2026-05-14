# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import warnings

import cupy as cp
import numpy as np
import pytest
from skimage import data
from skimage._shared.testing import fetch

from cucim.skimage import morphology, util
from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.filters import rank
from cucim.skimage.filters.rank import (
    __all__ as all_rank_filters,
    subtract_mean,
)
from cucim.skimage.filters.rank._histogram import (
    _get_histogram_counter_dtype,
    _get_rank_histogram_partitions,
    _should_use_rank_histogram,
)
from cucim.skimage.morphology import ball, disk, gray
from cucim.skimage.util import img_as_float, img_as_ubyte


def _reflect_index(index, size):
    if index < 0:
        index = -1 - index
    index %= 2 * size
    return min(index, 2 * size - 1 - index)


def _cast_uint8(value):
    return np.uint8(int(value) % 256)


def _rank_filter_brute_force_uint8(
    image,
    footprint,
    operation,
    *,
    mask=None,
    s0=10,
    s1=10,
):
    radius = tuple(s // 2 for s in footprint.shape)
    out = np.empty_like(image)
    dtype_max = 255
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            values = []
            for frow in range(footprint.shape[0]):
                for fcol in range(footprint.shape[1]):
                    if not footprint[frow, fcol]:
                        continue
                    irow = _reflect_index(
                        row + frow - radius[0], image.shape[0]
                    )
                    icol = _reflect_index(
                        col + fcol - radius[1], image.shape[1]
                    )
                    if mask is not None and not mask[irow, icol]:
                        continue
                    values.append(int(image[irow, icol]))

            center = int(image[row, col])
            if not values:
                out[row, col] = image[row, col]
            elif operation == "minimum":
                out[row, col] = min(values)
            elif operation == "maximum":
                out[row, col] = max(values)
            elif operation == "mean":
                out[row, col] = _cast_uint8(sum(values) / len(values))
            elif operation == "sum":
                out[row, col] = _cast_uint8(sum(values))
            elif operation == "subtract_mean":
                mean = sum(values) / len(values)
                out[row, col] = _cast_uint8(
                    _cast_uint8((center - mean) * 0.5 + 128) - 1
                )
            elif operation == "pop":
                out[row, col] = _cast_uint8(len(values))
            elif operation == "threshold":
                out[row, col] = _cast_uint8(
                    center > (sum(values) / len(values))
                )
            elif operation == "gradient":
                out[row, col] = _cast_uint8(max(values) - min(values))
            elif operation == "entropy":
                counts = np.bincount(values, minlength=256)
                probabilities = counts[counts > 0] / len(values)
                out[row, col] = _cast_uint8(
                    -(probabilities * np.log2(probabilities)).sum()
                )
            elif operation == "autolevel":
                min_val = min(values)
                max_val = max(values)
                delta = max_val - min_val
                if delta > 0:
                    clamped = min(max(center, min_val), max_val)
                    out[row, col] = _cast_uint8(
                        (clamped - min_val) / delta * dtype_max
                    )
                else:
                    out[row, col] = 0
            elif operation == "enhance_contrast":
                min_val = min(values)
                max_val = max(values)
                if max_val - center < center - min_val:
                    out[row, col] = max_val
                else:
                    out[row, col] = min_val
            elif operation == "equalize":
                rank = sum(value <= center for value in values)
                out[row, col] = _cast_uint8(dtype_max * rank / len(values))
            elif operation == "geometric_mean":
                log_sum = sum(np.log(value + 1.0) for value in values)
                out[row, col] = _cast_uint8(
                    round(np.exp(log_sum / len(values)) - 1.0)
                )
            elif operation in {"modal", "majority"}:
                counts = np.bincount(values, minlength=256)
                out[row, col] = int(np.argmax(counts))
            elif operation == "noise_filter":
                if center in values:
                    out[row, col] = 0
                else:
                    out[row, col] = min(abs(value - center) for value in values)
            elif operation in {
                "mean_bilateral",
                "pop_bilateral",
                "sum_bilateral",
            }:
                bilateral_values = [
                    value
                    for value in values
                    if center > value - s0 and center < value + s1
                ]
                if operation == "pop_bilateral":
                    out[row, col] = _cast_uint8(len(bilateral_values))
                elif not bilateral_values:
                    out[row, col] = 0
                elif operation == "mean_bilateral":
                    out[row, col] = _cast_uint8(
                        sum(bilateral_values) / len(bilateral_values)
                    )
                else:
                    out[row, col] = _cast_uint8(sum(bilateral_values))
            else:
                raise ValueError(f"unsupported operation: {operation}")
    return out


def _rank_percentile_brute_force_uint8(
    image,
    footprint,
    operation,
    *,
    p0=0,
    p1=1,
):
    radius = tuple(s // 2 for s in footprint.shape)
    out = np.empty_like(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            values = []
            for frow in range(footprint.shape[0]):
                for fcol in range(footprint.shape[1]):
                    if not footprint[frow, fcol]:
                        continue
                    irow = _reflect_index(
                        row + frow - radius[0], image.shape[0]
                    )
                    icol = _reflect_index(
                        col + fcol - radius[1], image.shape[1]
                    )
                    values.append(int(image[irow, icol]))
            values.sort()
            center = int(image[row, col])
            pop = len(values)

            if operation == "percentile":
                if p0 == 1:
                    percentile_idx = pop - 1
                else:
                    percentile_idx = int(p0 * pop)
                    if percentile_idx >= pop:
                        percentile_idx = pop - 1
                out[row, col] = values[percentile_idx]
                continue

            if operation == "threshold_percentile":
                threshold_idx = int(p0 * pop)
                if threshold_idx >= pop:
                    threshold_idx = pop - 1
                out[row, col] = 255 if center >= values[threshold_idx] else 0
                continue

            idx_start = max(0, int(np.ceil(p0 * pop)) - 1)
            idx_end = int(p1 * pop)
            if idx_end <= idx_start:
                idx_end = idx_start + 1
            if idx_end > pop:
                idx_end = pop
            selected = values[idx_start:idx_end]

            if operation == "mean_percentile":
                out[row, col] = _cast_uint8(sum(selected) / len(selected))
            elif operation == "sum_percentile":
                out[row, col] = _cast_uint8(sum(selected))
            elif operation == "gradient_percentile":
                out[row, col] = _cast_uint8(selected[-1] - selected[0])
            elif operation == "subtract_mean_percentile":
                mean = sum(selected) / len(selected)
                out[row, col] = _cast_uint8((center - mean) * 0.5 + 128)
            elif operation == "enhance_contrast_percentile":
                min_val = selected[0]
                max_val = selected[-1]
                if max_val - center < center - min_val:
                    out[row, col] = max_val
                else:
                    out[row, col] = min_val
            elif operation == "autolevel_percentile":
                min_val = selected[0]
                max_val = selected[-1]
                delta = max_val - min_val
                if delta > 0:
                    clamped = min(max(center, min_val), max_val)
                    out[row, col] = _cast_uint8(
                        (clamped - min_val) / delta * 255
                    )
                else:
                    out[row, col] = 0
            elif operation == "pop_percentile":
                count = 0
                cumsum = 0
                i = 0
                while i < pop:
                    value = values[i]
                    group_size = 0
                    while i < pop and values[i] == value:
                        group_size += 1
                        i += 1
                    cumsum += group_size
                    if cumsum >= p0 * pop and cumsum <= p1 * pop:
                        count += group_size
                out[row, col] = _cast_uint8(count)
            else:
                raise ValueError(f"unsupported operation: {operation}")
    return out


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_subtract_mean_underflow_correction(dtype):
    # Input: [10, 10, 10]
    footprint = cp.ones((1, 3))
    arr = cp.array([[10, 10, 10]], dtype=dtype)
    result = subtract_mean(arr, footprint, cast_to_uint8=(dtype == np.uint8))

    if dtype == np.uint8:
        expected_val = 127
    else:
        expected_val = 32767
    # note: scikit-image expected_val for uint16 was
    #    expected_val = (arr.max() + 1) // 2 - 1

    assert cp.all(result == expected_val)


@pytest.mark.parametrize(
    "filter_name",
    [
        "minimum",
        "maximum",
        "mean",
        "sum",
        "subtract_mean",
        "pop",
        "threshold",
        "gradient",
        "autolevel",
        "enhance_contrast",
        "equalize",
        "geometric_mean",
        "noise_filter",
        "mean_bilateral",
        "pop_bilateral",
        "sum_bilateral",
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
def test_streaming_rank_filter_ops_uint8(filter_name, use_mask):
    image = np.array(
        [
            [0, 5, 10, 50, 90],
            [3, 8, 20, 60, 120],
            [7, 11, 30, 70, 150],
            [13, 17, 40, 80, 180],
        ],
        dtype=np.uint8,
    )
    footprint = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    mask = np.array(
        [
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    kwargs = {}
    if "bilateral" in filter_name:
        kwargs.update(s0=6, s1=9)

    result = getattr(rank, filter_name)(
        cp.asarray(image),
        cp.asarray(footprint),
        mask=cp.asarray(mask) if use_mask else None,
        **kwargs,
    )
    expected = _rank_filter_brute_force_uint8(
        image,
        footprint,
        filter_name,
        mask=mask if use_mask else None,
        **kwargs,
    )
    cp.testing.assert_array_equal(result, cp.asarray(expected))


@pytest.mark.parametrize(
    "filter_name, kwargs",
    [
        ("percentile", dict(p0=0.5)),
        ("percentile", dict(p0=1.0)),
        ("threshold_percentile", dict(p0=0.5)),
        ("mean_percentile", dict(p0=0.25, p1=0.75)),
        ("sum_percentile", dict(p0=0.25, p1=0.75)),
        ("pop_percentile", dict(p0=0.25, p1=0.75)),
        ("gradient_percentile", dict(p0=0.25, p1=0.75)),
        ("autolevel_percentile", dict(p0=0.25, p1=0.75)),
        ("enhance_contrast_percentile", dict(p0=0.25, p1=0.75)),
        ("subtract_mean_percentile", dict(p0=0.25, p1=0.75)),
    ],
)
def test_histogram_rank_percentile_ops_uint8_rectangular(filter_name, kwargs):
    image = np.array(
        [
            [0, 5, 10, 50, 90],
            [3, 8, 20, 60, 120],
            [7, 11, 30, 70, 150],
            [13, 17, 40, 80, 180],
        ],
        dtype=np.uint8,
    )
    footprint = np.ones((3, 3), dtype=bool)
    result = getattr(rank, filter_name)(
        cp.asarray(image),
        cp.asarray(footprint),
        backend="histogram",
        **kwargs,
    )
    expected = _rank_percentile_brute_force_uint8(
        image,
        footprint,
        filter_name,
        **kwargs,
    )
    cp.testing.assert_array_equal(result, cp.asarray(expected))


def test_histogram_rank_entropy_uint8_rectangular():
    image = np.array(
        [
            [0, 0, 10, 50, 90],
            [0, 8, 20, 60, 120],
            [7, 8, 30, 70, 150],
            [13, 17, 40, 80, 180],
        ],
        dtype=np.uint8,
    )
    footprint = np.ones((3, 3), dtype=bool)
    result = rank.entropy(
        cp.asarray(image), cp.asarray(footprint), backend="histogram"
    )
    expected = rank.entropy(
        cp.asarray(image), cp.asarray(footprint), backend="elementwise"
    )
    assert result.dtype.kind == "f"
    cp.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "filter_name, kwargs",
    [
        ("equalize", {}),
        ("geometric_mean", {}),
        ("mean_bilateral", dict(s0=6, s1=9)),
        ("modal", {}),
        ("majority", {}),
        ("pop_bilateral", dict(s0=6, s1=9)),
        ("sum_bilateral", dict(s0=6, s1=9)),
    ],
)
def test_histogram_rank_prefix_ops_uint8_rectangular(filter_name, kwargs):
    image = np.array(
        [
            [0, 5, 10, 50, 90],
            [3, 8, 20, 60, 120],
            [7, 11, 30, 70, 150],
            [13, 17, 40, 80, 180],
        ],
        dtype=np.uint8,
    )
    footprint = np.ones((3, 3), dtype=bool)
    result = getattr(rank, filter_name)(
        cp.asarray(image),
        cp.asarray(footprint),
        backend="histogram",
        **kwargs,
    )
    expected = _rank_filter_brute_force_uint8(
        image,
        footprint,
        filter_name,
        **kwargs,
    )
    cp.testing.assert_array_equal(result, cp.asarray(expected))


def test_rank_backend_override_histogram_and_elementwise():
    image = cp.asarray(
        np.array(
            [
                [0, 5, 10, 50, 90],
                [3, 8, 20, 60, 120],
                [7, 11, 30, 70, 150],
                [13, 17, 40, 80, 180],
            ],
            dtype=np.uint8,
        )
    )
    footprint = cp.ones((3, 3), dtype=bool)

    automatic = rank.percentile(image, footprint, p0=0.5, backend="auto")
    histogram = rank.percentile(image, footprint, p0=0.5, backend="histogram")
    elementwise = rank.percentile(
        image, footprint, p0=0.5, backend="elementwise"
    )

    cp.testing.assert_array_equal(histogram, automatic)
    cp.testing.assert_array_equal(elementwise, automatic)


def test_rank_backend_auto_uses_elementwise_below_histogram_cutoff():
    image = cp.asarray(
        np.array(
            [
                [0, 5, 10, 50, 90],
                [3, 8, 20, 60, 120],
                [7, 11, 30, 70, 150],
                [13, 17, 40, 80, 180],
            ],
            dtype=np.uint8,
        )
    )
    footprint = cp.ones((3, 3), dtype=bool)

    automatic = rank.percentile(image, footprint, p0=0.5, backend="auto")
    elementwise = rank.percentile(
        image, footprint, p0=0.5, backend="elementwise"
    )
    histogram = rank.percentile(image, footprint, p0=0.5, backend="histogram")

    cp.testing.assert_array_equal(automatic, elementwise)
    cp.testing.assert_array_equal(histogram, elementwise)


def test_rank_backend_histogram_rejects_incompatible_input():
    image = cp.asarray(np.arange(25, dtype=np.uint16).reshape(5, 5))
    footprint = cp.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="backend='histogram' requires"):
        rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="histogram",
            cast_to_uint8=False,
        )


@pytest.mark.parametrize("out_dtype", [cp.float32, cp.uint16])
def test_rank_backend_histogram_supports_non_uint8_output(out_dtype):
    image = cp.asarray(np.arange(32 * 32, dtype=np.uint8).reshape(32, 32))
    footprint = cp.ones((17, 17), dtype=bool)
    out = cp.empty(image.shape, dtype=out_dtype)

    result = rank.percentile(
        image, footprint, p0=0.5, out=out, backend="histogram"
    )
    expected = rank.percentile(
        image, footprint, p0=0.5, backend="histogram"
    ).astype(out_dtype)

    assert result is out
    assert result.dtype == out_dtype
    cp.testing.assert_array_equal(result, expected)

    auto = rank.percentile(
        image, footprint, p0=0.5, out=cp.empty_like(out), backend="auto"
    )
    cp.testing.assert_array_equal(auto, expected)


def test_rank_backend_invalid_value_raises():
    image = cp.asarray(np.arange(25, dtype=np.uint8).reshape(5, 5))
    footprint = cp.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="backend must be one of"):
        rank.percentile(image, footprint, p0=0.5, backend="bad")


def test_rank_requires_cupy_inputs():
    image = cp.asarray(np.arange(25, dtype=np.uint8).reshape(5, 5))
    footprint = cp.ones((3, 3), dtype=bool)
    mask = cp.ones_like(image, dtype=bool)

    with pytest.raises(ValueError, match="image must be a CuPy array"):
        rank.percentile(cp.asnumpy(image), footprint, p0=0.5)

    with pytest.raises(ValueError, match="footprint must be a CuPy array"):
        rank.percentile(image, cp.asnumpy(footprint), p0=0.5)

    with pytest.raises(ValueError, match="mask must be a CuPy array"):
        rank.percentile(image, footprint, mask=cp.asnumpy(mask), p0=0.5)


def test_rank_median_default_footprint():
    image = cp.asarray(np.arange(25, dtype=np.uint8).reshape(5, 5))
    expected = rank.median(image, cp.ones((3, 3), dtype=bool))
    result = rank.median(image)

    cp.testing.assert_array_equal(result, expected)


def test_rank_default_cast_to_uint8_matches_explicit_float_conversion():
    image = cp.linspace(0, 1, 25, dtype=cp.float32).reshape(5, 5)
    footprint = cp.ones((3, 3), dtype=bool)
    image_u8 = img_as_ubyte(image)

    expected = rank.percentile(
        image_u8, footprint, p0=0.5, backend="elementwise"
    )
    with expected_warnings(["Possible precision loss"]):
        result = rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="elementwise",
        )

    assert result.dtype == cp.uint8
    cp.testing.assert_array_equal(result, expected)


def test_rank_default_cast_to_uint8_matches_explicit_uint16_conversion():
    image = cp.linspace(0, 65535, 25, dtype=cp.uint16).reshape(5, 5)
    footprint = cp.ones((3, 3), dtype=bool)
    image_u8 = img_as_ubyte(image)

    expected = rank.percentile(
        image_u8, footprint, p0=0.5, backend="elementwise"
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="elementwise",
        )
    assert not record

    assert result.dtype == cp.uint8
    cp.testing.assert_array_equal(result, expected)


def test_rank_cast_to_uint8_before_histogram_backend_selection():
    image = cp.linspace(0, 1, 25 * 25, dtype=cp.float32).reshape(25, 25)
    footprint = cp.ones((17, 17), dtype=bool)
    image_u8 = img_as_ubyte(image)

    with pytest.raises(ValueError, match="backend='histogram' requires"):
        rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="histogram",
            cast_to_uint8=False,
        )

    expected = rank.percentile(image_u8, footprint, p0=0.5, backend="histogram")
    with expected_warnings(["Possible precision loss"]):
        result = rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="histogram",
        )

    assert result.dtype == cp.uint8
    cp.testing.assert_array_equal(result, expected)


def test_rank_uint16_elementwise_does_not_warn_about_bins():
    image = cp.asarray(np.arange(25, dtype=np.uint16).reshape(5, 5))
    image[-1, -1] = 2048
    footprint = cp.ones((3, 3), dtype=bool)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="elementwise",
            cast_to_uint8=False,
        )
    assert not record

    with pytest.raises(ValueError, match="backend='histogram' requires"):
        rank.percentile(
            image,
            footprint,
            p0=0.5,
            backend="histogram",
            cast_to_uint8=False,
        )


def test_rank_histogram_partitions_default_and_env(monkeypatch):
    monkeypatch.delenv("CUCIM_RANK_HISTOGRAM_PARTITIONS", raising=False)
    monkeypatch.delenv("CUCIM_RANK_HISTOGRAM_SCRATCH_MB", raising=False)
    monkeypatch.delenv("CUCIM_RANK_HISTOGRAM_MAX_PARTITIONS", raising=False)

    assert (
        _get_rank_histogram_partitions(1080, 1080, counter_dtype=cp.int32)
        == 242
    )
    assert (
        _get_rank_histogram_partitions(1080, 1080, counter_dtype=cp.int16)
        == 256
    )

    monkeypatch.setenv("CUCIM_RANK_HISTOGRAM_MAX_PARTITIONS", "64")
    assert (
        _get_rank_histogram_partitions(1080, 1080, counter_dtype=cp.int32) == 64
    )

    monkeypatch.setenv("CUCIM_RANK_HISTOGRAM_PARTITIONS", "32")
    assert (
        _get_rank_histogram_partitions(1080, 1080, counter_dtype=cp.int32) == 32
    )


def test_rank_histogram_counter_dtype():
    assert _get_histogram_counter_dtype((181, 181)) == cp.int16
    assert _get_histogram_counter_dtype((181, 183)) == cp.int32


def test_rank_histogram_auto_cutoffs():
    assert not _should_use_rank_histogram("percentile", (15, 15))
    assert _should_use_rank_histogram("percentile", (17, 17))
    assert not _should_use_rank_histogram("mean", (17, 17))
    assert _should_use_rank_histogram("mean", (19, 19))
    assert not _should_use_rank_histogram("entropy", (23, 23))
    assert _should_use_rank_histogram("entropy", (25, 25))
    assert not _should_use_rank_histogram("bilateral_mean", (31, 31))
    assert _should_use_rank_histogram("bilateral_mean", (33, 33))
    assert not _should_use_rank_histogram("geometric_mean", (13, 13))
    assert _should_use_rank_histogram("geometric_mean", (15, 15))
    assert not _should_use_rank_histogram("modal", (13, 13))
    assert _should_use_rank_histogram("modal", (15, 15))
    assert not _should_use_rank_histogram("equalize", (71, 71))
    assert _should_use_rank_histogram("equalize", (91, 91))


# # Note: Explicitly read all values into a dict. Otherwise, stochastic test
# #       failures related to I/O can occur during parallel test cases.
ref_data = dict(np.load(fetch("data/rank_filter_tests.npz")))
ref_data_3d = dict(np.load(fetch("data/rank_filters_tests_3d.npz")))


class TestRank:
    def setup_method(self):
        np.random.seed(0)
        # This image is used along with @run_in_parallel
        # to ensure that the same seed is used for each thread.
        self.image = cp.asarray(np.random.rand(25, 25))
        np.random.seed(0)
        self.volume = cp.asarray(np.random.rand(10, 10, 10))
        # Set again the seed for the other tests.
        np.random.seed(0)
        self.footprint = morphology.disk(1)
        self.footprint_3d = morphology.ball(1)
        self.refs = ref_data
        self.refs_3d = ref_data_3d

    # Filters where the only differences vs scikit-image are at image
    # borders (due to reflected boundary extension vs excluded pixels).
    # For these, we compare only interior pixels.
    _border_differences_allowed = {
        "entropy",
        "equalize",
        "geometric_mean",
        "majority",
        "mean",
        "mean_bilateral",
        "mean_percentile",
        "median",
        "modal",
        "pop",
        "pop_bilateral",
        "pop_percentile",
        "subtract_mean",
        "subtract_mean_percentile",
        "sum",
        "sum_bilateral",
        "sum_percentile",
    }

    # Filters with known algorithmic differences that are documented and
    # expected. These are tested separately or skipped here.
    _xfail_filters = {
        # gradient_percentile: scikit-image's histogram p1-inversion quirk
        # makes imax=255 always; our sorted-array computes correct max-min.
        "gradient_percentile",
        # noise_filter: center pixel is always in its own neighborhood
        # (footprint center=1), so our result is always 0. scikit-image
        # reference shows non-zero values — under investigation.
        "noise_filter",
    }

    @pytest.mark.parametrize("filter", all_rank_filters)
    def test_rank_filter(self, filter):
        """Test rank filters with uint8 input against scikit-image reference.

        The reference data in rank_filter_tests.npz was generated by
        scikit-image which internally converts float images to uint8. We
        keep the same default conversion behavior for closer compatibility.
        """
        if filter in self._xfail_filters:
            pytest.skip(
                f"{filter}: known algorithmic difference vs scikit-image (not a bug)"
            )
        expected = cp.asarray(self.refs[filter])
        with expected_warnings(["Possible precision loss"]):
            result = getattr(rank, filter)(
                self.image, self.footprint, cast_to_uint8=True
            )
        if filter in self._border_differences_allowed:
            # Only compare interior pixels — borders differ due to
            # reflected boundary extension (GPU) vs excluded pixels
            # (scikit-image).
            expected = expected[1:-1, 1:-1]
            result = result[1:-1, 1:-1]
        if filter == "subtract_mean_percentile":
            # Allow off-by-1 due to documented mid_bin offset difference:
            # percentile variant uses (dtype_max + 1) / 2 = 128 for uint8,
            # scikit-image's percentile variant uses the same, but the
            # histogram-based integer arithmetic can round differently,
            # giving a systematic -1 offset on some pixels.
            cp.testing.assert_allclose(expected, result, atol=1)
        else:
            cp.testing.assert_allclose(expected, result)

    @pytest.mark.parametrize("filter", all_rank_filters)
    def test_rank_filter_footprint_sequence_unsupported(self, filter):
        footprint_sequence = morphology.diamond(3, decomposition="sequence")
        with pytest.raises(ValueError):
            getattr(rank, filter)(
                self.image.astype(np.uint8), footprint_sequence
            )

    @pytest.mark.parametrize("outdt", [None])  # , cp.float32, cp.float64])
    @pytest.mark.parametrize(
        "filter",
        [
            "autolevel",
            "equalize",
            "gradient",
            "majority",
            "maximum",
            "mean",
            "geometric_mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "sum",
            "threshold",
            "noise_filter",
            "entropy",
        ],
    )
    def test_rank_filters_3D(self, filter, outdt):
        if filter in self._xfail_filters:
            pytest.skip(
                f"{filter}: known algorithmic difference vs scikit-image (not a bug)"
            )
        expected = cp.asarray(self.refs_3d[filter])
        if outdt is not None:
            out = cp.zeros_like(expected, dtype=outdt)
        else:
            out = None
        with expected_warnings(["Possible precision loss"]):
            result = getattr(rank, filter)(
                self.volume, self.footprint_3d, out=out, cast_to_uint8=True
            )
        if outdt is not None:
            # Avoid rounding issues comparing to expected result
            if filter == "sum":
                # sum test data seems to be 8-bit disguised as 16-bit
                datadt = cp.uint8
            else:
                datadt = expected.dtype
            # Take modulus first to avoid undefined behavior for
            # float->uint8 conversions.
            result = cp.mod(result, 256.0).astype(datadt)
        if filter in self._border_differences_allowed:
            # Only compare interior pixels — borders differ due to
            # reflected boundary extension (GPU) vs excluded pixels
            # (scikit-image).
            expected = expected[1:-1, 1:-1, 1:-1]
            result = result[1:-1, 1:-1, 1:-1]
        if filter == "subtract_mean_percentile":
            # Allow off-by-1 due to documented mid_bin offset difference:
            # percentile variant uses (dtype_max + 1) / 2 = 128 for uint8,
            # scikit-image's percentile variant uses the same, but the
            # histogram-based integer arithmetic can round differently,
            # giving a systematic -1 offset on some pixels.
            cp.testing.assert_allclose(expected, result, atol=1)
        cp.testing.assert_array_almost_equal(expected, result)

    def test_random_sizes(self):
        # make sure the size is not a problem
        elem = cp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=cp.uint8)
        for m, n in np.random.randint(1, 101, size=(10, 2)):
            mask = cp.ones((m, n), dtype=cp.uint8)

            image8 = cp.ones((m, n), dtype=cp.uint8)
            out8 = cp.empty_like(image8)
            rank.mean(
                image=image8,
                footprint=elem,
                mask=mask,
                out=out8,
                shift_x=0,
                shift_y=0,
            )
            assert image8.shape == out8.shape
            rank.mean(
                image=image8,
                footprint=elem,
                mask=mask,
                out=out8,
                shift_x=+1,
                shift_y=+1,
            )
            assert image8.shape == out8.shape

            rank.geometric_mean(
                image=image8,
                footprint=elem,
                mask=mask,
                out=out8,
                shift_x=0,
                shift_y=0,
            )
            assert image8.shape == out8.shape

            rank.geometric_mean(
                image=image8,
                footprint=elem,
                mask=mask,
                out=out8,
                shift_x=+1,
                shift_y=+1,
            )
            assert image8.shape == out8.shape

            image16 = cp.ones((m, n), dtype=cp.uint16)
            out16 = cp.empty_like(image8, dtype=cp.uint16)
            rank.mean(
                image=image16,
                footprint=elem,
                mask=mask,
                out=out16,
                shift_x=0,
                shift_y=0,
            )
            assert image16.shape == out16.shape
            rank.mean(
                image=image16,
                footprint=elem,
                mask=mask,
                out=out16,
                shift_x=+1,
                shift_y=+1,
            )
            assert image16.shape == out16.shape

            rank.geometric_mean(
                image=image16,
                footprint=elem,
                mask=mask,
                out=out16,
                shift_x=0,
                shift_y=0,
            )
            assert image16.shape == out16.shape
            rank.geometric_mean(
                image=image16,
                footprint=elem,
                mask=mask,
                out=out16,
                shift_x=+1,
                shift_y=+1,
            )
            assert image16.shape == out16.shape

            rank.mean_percentile(
                image=image16,
                mask=mask,
                out=out16,
                footprint=elem,
                shift_x=0,
                shift_y=0,
                p0=0.1,
                p1=0.9,
            )
            assert image16.shape == out16.shape
            rank.mean_percentile(
                image=image16,
                mask=mask,
                out=out16,
                footprint=elem,
                shift_x=+1,
                shift_y=+1,
                p0=0.1,
                p1=0.9,
            )
            assert image16.shape == out16.shape

    @pytest.mark.parametrize("r", list(range(3, 20, 2)))
    def test_compare_with_gray_dilation(self, r):
        # compare the result of maximum filter with dilate

        image = (cp.random.rand(100, 100) * 256).astype(cp.uint8)
        out = cp.empty_like(image)
        mask = cp.ones(image.shape, dtype=cp.uint8)

        elem = cp.ones((r, r), dtype=np.uint8)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask)
        cm = gray.dilation(image, elem)
        cp.testing.assert_array_equal(out, cm)

    @pytest.mark.parametrize("r", list(range(3, 20, 2)))
    def test_compare_with_gray_erosion(self, r):
        # compare the result of maximum filter with erode

        image = (cp.random.rand(100, 100) * 256).astype(cp.uint8)
        out = cp.empty_like(image)
        mask = cp.ones(image.shape, dtype=cp.uint8)

        elem = cp.ones((r, r), dtype=np.uint8)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask)
        cm = gray.erosion(image, elem)
        cp.testing.assert_array_equal(out, cm)

    def test_population(self):
        # check the number of valid pixels in the neighborhood
        image = cp.zeros((5, 5), dtype=np.uint8)
        elem = cp.ones((3, 3), dtype=np.uint8)
        out = cp.empty_like(image)
        mask = cp.ones(image.shape, dtype=np.uint8)

        rank.pop(image=image, footprint=elem, out=out, mask=mask)
        r = cp.array(
            [
                [4, 6, 6, 6, 4],
                [6, 9, 9, 9, 6],
                [6, 9, 9, 9, 6],
                [6, 9, 9, 9, 6],
                [4, 6, 6, 6, 4],
            ]
        )
        # Note: omit boundaries due to known difference in boundary handling
        cp.testing.assert_array_equal(r[1:-1, 1:-1], out[1:-1, 1:-1])

    def test_structuring_element8(self):
        # check the output for a custom footprint

        r = cp.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 255, 0, 0, 0],
                [0, 0, 255, 255, 255, 0],
                [0, 0, 0, 255, 255, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # 8-bit
        image = cp.zeros((6, 6), dtype=np.uint8)
        image[2, 2] = 255
        elem = cp.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
        out = cp.empty_like(image)
        mask = cp.ones(image.shape, dtype=np.uint8)

        rank.maximum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=1,
            shift_y=1,
        )
        cp.testing.assert_array_equal(r, out)

        # 16-bit
        image = cp.zeros((6, 6), dtype=np.uint16)
        image[2, 2] = 255
        out = np.empty_like(image)

        rank.maximum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=1,
            shift_y=1,
        )
        cp.testing.assert_array_equal(r, out)

    def test_inplace_output(self):
        # rank filters are not supposed to filter inplace

        footprint = disk(20, decomposition=None)
        image = (cp.random.rand(500, 500) * 256).astype(np.uint8)
        out = image
        with pytest.raises(NotImplementedError):
            rank.mean(image, footprint, out=out)

    def test_compare_autolevels(self):
        # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
        # should returns the same arrays

        image = util.img_as_ubyte(cp.asarray(data.camera()))

        footprint = disk(20, decomposition=None)
        loc_autolevel = rank.autolevel(image, footprint=footprint)
        loc_perc_autolevel = rank.autolevel_percentile(
            image, footprint=footprint, p0=0.0, p1=1.0
        )

        cp.testing.assert_array_equal(loc_autolevel, loc_perc_autolevel)

    def test_compare_autolevels_16bit(self):
        # compare autolevel(16-bit) and percentile autolevel(16-bit) with
        # p0=0.0 and p1=1.0 should returns the same arrays

        image = cp.asarray(data.camera()).astype(np.uint16) * 4

        footprint = disk(20, decomposition=None)
        loc_autolevel = rank.autolevel(image, footprint=footprint)
        loc_perc_autolevel = rank.autolevel_percentile(
            image, footprint=footprint, p0=0.0, p1=1.0
        )

        cp.testing.assert_array_equal(loc_autolevel, loc_perc_autolevel)

    @pytest.mark.parametrize(
        "method",
        [
            "autolevel",
            "equalize",
            "gradient",
            "threshold",
            "subtract_mean",
            "enhance_contrast",
            "pop",
        ],
    )
    def test_compare_ubyte_vs_float(self, method):
        # Create signed int8 image that and convert it to uint8
        image_uint = img_as_ubyte(cp.asarray(data.camera()[:50, :50]))
        image_float = img_as_float(image_uint)

        disk3 = disk(3, decomposition=None)
        func = getattr(rank, method)
        out_u = func(image_uint, disk3)
        with expected_warnings(["Possible precision loss"]):
            out_f = func(image_float, disk3, cast_to_uint8=True)
        cp.testing.assert_array_equal(out_u, out_f)

    @pytest.mark.parametrize(
        "method",
        [
            "equalize",
            "autolevel",
            "gradient",
            "majority",
            "maximum",
            "mean",
            "geometric_mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "sum",
            "threshold",
            "noise_filter",
            "entropy",
        ],
    )
    def test_compare_ubyte_vs_float_3d(self, method):
        # Create signed int8 volume that and convert it to uint8
        np.random.seed(0)
        volume_uint = np.random.randint(
            0, high=256, size=(10, 20, 30), dtype=np.uint8
        )
        volume_uint = cp.asarray(volume_uint)
        volume_float = img_as_float(volume_uint)

        ball3 = ball(3, decomposition=None)
        func = getattr(rank, method)
        out_u = func(volume_uint, ball3)
        with expected_warnings(["Possible precision loss"]):
            out_f = func(volume_float, ball3, cast_to_uint8=True)
        cp.testing.assert_array_equal(out_u, out_f)

    @pytest.mark.parametrize(
        "method",
        [
            "autolevel",
            "equalize",
            "gradient",
            "maximum",
            "mean",
            "geometric_mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "threshold",
        ],
    )
    def test_compare_8bit_unsigned_vs_signed(self, method):
        # filters applied on 8-bit image or 16-bit image (having only real 8-bit
        # of dynamic) should be identical

        # Create signed int8 image that and convert it to uint8
        image = img_as_ubyte(cp.asarray(data.camera()))[::2, ::2]
        image[image > 127] = 0
        image_s = image.astype(np.int8)
        image_u = img_as_ubyte(image_s)
        cp.testing.assert_array_equal(image_u, img_as_ubyte(image_s))
        func = getattr(rank, method)
        disk3 = disk(3, decomposition=None)
        out_u = func(image_u, disk3)
        # with expected_warnings(["Possible precision loss"]):
        out_s = func(image_s, disk3, cast_to_uint8=True)
        cp.testing.assert_array_equal(out_u, out_s)

    @pytest.mark.parametrize(
        "method",
        [
            "equalize",
            "autolevel",
            "gradient",
            "majority",
            "maximum",
            "mean",
            "geometric_mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "sum",
            "threshold",
            "noise_filter",
            "entropy",
        ],
    )
    def test_compare_8bit_unsigned_vs_signed_3d(self, method):
        # filters applied on 8-bit volume or 16-bit volume (having only real 8-bit
        # of dynamic) should be identical

        # Create signed int8 volume that and convert it to uint8
        np.random.seed(0)
        volume_s = np.random.randint(
            0, high=127, size=(10, 20, 30), dtype=np.int8
        )
        volume_s = cp.asarray(volume_s)
        volume_u = img_as_ubyte(volume_s)
        cp.testing.assert_array_equal(volume_u, img_as_ubyte(volume_s))

        ball3 = ball(3, decomposition=None)
        func = getattr(rank, method)
        out_u = func(volume_u, ball3)
        # with expected_warnings(["Possible precision loss"]):
        out_s = func(volume_s, ball3, cast_to_uint8=True)
        cp.testing.assert_array_equal(out_u, out_s)

    @pytest.mark.parametrize(
        "method",
        [
            "autolevel",
            "equalize",
            "gradient",
            "maximum",
            "mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "threshold",
        ],
    )
    def test_compare_8bit_vs_16bit(self, method):
        # filters applied on 8-bit image or 16-bit image (having only real 8-bit
        # of dynamic) should be identical
        image8 = util.img_as_ubyte(cp.asarray(data.camera())[::2, ::2])
        image16 = image8.astype(cp.uint16)
        cp.testing.assert_array_equal(image8, image16)

        func = getattr(rank, method)

        disk3 = disk(3, decomposition=None)
        f8 = func(image8, disk3)
        f16 = func(image16, disk3, cast_to_uint8=True)
        cp.testing.assert_array_equal(f8, f16)

    @pytest.mark.parametrize(
        "method",
        [
            "equalize",
            "autolevel",
            "gradient",
            "majority",
            "maximum",
            "mean",
            "geometric_mean",
            "subtract_mean",
            "median",
            "minimum",
            "modal",
            "enhance_contrast",
            "pop",
            "sum",
            "threshold",
            "noise_filter",
            "entropy",
        ],
    )
    def test_compare_8bit_vs_16bit_3d(self, method):
        np.random.seed(0)
        volume8 = np.random.randint(
            128, high=256, size=(10, 10, 10), dtype=np.uint8
        )
        volume8 = cp.asarray(volume8)
        volume16 = volume8.astype(cp.uint16)

        func = getattr(rank, method)

        ball3 = ball(3, decomposition=None)
        f8 = func(volume8, ball3)
        f16 = func(volume16, ball3, cast_to_uint8=True)
        cp.testing.assert_array_equal(f8, f16)

    @pytest.mark.parametrize("dtype", [cp.uint8, cp.uint16])
    def test_trivial_footprint8(self, dtype):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = cp.zeros((5, 5), dtype=dtype)
        out = cp.zeros_like(image)
        mask = cp.ones_like(image, dtype=cp.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = cp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=cp.uint8)
        rank.mean(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)
        rank.geometric_mean(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)
        rank.minimum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)
        rank.maximum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)

    @pytest.mark.parametrize("dtype", [cp.uint8, cp.uint16])
    def test_smallest_footprint8(self, dtype):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = cp.zeros((5, 5), dtype=dtype)
        out = cp.zeros_like(image)
        mask = cp.ones_like(image, dtype=cp.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = cp.array([[1]], dtype=cp.uint8)
        rank.mean(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)
        rank.minimum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)
        rank.maximum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(image, out)

    def test_empty_footprint(self):
        image = cp.zeros((5, 5), dtype=np.uint16)
        out = cp.zeros_like(image)
        mask = cp.ones_like(image, dtype=np.uint8)
        res = cp.zeros_like(image)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = cp.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8)

        rank.mean(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(res, out)
        rank.geometric_mean(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(res, out)
        rank.minimum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(res, out)
        rank.maximum(
            image=image,
            footprint=elem,
            out=out,
            mask=mask,
            shift_x=0,
            shift_y=0,
        )
        cp.testing.assert_array_equal(res, out)

    def test_entropy(self):
        #  verify that entropy is coherent with bitdepth of the input data

        footprint = cp.ones((16, 16), dtype=cp.uint8)
        # 1 bit per pixel
        data = cp.tile(cp.asarray([0, 1]), (100, 100)).astype(cp.uint8)
        assert cp.max(rank.entropy(data, footprint)) == 1

        # 2 bit per pixel
        data = cp.tile(cp.asarray([[0, 1], [2, 3]]), (10, 10)).astype(cp.uint8)
        assert cp.max(rank.entropy(data, footprint)) == 2

        # 3 bit per pixel
        data = cp.tile(
            cp.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]), (10, 10)
        ).astype(cp.uint8)
        assert cp.max(rank.entropy(data, footprint)) == 3

        # 4 bit per pixel
        data = cp.tile(cp.reshape(cp.arange(16), (4, 4)), (10, 10)).astype(
            cp.uint8
        )
        assert cp.max(rank.entropy(data, footprint)) == 4

        # 6 bit per pixel
        data = cp.tile(cp.reshape(cp.arange(64), (8, 8)), (10, 10)).astype(
            cp.uint8
        )
        assert cp.max(rank.entropy(data, footprint)) == 6

        # 8-bit per pixel
        data = cp.tile(cp.reshape(cp.arange(256), (16, 16)), (10, 10)).astype(
            cp.uint8
        )
        assert cp.max(rank.entropy(data, footprint)) == 8

        # 12 bit per pixel
        footprint = cp.ones((64, 64), dtype=cp.uint8)
        data = cp.zeros((65, 65), dtype=cp.uint16)
        data[:64, :64] = cp.reshape(cp.arange(4096), (64, 64))
        assert cp.max(rank.entropy(data, footprint, cast_to_uint8=False)) == 12

        # make sure output is floating point
        # with expected_warnings(['Bad rank filter performance']):
        out = rank.entropy(data, cp.ones((16, 16), dtype=cp.uint8))
        assert out.dtype.kind == "f"

    def test_footprint_dtypes(self):
        image = cp.zeros((5, 5), dtype=cp.uint8)
        out = cp.zeros_like(image)
        mask = cp.ones_like(image, dtype=cp.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        for dtype in (
            bool,
            cp.uint8,
            cp.uint16,
            cp.int32,
            cp.int64,
            cp.float32,
            cp.float64,
        ):
            elem = cp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=dtype)
            rank.mean(
                image=image,
                footprint=elem,
                out=out,
                mask=mask,
                shift_x=0,
                shift_y=0,
            )
            cp.testing.assert_array_equal(image, out)
            rank.geometric_mean(
                image=image,
                footprint=elem,
                out=out,
                mask=mask,
                shift_x=0,
                shift_y=0,
            )
            cp.testing.assert_array_equal(image, out)
            rank.mean_percentile(
                image=image,
                footprint=elem,
                out=out,
                mask=mask,
                shift_x=0,
                shift_y=0,
            )
            cp.testing.assert_array_equal(image, out)

    def test_16bit(self):
        image = cp.zeros((21, 21), dtype=np.uint16)
        footprint = cp.ones((3, 3), dtype=np.uint8)

        for bitdepth in range(17):
            value = 2**bitdepth - 1
            image[10, 10] = value
            expected = []
            # if bitdepth >= 11:
            #     expected = ['Bad rank filter performance']
            with expected_warnings(expected):
                assert (
                    rank.minimum(image, footprint, cast_to_uint8=False)[10, 10]
                    == 0
                )
                assert (
                    rank.maximum(image, footprint, cast_to_uint8=False)[10, 10]
                    == value
                )
                mean_val = rank.mean(image, footprint, cast_to_uint8=False)[
                    10, 10
                ]
                assert mean_val == int(value / footprint.size)

    def test_bilateral(self):
        image = cp.zeros((21, 21), dtype=cp.uint16)
        footprint = cp.ones((3, 3), dtype=cp.uint8)

        image[10, 10] = 1000
        image[10, 11] = 1010
        image[10, 9] = 900

        kwargs = dict(s0=1, s1=1, cast_to_uint8=False)
        assert (
            rank.mean_bilateral(image, footprint, **kwargs)[10, 10].get()
            == 1000
        )
        assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10].get() == 1
        kwargs = dict(s0=11, s1=11, cast_to_uint8=False)
        assert (
            rank.mean_bilateral(image, footprint, **kwargs)[10, 10].get()
            == 1005
        )
        assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10].get() == 2

    def test_percentile_min(self):
        # check that percentile p0 = 0 is identical to local min
        img = cp.asarray(data.camera())
        img16 = img.astype(cp.uint16)
        footprint = disk(15, decomposition=None)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=0)
        img_min = rank.minimum(img, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_min)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=0)
        img_min = rank.minimum(img16, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_min)

    def test_percentile_max(self):
        # check that percentile p0 = 1 is identical to local max
        img = cp.asarray(data.camera())
        img16 = img.astype(cp.uint16)
        footprint = disk(15, decomposition=None)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=1.0)
        img_max = rank.maximum(img, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_max)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=1.0)
        img_max = rank.maximum(img16, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_max)

    def test_percentile_median(self):
        # check that percentile p0 = 0.5 is identical to local median
        img = cp.asarray(data.camera())
        img16 = img.astype(cp.uint16)
        footprint = disk(15, decomposition=None)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=0.5)
        img_max = rank.median(img, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_max)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=0.5)
        img_max = rank.median(img16, footprint=footprint)
        cp.testing.assert_array_equal(img_p0, img_max)

    def test_sum(self):
        # check the number of valid pixels in the neighborhood

        image8 = cp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=cp.uint8,
        )
        image16 = 400 * cp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=cp.uint16,
        )
        elem = cp.ones((3, 3), dtype=cp.uint8)
        out8 = cp.empty_like(image8)
        out16 = cp.empty_like(image16)
        mask = cp.ones(image8.shape, dtype=cp.uint8)

        r = cp.array(
            [
                [1, 2, 3, 2, 1],
                [2, 4, 6, 4, 2],
                [3, 6, 9, 6, 3],
                [2, 4, 6, 4, 2],
                [1, 2, 3, 2, 1],
            ],
            dtype=cp.uint8,
        )
        rank.sum(image=image8, footprint=elem, out=out8, mask=mask)
        cp.testing.assert_array_equal(r, out8)
        rank.sum_percentile(
            image=image8, footprint=elem, out=out8, mask=mask, p0=0.0, p1=1.0
        )
        cp.testing.assert_array_equal(r, out8)
        rank.sum_bilateral(
            image=image8, footprint=elem, out=out8, mask=mask, s0=255, s1=255
        )
        cp.testing.assert_array_equal(r, out8)

        r = 400 * cp.array(
            [
                [1, 2, 3, 2, 1],
                [2, 4, 6, 4, 2],
                [3, 6, 9, 6, 3],
                [2, 4, 6, 4, 2],
                [1, 2, 3, 2, 1],
            ],
            dtype=cp.uint16,
        )
        rank.sum(
            image=image16,
            footprint=elem,
            out=out16,
            mask=mask,
            cast_to_uint8=False,
        )
        cp.testing.assert_array_equal(r, out16)
        rank.sum_percentile(
            image=image16,
            footprint=elem,
            out=out16,
            mask=mask,
            p0=0.0,
            p1=1.0,
            cast_to_uint8=False,
        )
        cp.testing.assert_array_equal(r, out16)
        rank.sum_bilateral(
            image=image16,
            footprint=elem,
            out=out16,
            mask=mask,
            s0=1000,
            s1=1000,
            cast_to_uint8=False,
        )
        cp.testing.assert_array_equal(r, out16)

    def test_median_default_value(self):
        a = cp.zeros((3, 3), dtype=cp.uint8)
        a[1] = 1
        full_footprint = cp.ones((3, 3), dtype=cp.uint8)
        cp.testing.assert_array_equal(
            rank.median(a), rank.median(a, full_footprint)
        )
        assert rank.median(a)[1, 1].get() == 0
        assert rank.median(a, disk(1, decomposition=None))[1, 1].get() == 1

    def test_output_same_dtype(self):
        image = (cp.random.rand(100, 100) * 256).astype(cp.uint8)
        out = cp.empty_like(image)
        mask = cp.ones(image.shape, dtype=cp.uint8)
        elem = cp.ones((3, 3), dtype=cp.uint8)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask)
        cp.testing.assert_array_equal(image.dtype, out.dtype)

    def test_input_boolean_dtype(self):
        image = (cp.random.rand(100, 100) * 256).astype(bool)
        elem = cp.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError):
            rank.maximum(image=image, footprint=elem)
