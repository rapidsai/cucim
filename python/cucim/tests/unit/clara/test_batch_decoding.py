#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for batch ROI decoding using nvImageCodec v0.7.0+ API.

These tests verify that the ThreadBatchDataLoader + NvImageCodecProcessor
correctly batch-decodes multiple ROI regions from TIFF files.
"""

import numpy as np
import pytest

from ...util.io import open_image_cucim

# Skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


class TestBatchDecoding:
    """Test batch decoding functionality."""

    def test_batch_read_multiple_locations(
        self, testimg_tiff_stripe_4096x4096_256
    ):
        """Test reading multiple locations in a single call."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            # Define multiple locations
            locations = [
                (0, 0),
                (256, 256),
                (512, 512),
                (768, 768),
            ]
            size = (256, 256)
            level = 0

            # First, get reference results using single-image decoding (no batching)
            reference_results = []
            for loc in locations:
                ref = img.read_region(loc, size, level)
                reference_results.append(np.asarray(ref).copy())

            # Read with multiple workers (triggers batch decoding path)
            gen = img.read_region(locations, size, level, num_workers=2)

            results = list(gen)
            assert len(results) == len(locations)

            for i, result in enumerate(results):
                arr = np.asarray(result)
                assert arr.shape == (256, 256, 3), f"Region {i} has wrong shape"
                # Verify batch result matches single-image decoding
                np.testing.assert_array_equal(
                    arr, reference_results[i],
                    err_msg=f"Region {i} batch result differs from single decode"
                )

    def test_batch_read_with_batch_size(
        self, testimg_tiff_stripe_4096x4096_256
    ):
        """Test batch reading with explicit batch_size parameter."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            locations = [
                (0, 0),
                (128, 128),
                (256, 256),
                (384, 384),
                (512, 512),
                (640, 640),
            ]
            size = (128, 128)
            level = 0
            batch_size = 2

            gen = img.read_region(
                locations, size, level, batch_size=batch_size, num_workers=2
            )

            batch_count = 0
            for batch in gen:
                arr = np.asarray(batch)
                # Batch should have shape [batch_size, H, W, C]
                if batch_count < len(locations) // batch_size:
                    assert arr.shape[0] == batch_size
                batch_count += 1

            expected_batches = (len(locations) + batch_size - 1) // batch_size
            assert batch_count == expected_batches

    def test_batch_read_with_prefetch(self, testimg_tiff_stripe_4096x4096_256):
        """Test batch reading with prefetch_factor."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            locations = [(i * 64, i * 64) for i in range(16)]
            size = (64, 64)
            level = 0

            gen = img.read_region(
                locations, size, level, num_workers=4, prefetch_factor=4
            )

            results = list(gen)
            assert len(results) == len(locations)

    def test_batch_read_shuffle(self, testimg_tiff_stripe_4096x4096_256):
        """Test batch reading with shuffle enabled."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            locations = [(i * 100, i * 100) for i in range(10)]
            size = (100, 100)
            level = 0

            # Read without shuffle
            gen1 = img.read_region(
                locations.copy(), size, level, num_workers=2, shuffle=False
            )
            results1 = [np.asarray(r).copy() for r in gen1]

            # Read with shuffle (same seed for reproducibility)
            gen2 = img.read_region(
                locations.copy(),
                size,
                level,
                num_workers=2,
                shuffle=True,
                seed=42,
            )
            results2 = [np.asarray(r).copy() for r in gen2]

            assert len(results1) == len(results2)

    def test_batch_read_drop_last(self, testimg_tiff_stripe_4096x4096_256):
        """Test batch reading with drop_last=True."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            # 5 locations with batch_size=2 should give 2 batches (drop last 1)
            locations = [(i * 100, i * 100) for i in range(5)]
            size = (100, 100)
            level = 0
            batch_size = 2

            gen = img.read_region(
                locations,
                size,
                level,
                batch_size=batch_size,
                num_workers=2,
                drop_last=True,
            )

            results = list(gen)
            # With drop_last, we should have floor(5/2) = 2 batches
            assert len(results) == 2

    def test_batch_read_boundary_regions(
        self, testimg_tiff_stripe_4096x4096_256
    ):
        """Test batch reading regions at image boundaries."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            width, height = img.size("XY")

            # Regions at corners and edges
            locations = [
                (0, 0),  # top-left
                (width - 256, 0),  # top-right
                (0, height - 256),  # bottom-left
                (width - 256, height - 256),  # bottom-right
            ]
            size = (256, 256)
            level = 0

            gen = img.read_region(locations, size, level, num_workers=2)
            results = list(gen)

            assert len(results) == len(locations)
            for r in results:
                arr = np.asarray(r)
                assert arr.shape == (256, 256, 3)

    def test_batch_read_multiresolution(
        self, testimg_tiff_stripe_4096x4096_256
    ):
        """Test batch reading at different resolution levels."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            level_count = img.resolutions["level_count"]

            for level in range(min(3, level_count)):
                locations = [(i * 50, i * 50) for i in range(4)]
                size = (128, 128)

                gen = img.read_region(locations, size, level, num_workers=2)
                results = list(gen)

                assert len(results) == len(locations), (
                    f"Failed at level {level}"
                )


class TestBatchDecodingCUDA:
    """Test batch decoding with CUDA output.

    Note: CUDA output is only supported for certain compression types (JPEG).
    Deflate and other compression types may not support direct CUDA decoding.
    """

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip("No CUDA device available")
        except ImportError:
            pytest.skip("CuPy not installed")

    def test_batch_read_cuda_output(
        self, testimg_tiff_stripe_4096x4096_256_jpeg
    ):
        """Test batch reading with CUDA output device.

        Note: Only JPEG-compressed images support CUDA output in nvImageCodec.
        """
        import cupy as cp

        with open_image_cucim(testimg_tiff_stripe_4096x4096_256_jpeg) as img:
            locations = [
                (0, 0),
                (256, 256),
                (512, 512),
            ]
            size = (256, 256)
            level = 0

            gen = img.read_region(
                locations, size, level, num_workers=2, device="cuda"
            )

            for result in gen:
                # Should have CUDA array interface
                assert hasattr(result, "__cuda_array_interface__")
                arr = cp.asarray(result)
                assert arr.shape == (256, 256, 3)
                # Verify data on GPU
                assert arr.device.id >= 0

    def test_batch_read_cuda_memory_cleanup(
        self, testimg_tiff_stripe_4096x4096_256_jpeg
    ):
        """Test that CUDA memory is properly cleaned up after batch read.

        Note: Only JPEG-compressed images support CUDA output in nvImageCodec.
        """
        import cupy as cp

        # Get initial memory usage
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        initial_used = mempool.used_bytes()

        with open_image_cucim(testimg_tiff_stripe_4096x4096_256_jpeg) as img:
            locations = [(i * 100, i * 100) for i in range(10)]
            size = (256, 256)

            for _ in range(5):
                gen = img.read_region(
                    locations, size, 0, num_workers=2, device="cuda"
                )
                for result in gen:
                    _ = cp.asarray(result)

                # Free memory after each iteration
                mempool.free_all_blocks()

        # Memory should be back to initial level (within tolerance)
        mempool.free_all_blocks()
        final_used = mempool.used_bytes()
        # Allow some overhead (1MB)
        assert final_used - initial_used < 1024 * 1024


class TestBatchDecodingPerformance:
    """Performance-related tests for batch decoding."""

    @pytest.mark.parametrize("num_locations", [1, 4, 16, 64])
    def test_scaling_with_locations(
        self, testimg_tiff_stripe_4096x4096_256, num_locations
    ):
        """Test that batch decoding works with varying number of locations."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            width, height = img.size("XY")

            # Generate random locations within bounds
            np.random.seed(42)
            locations = [
                (
                    np.random.randint(0, max(1, width - 128)),
                    np.random.randint(0, max(1, height - 128)),
                )
                for _ in range(num_locations)
            ]
            size = (128, 128)

            gen = img.read_region(
                locations, size, 0, num_workers=min(4, num_locations)
            )
            results = list(gen)

            assert len(results) == num_locations

    @pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
    def test_scaling_with_workers(
        self, testimg_tiff_stripe_4096x4096_256, num_workers
    ):
        """Test batch decoding with different worker counts."""
        with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as img:
            locations = [(i * 64, i * 64) for i in range(32)]
            size = (64, 64)

            gen = img.read_region(locations, size, 0, num_workers=num_workers)
            results = list(gen)

            assert len(results) == len(locations)
