#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script for cuslide2 plugin with Aperio SVS files
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from test_common import setup_environment, test_tile_level_caching


def _generate_tile_locations(level_dimensions, tile_w=256, tile_h=256):
    """Generate a grid of tile locations for batch testing."""
    level_dims = level_dimensions[0]
    max_x = max(0, level_dims[0] - tile_w)
    max_y = max(0, level_dims[1] - tile_h)

    locations = []
    for y_off in range(0, min(max_y + 1, tile_h * 4), tile_h):
        for x_off in range(0, min(max_x + 1, tile_w * 4), tile_w):
            locations.append([x_off, y_off])
    return locations


def test_batch_decode_correctness(img, level_dimensions):
    """Verify that GPU batch-decoded tiles are pixel-identical to CPU-decoded tiles.

    This is the critical correctness test for the NvImageCodecProcessor →
    ThreadBatchDataLoader handoff.  Each GPU tile is compared against the
    corresponding CPU tile at the same location.

    Raises:
        RuntimeError: If any GPU tile does not match the corresponding CPU tile.
    """
    import cupy as cp

    print("\n🔍 Batch decode correctness test")
    print("-" * 50)

    tile_w, tile_h = 256, 256
    locations = _generate_tile_locations(level_dimensions, tile_w, tile_h)

    if len(locations) < 2:
        print("  ⚠️  Image too small for batch correctness test")
        return  # Not a failure

    batch_size = min(len(locations), 8)
    print(
        f"  Locations: {len(locations)}, tile: {tile_w}x{tile_h}, batch_size: {batch_size}"
    )

    # Decode all tiles via GPU batch path
    gpu_tiles = list(
        img.read_region(
            location=locations,
            size=[tile_w, tile_h],
            level=0,
            device="cuda",
            batch_size=batch_size,
            num_workers=1,
        )
    )

    # Decode all tiles via CPU batch path (ground truth)
    cpu_tiles = list(
        img.read_region(
            location=locations,
            size=[tile_w, tile_h],
            level=0,
            device="cpu",
            batch_size=batch_size,
            num_workers=1,
        )
    )

    if len(gpu_tiles) != len(cpu_tiles):
        raise RuntimeError(
            f"Tile count mismatch: GPU={len(gpu_tiles)}, CPU={len(cpu_tiles)}"
        )

    mismatch_count = 0
    for idx, (gpu_t, cpu_t) in enumerate(zip(gpu_tiles, cpu_tiles)):
        gpu_np = cp.asnumpy(cp.asarray(gpu_t))
        cpu_np = np.asarray(cpu_t)
        if gpu_np.shape != cpu_np.shape:
            print(
                f"    ❌ Tile {idx} at {locations[idx]}: shape mismatch GPU={gpu_np.shape} vs CPU={cpu_np.shape}"
            )
            mismatch_count += 1
        elif not np.array_equal(gpu_np, cpu_np):
            max_diff = int(np.max(np.abs(gpu_np.astype(int) - cpu_np.astype(int))))
            print(
                f"    ❌ Tile {idx} at {locations[idx]}: pixel mismatch (max diff={max_diff})"
            )
            mismatch_count += 1

    if mismatch_count == 0:
        print(f"  ✅ All {len(gpu_tiles)} tiles match (GPU == CPU)")
    else:
        raise RuntimeError(
            f"{mismatch_count}/{len(gpu_tiles)} tiles have GPU/CPU pixel mismatches"
        )


def test_batch_decode_performance(img, level_dimensions):
    """Benchmark GPU vs CPU batch decode throughput.

    This is a timing smoke test — it measures wall-clock time for batch
    iteration and reports the GPU/CPU speedup ratio.
    """
    print("\n⏱️  Batch decode performance test")
    print("-" * 50)

    tile_w, tile_h = 256, 256
    locations = _generate_tile_locations(level_dimensions, tile_w, tile_h)

    if len(locations) < 2:
        print("  ⚠️  Image too small for batch performance test")
        return

    batch_size = min(len(locations), 8)
    print(
        f"  Locations: {len(locations)}, tile: {tile_w}x{tile_h}, batch_size: {batch_size}"
    )

    # GPU batch decode
    start = time.time()
    gpu_tiles = list(
        img.read_region(
            location=locations,
            size=[tile_w, tile_h],
            level=0,
            device="cuda",
            batch_size=batch_size,
            num_workers=1,
        )
    )
    gpu_batch_time = time.time() - start
    print(f"  GPU batch: {gpu_batch_time:.4f}s ({len(gpu_tiles)} tiles)")

    # CPU batch decode
    start = time.time()
    cpu_tiles = list(
        img.read_region(
            location=locations,
            size=[tile_w, tile_h],
            level=0,
            device="cpu",
            batch_size=batch_size,
            num_workers=1,
        )
    )
    cpu_batch_time = time.time() - start
    print(f"  CPU batch: {cpu_batch_time:.4f}s ({len(cpu_tiles)} tiles)")

    if gpu_batch_time > 0:
        speedup = cpu_batch_time / gpu_batch_time
        print(f"  🎯 Batch speedup: {speedup:.2f}x")

    # Single-region comparison (512x512)
    print("\n  Single-region decode (512x512):")
    start = time.time()
    _ = img.read_region([0, 0], [512, 512], 0, device="cuda")
    gpu_single = time.time() - start

    start = time.time()
    _ = img.read_region([0, 0], [512, 512], 0, device="cpu")
    cpu_single = time.time() - start

    print(f"    GPU: {gpu_single:.4f}s, CPU: {cpu_single:.4f}s", end="")
    if gpu_single > 0:
        print(f" ({cpu_single / gpu_single:.2f}x)")
    else:
        print()

    # Larger region (2048x2048)
    print("  Single-region decode (2048x2048):")
    try:
        start = time.time()
        _ = img.read_region([0, 0], [2048, 2048], 0, device="cuda")
        gpu_large = time.time() - start

        start = time.time()
        _ = img.read_region([0, 0], [2048, 2048], 0, device="cpu")
        cpu_large = time.time() - start

        print(f"    GPU: {gpu_large:.4f}s, CPU: {cpu_large:.4f}s", end="")
        if gpu_large > 0:
            print(f" ({cpu_large / gpu_large:.2f}x)")
        else:
            print()
    except Exception as e:
        print(f"    ⚠️  Failed: {e}")


def test_aperio_svs(svs_path, plugin_lib):
    """Test cuslide2 plugin with an Aperio SVS file.

    Raises:
        FileNotFoundError: If *svs_path* does not exist.
        RuntimeError: If the correctness test detects GPU/CPU pixel mismatches.
    """

    print("\n🔬 Testing cuslide2 plugin with Aperio SVS")
    print("=" * 60)
    print(f"📁 File: {svs_path}")

    if not Path(svs_path).exists():
        raise FileNotFoundError(f"SVS file not found: {svs_path}")

    from cucim.clara import _set_plugin_root

    _set_plugin_root(str(plugin_lib))
    print(f"✅ Plugin root set: {plugin_lib}")

    from cucim import CuImage

    print("\n📂 Loading SVS file...")
    start = time.time()
    img = CuImage(svs_path)
    load_time = time.time() - start

    print(f"✅ Loaded in {load_time:.3f}s")

    # Show basic info
    print("\n📊 Image Information:")
    print(f"  Dimensions: {img.shape}")
    level_count = img.resolutions["level_count"]
    print(f"  Levels: {level_count}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Device: {img.device}")

    # Show all levels
    print("\n🔍 Resolution Levels:")
    level_dimensions = img.resolutions["level_dimensions"]
    level_downsamples = img.resolutions["level_downsamples"]
    for level in range(level_count):
        level_dims = level_dimensions[level]
        level_downsample = level_downsamples[level]
        print(
            f"  Level {level}: {level_dims[0]}x{level_dims[1]} (downsample: {level_downsample:.1f}x)"
        )

    # ==============================================================
    # Correctness test: GPU tiles must match CPU tiles pixel-for-pixel
    # ==============================================================
    test_batch_decode_correctness(img, level_dimensions)

    # ==============================================================
    # Performance test: GPU vs CPU timing comparison
    # ==============================================================
    test_batch_decode_performance(img, level_dimensions)

    # ==============================================================
    # Tile-level caching test
    # ==============================================================
    test_tile_level_caching(img, svs_path, CuImage)

    print("\n✅ Test completed successfully!")


def download_test_svs():
    """Download a small Aperio SVS test file from OpenSlide"""

    print("\n📥 Downloading Aperio SVS test file...")

    test_file = Path("/tmp/CMU-1-Small-Region.svs")

    if test_file.exists():
        print(f"✅ Test file already exists: {test_file}")
        return str(test_file)

    try:
        import urllib.request

        # Download small test file (2MB) from OpenSlide test data
        url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"

        print(f"   Downloading from: {url}")
        print("   Size: ~2MB (small test file)")
        print("   This may take a minute...")

        urllib.request.urlretrieve(url, test_file)

        print(f"✅ Downloaded: {test_file}")
        return str(test_file)

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def list_available_test_files():
    """List available Aperio SVS test files from OpenSlide"""

    print("\n📋 Available Aperio SVS Test Files from OpenSlide:")
    print("=" * 70)

    test_files = [
        ("CMU-1-Small-Region.svs", "~2MB", "Small region, JPEG, single pyramid level"),
        ("CMU-1.svs", "~177MB", "Brightfield, JPEG compression"),
        ("CMU-1-JP2K-33005.svs", "~126MB", "JPEG 2000, RGB"),
        ("CMU-2.svs", "~390MB", "Brightfield, JPEG compression"),
        ("CMU-3.svs", "~253MB", "Brightfield, JPEG compression"),
        ("JP2K-33003-1.svs", "~63MB", "Aorta tissue, JPEG 2000, YCbCr"),
        ("JP2K-33003-2.svs", "~275MB", "Heart tissue, JPEG 2000, YCbCr"),
    ]

    print(f"{'Filename':<25} {'Size':<10} {'Description'}")
    print("-" * 70)
    for filename, size, description in test_files:
        print(f"{filename:<25} {size:<10} {description}")

    print("\n💡 To download:")
    print(
        "   wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/<filename>"
    )
    print("\n📖 More info: https://openslide.cs.cmu.edu/download/openslide-testdata/")


def main():
    """Main function"""

    if len(sys.argv) < 2:
        print("Usage: python test_aperio_svs.py <path_to_svs_file>")
        print("   or: python test_aperio_svs.py --download (auto-download test file)")
        print("")
        print("Example:")
        print("  python test_aperio_svs.py /path/to/slide.svs")
        print("  python test_aperio_svs.py --download")
        print("")
        print("This script will:")
        print("  ✅ Configure cuslide2 plugin with nvImageCodec")
        print("  ✅ Load and analyze the SVS file")
        print("  ✅ Test GPU-accelerated decoding")
        print("  ✅ Compare CPU vs GPU performance")

        # List available test files
        list_available_test_files()
        return 1

    svs_path = sys.argv[1]

    # Handle --download flag
    if svs_path == "--download":
        svs_path = download_test_svs()
        if svs_path is None:
            print("\n❌ Failed to download test file")
            print("💡 You can manually download with:")
            print(
                "   wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
            )
            return 1

    # Setup environment
    plugin_lib = setup_environment("cucim_aperio_test")

    # Test the SVS file — exceptions indicate failure
    try:
        test_aperio_svs(svs_path, plugin_lib)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
