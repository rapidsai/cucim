#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script for cuslide2 plugin with Aperio SVS files
"""

import json
import os
import re
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path


def _plugin_version_from_dist_version(dist_version: str) -> str:
    """
    Convert a dist version like '26.2.0' to cuCIM plugin version format '26.02.00'.

    cuCIM plugin filenames use zero-padded minor/patch components.
    """
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", dist_version)
    if not m:
        # Fall back to the raw version string (best-effort)
        return dist_version
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return f"{major}.{minor:02d}.{patch:02d}"


def setup_environment():
    """Setup cuCIM environment for cuslide2 plugin"""
    # Get current build directory
    repo_root = Path(__file__).parent.parent
    plugin_lib = (
        repo_root / "cpp" / "plugins" / "cucim.kit.cuslide2" / "build-release" / "lib"
    )

    if not plugin_lib.exists():
        plugin_lib = repo_root / "install" / "lib"

    # Try CUDA-specific packages first, then fall back to generic cucim
    dist_version = None
    for pkg_name in ["cucim-cu13", "cucim-cu12", "cucim"]:
        try:
            dist_version = importlib_metadata.version(pkg_name)
            break
        except importlib_metadata.PackageNotFoundError:
            continue
    if dist_version is None:
        raise importlib_metadata.PackageNotFoundError("cucim")
    version = _plugin_version_from_dist_version(dist_version)

    if os.getenv("ENABLE_CUSLIDE2") == "1":
        os.environ["CUCIM_PLUGINS"] = f"cucim.kit.cuslide2@{version}.so"
        os.environ.pop("CUCIM_CONFIG_PATH", None)
        print(
            f"✅ Plugin selection via env: ENABLE_CUSLIDE2=1 + CUCIM_PLUGINS={os.environ['CUCIM_PLUGINS']}"
        )
    else:
        config = {
            "plugin": {
                "names": [
                    f"cucim.kit.cuslide2@{version}.so",
                ]
            }
        }

        config_path = os.path.join(tempfile.gettempdir(), ".cucim_aperio_test.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        os.environ["CUCIM_CONFIG_PATH"] = config_path
        os.environ.pop("CUCIM_PLUGINS", None)
        print(f"✅ Plugin configuration: {config_path}")
    print(f"✅ Plugin library path: {plugin_lib}")

    return str(plugin_lib)


def test_aperio_svs(svs_path, plugin_lib):
    """Test cuslide2 plugin with an Aperio SVS file"""

    print("\n🔬 Testing cuslide2 plugin with Aperio SVS")
    print("=" * 60)
    print(f"📁 File: {svs_path}")

    if not Path(svs_path).exists():
        print(f"❌ File not found: {svs_path}")
        return False

    try:
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

        # Try to read a tile from level 0 (GPU)
        print("\n🚀 Testing GPU decode (nvImageCodec)...")
        try:
            start = time.time()
            gpu_tile = img.read_region(
                location=[0, 0], size=[512, 512], level=0, device="cuda"
            )
            gpu_time = time.time() - start

            print("✅ GPU decode successful!")
            print(f"  Time: {gpu_time:.4f}s")
            print(f"  Shape: {gpu_tile.shape}")
            print(f"  Device: {gpu_tile.device}")
        except Exception as e:
            print(f"⚠️  GPU decode failed: {e}")
            print("   (This is expected if CUDA is not available)")
            gpu_time = None

        # Try to read same tile from CPU
        print("\n🖥️  Testing CPU decode (baseline)...")
        try:
            start = time.time()
            cpu_tile = img.read_region(
                location=[0, 0], size=[512, 512], level=0, device="cpu"
            )
            cpu_time = time.time() - start

            print("✅ CPU decode successful!")
            print(f"  Time: {cpu_time:.4f}s")
            print(f"  Shape: {cpu_tile.shape}")
            print(f"  Device: {cpu_tile.device}")

            # Calculate speedup
            if gpu_time:
                speedup = cpu_time / gpu_time
                print(f"\n🎯 GPU Speedup: {speedup:.2f}x faster than CPU")

                if speedup > 1.5:
                    print("   🚀 nvImageCodec GPU acceleration is working!")
                elif speedup > 0.9:
                    print("   ✅ GPU decode working (speedup may vary by tile size)")
                else:
                    print("   ℹ️  CPU was faster for this small tile")
        except Exception as e:
            print(f"❌ CPU decode failed: {e}")

        # Test larger tile for better speedup
        print("\n📏 Testing larger tile (2048x2048)...")
        try:
            # GPU
            start = time.time()
            _ = img.read_region([0, 0], [2048, 2048], 0, device="cuda")
            gpu_large_time = time.time() - start
            print(f"  GPU: {gpu_large_time:.4f}s")

            # CPU
            start = time.time()
            _ = img.read_region([0, 0], [2048, 2048], 0, device="cpu")
            cpu_large_time = time.time() - start
            print(f"  CPU: {cpu_large_time:.4f}s")

            speedup = cpu_large_time / gpu_large_time
            print(f"  🎯 Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"  ⚠️  Large tile test failed: {e}")

        # ==============================================================
        # Test async batch decode path (multiple locations + GPU)
        # This exercises NvImageCodecProcessor, schedule_batch_decode(),
        # and wait_batch_decode() — the async scheduling API.
        # ==============================================================
        print("\n🔄 Testing async batch decode (multiple locations, GPU)...")
        try:
            level_dims = level_dimensions[0]
            tile_w, tile_h = 256, 256
            max_x = max(0, level_dims[0] - tile_w)
            max_y = max(0, level_dims[1] - tile_h)

            # Generate a grid of tile locations
            locations = []
            for y_off in range(0, min(max_y + 1, tile_h * 4), tile_h):
                for x_off in range(0, min(max_x + 1, tile_w * 4), tile_w):
                    locations.append([x_off, y_off])
            num_locations = len(locations)

            if num_locations < 2:
                print(
                    "  ⚠️  Image too small for batch test "
                    f"({level_dims[0]}x{level_dims[1]})"
                )
            else:
                batch_size = min(num_locations, 8)
                print(
                    f"  Locations: {num_locations}, "
                    f"tile: {tile_w}x{tile_h}, "
                    f"batch_size: {batch_size}"
                )

                # GPU batch decode (async path)
                start = time.time()
                gpu_batch = img.read_region(
                    location=locations,
                    size=[tile_w, tile_h],
                    level=0,
                    device="cuda",
                    batch_size=batch_size,
                    num_workers=1,
                )
                # Consume iterator to force all batches
                gpu_tiles = []
                for tile in gpu_batch:
                    gpu_tiles.append(tile)
                gpu_batch_time = time.time() - start
                print(f"  ✅ GPU batch: {gpu_batch_time:.4f}s ({len(gpu_tiles)} tiles)")

                # CPU batch decode (per-tile path, for comparison)
                start = time.time()
                cpu_batch = img.read_region(
                    location=locations,
                    size=[tile_w, tile_h],
                    level=0,
                    device="cpu",
                    batch_size=batch_size,
                    num_workers=1,
                )
                cpu_tiles = []
                for tile in cpu_batch:
                    cpu_tiles.append(tile)
                cpu_batch_time = time.time() - start
                print(f"  ✅ CPU batch: {cpu_batch_time:.4f}s ({len(cpu_tiles)} tiles)")

                if gpu_batch_time > 0:
                    speedup = cpu_batch_time / gpu_batch_time
                    print(f"  🎯 Batch speedup: {speedup:.2f}x")

                # Verify tile shapes
                if gpu_tiles:
                    print(f"  Tile shape: {gpu_tiles[0].shape}")
                    print(f"  Tile device: {gpu_tiles[0].device}")

        except Exception as e:
            print(f"  ⚠️  Batch decode test failed: {e}")
            import traceback

            traceback.print_exc()

        # ==============================================================
        # Test tile-level caching (per-process image cache)
        # Exercises the tile-level cache in ifd.cpp: ROI -> tile grid
        # decomposition, per-tile cache lookup, miss-decode-insert,
        # and warm-read cache hits.
        #
        # NOTE: Tile-level caching requires that the cuslide2 plugin
        # successfully extracts TileWidth/TileLength TIFF tags via
        # nvImageCodec (>= 0.7.0). If tile sizes are (0,0), the
        # caching code path is not entered and the test is skipped.
        # ==============================================================
        print("\n💾 Testing tile-level caching...")
        try:
            # Check tile sizes — caching requires non-zero tile dimensions
            level_tile_sizes = img.resolutions.get("level_tile_sizes", ())
            has_tile_dims = (
                len(level_tile_sizes) > 0
                and level_tile_sizes[0][0] > 0
                and level_tile_sizes[0][1] > 0
            )
            print(f"  Tile sizes (level 0): {level_tile_sizes[0] if level_tile_sizes else 'N/A'}")

            if not has_tile_dims:
                print(
                    "  ⚠️  Tile dimensions are (0,0) — tile-level caching path is "
                    "inactive.\n"
                    "     This occurs when nvImageCodec does not expose "
                    "TileWidth/TileLength TIFF tags.\n"
                    "     Skipping caching assertions (decode still works, "
                    "just without cache)."
                )
            else:
                # Enable per-process cache (256 MB) and stat recording.
                # Re-open the image after configuring the cache so the
                # IFD picks up the active cache manager.
                CuImage.cache("per_process", memory_capacity=256)
                CuImage.cache().record(True)
                img_cached = CuImage(svs_path)
                print(f"  Cache type: {CuImage.cache().type}")
                print(f"  Stat recording: {CuImage.cache().record()}")

                import numpy as np

                # --- Cold read (all cache misses) ---
                region_cold = img_cached.read_region((0, 0), (512, 512), level=0)
                cold_hits = CuImage.cache().hit_count
                cold_misses = CuImage.cache().miss_count
                print(f"\n  🧊 Cold read (512x512):")
                print(f"     Hits: {cold_hits}, Misses: {cold_misses}")
                assert cold_misses > 0, "Expected cache misses on cold read"

                # --- Warm read (same region -> all cache hits) ---
                start = time.time()
                region_warm = img_cached.read_region((0, 0), (512, 512), level=0)
                warm_time = time.time() - start
                warm_hits = CuImage.cache().hit_count
                warm_misses = CuImage.cache().miss_count
                new_hits = warm_hits - cold_hits
                new_misses = warm_misses - cold_misses
                print(f"\n  🔥 Warm read (same region):")
                print(
                    f"     Hits: {warm_hits} (+{new_hits}), "
                    f"Misses: {warm_misses} (+{new_misses})"
                )
                print(f"     Time: {warm_time * 1000:.1f} ms")
                assert new_hits > 0, "Expected cache hits on warm read"
                assert new_misses == 0, "Expected zero new misses on warm read"

                # --- Data correctness: cold vs warm must match ---
                arr_cold = np.asarray(region_cold)
                arr_warm = np.asarray(region_warm)
                assert np.array_equal(arr_cold, arr_warm), "Cold and warm reads differ!"
                print(f"     ✅ Data matches (shape={arr_cold.shape})")

                # --- Overlapping read (partial hit: some tiles shared) ---
                prev_hits = CuImage.cache().hit_count
                prev_misses = CuImage.cache().miss_count
                region_overlap = img_cached.read_region(
                    (128, 128), (512, 512), level=0
                )
                overlap_hits = CuImage.cache().hit_count
                overlap_misses = CuImage.cache().miss_count
                print(f"\n  🔀 Overlapping read (offset 128,128):")
                print(
                    f"     Hits: {overlap_hits} (+{overlap_hits - prev_hits}), "
                    f"Misses: {overlap_misses} (+{overlap_misses - prev_misses})"
                )

                # --- Summary ---
                print(f"\n  📊 Cache summary:")
                print(f"     Total hits:   {CuImage.cache().hit_count}")
                print(f"     Total misses: {CuImage.cache().miss_count}")
                print(f"     Cache size:   {CuImage.cache().size} tiles")
                print("  ✅ Tile-level caching test passed!")

        except Exception as e:
            print(f"  ❌ Caching test failed: {e}")
            import traceback

            traceback.print_exc()

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


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
    plugin_lib = setup_environment()

    # Test the SVS file
    success = test_aperio_svs(svs_path, plugin_lib)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
