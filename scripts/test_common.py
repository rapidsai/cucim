#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for cuslide2 test scripts."""

import json
import os
import re
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np


def plugin_version_from_dist_version(dist_version: str) -> str:
    """Convert a dist version like '26.2.0' to cuCIM plugin version format '26.02.00'.

    cuCIM plugin filenames use zero-padded minor/patch components.
    """
    m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", dist_version)
    if not m:
        return dist_version
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return f"{major}.{minor:02d}.{patch:02d}"


def setup_environment(config_name: str = "cucim_test") -> str:
    """Setup cuCIM environment for cuslide2 plugin.

    Args:
        config_name: Base name for the temporary config JSON file.

    Returns:
        Plugin library path as a string.
    """
    repo_root = Path(__file__).parent.parent
    plugin_lib = (
        repo_root / "cpp" / "plugins" / "cucim.kit.cuslide2" / "build-release" / "lib"
    )

    if not plugin_lib.exists():
        plugin_lib = repo_root / "install" / "lib"

    dist_version = None
    for pkg_name in ["cucim-cu13", "cucim-cu12", "cucim"]:
        try:
            dist_version = importlib_metadata.version(pkg_name)
            break
        except importlib_metadata.PackageNotFoundError:
            continue
    if dist_version is None:
        raise importlib_metadata.PackageNotFoundError("cucim")
    version = plugin_version_from_dist_version(dist_version)

    if os.getenv("ENABLE_CUSLIDE2") == "1":
        os.environ["CUCIM_PLUGINS"] = f"cucim.kit.cuslide2@{version}.so"
        os.environ.pop("CUCIM_CONFIG_PATH", None)
        print(
            f"✅ Plugin selection via env: ENABLE_CUSLIDE2=1 "
            f"+ CUCIM_PLUGINS={os.environ['CUCIM_PLUGINS']}"
        )
    else:
        config = {
            "plugin": {
                "names": [
                    f"cucim.kit.cuslide2@{version}.so",
                ]
            }
        }

        config_path = os.path.join(tempfile.gettempdir(), f".{config_name}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        os.environ["CUCIM_CONFIG_PATH"] = config_path
        os.environ.pop("CUCIM_PLUGINS", None)
        print(f"✅ Plugin configuration: {config_path}")
    print(f"✅ Plugin library path: {plugin_lib}")

    return str(plugin_lib)


def test_tile_level_caching(img, file_path, CuImage):
    """Test tile-level caching (per-process image cache).

    Exercises the tile-level cache in ifd.cpp: ROI -> tile grid
    decomposition, per-tile cache lookup, miss-decode-insert,
    and warm-read cache hits.

    NOTE: Tile-level caching requires that the cuslide2 plugin
    successfully extracts TileWidth/TileLength TIFF tags via
    nvImageCodec (>= 0.7.0). If tile sizes are (0,0), the
    caching code path is not entered and the test is skipped.

    Raises:
        RuntimeError: If caching assertions fail.
    """
    print("\n💾 Testing tile-level caching...")
    print("-" * 50)

    level_tile_sizes = img.resolutions.get("level_tile_sizes", ())
    has_tile_dims = (
        len(level_tile_sizes) > 0
        and level_tile_sizes[0][0] > 0
        and level_tile_sizes[0][1] > 0
    )
    print(
        f"  Tile sizes (level 0): {level_tile_sizes[0] if level_tile_sizes else 'N/A'}"
    )

    if not has_tile_dims:
        print(
            "  ⚠️  Tile dimensions are (0,0) — tile-level caching path is "
            "inactive.\n"
            "     This occurs when nvImageCodec does not expose "
            "TileWidth/TileLength TIFF tags.\n"
            "     Skipping caching assertions (decode still works, "
            "just without cache)."
        )
        return

    # --- Baseline: non-cached direct decode (before enabling cache) ---
    region_direct = img.read_region((0, 0), (512, 512), level=0)
    arr_direct = np.asarray(region_direct)
    print(f"\n  📐 Non-cached baseline (512x512): shape={arr_direct.shape}")

    # Enable per-process cache (256 MB) and stat recording.
    # Re-open the image after configuring the cache so the
    # IFD picks up the active cache manager.
    CuImage.cache("per_process", memory_capacity=256)
    CuImage.cache().record(True)
    img_cached = CuImage(file_path)
    print(f"  Cache type: {CuImage.cache().type}")
    print(f"  Stat recording: {CuImage.cache().record()}")

    # --- Cold read (all cache misses) ---
    region_cold = img_cached.read_region((0, 0), (512, 512), level=0)
    cold_hits = CuImage.cache().hit_count
    cold_misses = CuImage.cache().miss_count
    print("\n  🧊 Cold read (512x512):")
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
    print("\n  🔥 Warm read (same region):")
    print(
        f"     Hits: {warm_hits} (+{new_hits}), Misses: {warm_misses} (+{new_misses})"
    )
    print(f"     Time: {warm_time * 1000:.1f} ms")
    assert new_hits > 0, "Expected cache hits on warm read"
    assert new_misses == 0, "Expected zero new misses on warm read"

    # --- Data correctness: cold vs warm must match ---
    arr_cold = np.asarray(region_cold)
    arr_warm = np.asarray(region_warm)
    assert np.array_equal(arr_cold, arr_warm), "Cold and warm reads differ!"
    print(f"     ✅ Cold vs warm data matches (shape={arr_cold.shape})")

    # --- Data correctness: cached vs non-cached direct decode must match ---
    assert np.array_equal(arr_direct, arr_cold), "Cold and non-cached reads differ!"
    print("     ✅ Cached vs non-cached data matches")

    # --- Overlapping read (partial hit: some tiles shared, some new) ---
    overlap_origin = (128, 128)
    overlap_size = (512, 512)

    prev_hits = CuImage.cache().hit_count
    prev_misses = CuImage.cache().miss_count
    region_overlap = img_cached.read_region(overlap_origin, overlap_size, level=0)
    overlap_new_hits = CuImage.cache().hit_count - prev_hits
    overlap_new_misses = CuImage.cache().miss_count - prev_misses
    print("\n  🔀 Overlapping read (offset 128,128):")
    print(
        f"     Hits: {CuImage.cache().hit_count} (+{overlap_new_hits}), "
        f"Misses: {CuImage.cache().miss_count} (+{overlap_new_misses})"
    )
    assert overlap_new_hits > 0, (
        "Expected cache hits from tiles shared with previous (0,0) read"
    )
    assert overlap_new_misses > 0, (
        "Expected cache misses for tiles outside the previous (0,0) region"
    )

    # Data correctness: a second read of the same region (now fully cached)
    # must produce identical output.
    region_overlap_again = img_cached.read_region(overlap_origin, overlap_size, level=0)
    arr_overlap_cached = np.asarray(region_overlap)
    arr_overlap_again = np.asarray(region_overlap_again)
    assert np.array_equal(arr_overlap_cached, arr_overlap_again), (
        "Overlapping cached reads differ!"
    )
    print("     ✅ Overlapping read data is consistent across cached reads")

    # --- Summary ---
    print("\n  📊 Cache summary:")
    print(f"     Total hits:   {CuImage.cache().hit_count}")
    print(f"     Total misses: {CuImage.cache().miss_count}")
    print(f"     Cache size:   {CuImage.cache().size} tiles")
    print("  ✅ Tile-level caching test passed!")
