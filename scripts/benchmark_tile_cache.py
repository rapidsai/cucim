#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure whether cuslide2's tile cache pays for itself, per codec and device.

Review feedback on PR #1034 / issue #1069 raised two questions that no
existing benchmark answers:

1. For light codecs, is a warm cache read actually *slower* than decoding
   again?  The cache stores decoded tiles in host memory and assembles the
   requested region with a single-threaded per-row ``memcpy``, so when decode
   is cheap that copy can dominate.

2. Would a GPU-resident cache be better?  This matters more than it first
   appears: with the cache disabled, nvImageCodec decodes straight into device
   memory (``STRIDED_DEVICE``).  With it enabled, cuslide2 decodes tiles to
   host, assembles on host, then copies the whole region up with a single
   ``cudaMemcpyHostToDevice``.  Enabling the cache therefore *gives up*
   direct-to-device decode for CUDA output, so a "cache on" number that looks
   bad for ``device="cuda"`` is evidence for the GPU-resident design rather
   than against caching as such.

The measured grid is compression x cache setting x output device.  Two access
patterns are timed for each cell: repeated reads of one region (the warm-hit
case the cache is meant to win) and reads of distinct regions (the miss path,
which pays decode plus insert overhead).

Cache hit/miss counters are reported alongside the timings.  Without them a
run where the cache silently never engaged -- which happens whenever
nvImageCodec does not expose TileWidth/TileLength, leaving tile dims at (0,0)
-- is indistinguishable from one where caching genuinely did not help.
"""

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from test_common import setup_environment

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPRESSIONS = ("jpeg", "deflate", "raw")


def generate_images(dest, compressions, image_size, tile_size):
    """Generate one tiled TIFF per compression scheme.

    Each recipe gets its own subdirectory because the generator derives the
    output filename from pattern/size/tile only -- compression is not part of
    it, so same-size recipes would otherwise overwrite each other.
    """
    sys.path.insert(0, str(REPO_ROOT / "python" / "cucim" / "tests" / "util"))
    from gen_image import ImageGenerator

    paths = {}
    for compression in compressions:
        subdir = dest / compression
        subdir.mkdir(parents=True, exist_ok=True)
        recipe = f"tiff::stripe:{image_size}:{tile_size}:{compression}"
        (path,) = ImageGenerator(str(subdir), [recipe]).gen()
        paths[compression] = Path(path)
        print(
            f"  {compression:<8} {Path(path).name}  "
            f"{Path(path).stat().st_size / 2**20:.1f} MB"
        )
    return paths


def _synchronize():
    """Best-effort device sync so async work is not attributed to the next read."""
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def time_reads(img, regions, size, device, repeats):
    """Return per-read wall times in milliseconds for *device* output.

    The CPU path is materialized via ``np.asarray``, which is a zero-copy view
    over the returned buffer.  The CUDA path deliberately does *not* copy back
    to host: a device-to-host transfer is not part of what read_region costs,
    and adding one would swamp the very difference being measured.
    """
    on_cuda = device == "cuda"
    timings = []
    for _ in range(repeats):
        for origin in regions:
            if on_cuda:
                _synchronize()
            start = time.perf_counter()
            region = img.read_region(origin, size, level=0, device=device)
            if on_cuda:
                _synchronize()
            else:
                # Zero-copy view; ensures the buffer is real, not lazy.
                np.asarray(region)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def measure(path, CuImage, cache_type, device, size, repeats, num_regions):
    """Time warm and cold access patterns for one cache/device combination."""
    # The IFD binds to whatever cache manager is active when the image is
    # opened, so configure the cache first and open afterwards.
    if cache_type == "nocache":
        CuImage.cache("nocache")
    else:
        CuImage.cache(cache_type, memory_capacity=1024)
    CuImage.cache().record(True)

    img = CuImage(str(path))
    tile_sizes = img.resolutions.get("level_tile_sizes", ())
    tile = list(tile_sizes[0]) if tile_sizes else [0, 0]

    width, height = img.shape[1], img.shape[0]
    # Distinct, non-overlapping origins for the miss path.
    step = max(size[0], size[1])
    regions = [
        (
            min((i * step) % max(width - size[0], 1), width - size[0]),
            min((i * step) % max(height - size[1], 1), height - size[1]),
        )
        for i in range(num_regions)
    ]

    result = {
        "cache": cache_type,
        "device": device,
        "tile": tile,
        "cache_active": bool(tile[0] > 0 and tile[1] > 0) and cache_type != "nocache",
    }

    # Warm: same region repeatedly.  First read populates, the rest should hit.
    time_reads(img, [regions[0]], size, device, 1)
    before_hits = CuImage.cache().hit_count
    before_misses = CuImage.cache().miss_count
    warm = time_reads(img, [regions[0]], size, device, repeats)
    result["warm_ms"] = statistics.median(warm)
    result["warm_hits"] = CuImage.cache().hit_count - before_hits
    result["warm_misses"] = CuImage.cache().miss_count - before_misses

    # Cold: distinct regions, so every tile is a first touch.  Reopening also
    # drops any per-image state carried over from the warm pass.
    img_cold = CuImage(str(path))
    before_hits = CuImage.cache().hit_count
    before_misses = CuImage.cache().miss_count
    cold = time_reads(img_cold, regions, size, device, 1)
    result["cold_ms"] = statistics.median(cold)
    result["cold_hits"] = CuImage.cache().hit_count - before_hits
    result["cold_misses"] = CuImage.cache().miss_count - before_misses

    return result


def run(paths, CuImage, devices, size, repeats, num_regions):
    rows = []
    for compression, path in paths.items():
        print(f"\n--- {compression} ---")
        for device in devices:
            for cache_type in ("nocache", "per_process"):
                try:
                    row = measure(
                        path,
                        CuImage,
                        cache_type,
                        device,
                        size,
                        repeats,
                        num_regions,
                    )
                except Exception as exc:
                    print(
                        f"  {device:<5} {cache_type:<12} "
                        f"FAILED: {type(exc).__name__}: {exc}"
                    )
                    rows.append(
                        {
                            "compression": compression,
                            "cache": cache_type,
                            "device": device,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                row["compression"] = compression
                rows.append(row)
                print(
                    f"  {device:<5} {cache_type:<12} "
                    f"warm {row['warm_ms']:7.2f} ms "
                    f"(h{row['warm_hits']}/m{row['warm_misses']})   "
                    f"cold {row['cold_ms']:7.2f} ms "
                    f"(h{row['cold_hits']}/m{row['cold_misses']})"
                )
    return rows


def report(rows, devices):
    print()
    print("=" * 72)
    print("CACHE SPEEDUP (nocache / per_process; >1 means caching helped)")
    print("=" * 72)

    indexed = {
        (r.get("compression"), r.get("device"), r.get("cache")): r
        for r in rows
        if "error" not in r
    }
    compressions = []
    for r in rows:
        if r.get("compression") not in compressions:
            compressions.append(r.get("compression"))

    inactive = []
    for device in devices:
        print(f"\noutput device = {device}")
        print(f"  {'codec':<10} {'warm':>10} {'cold':>10}   note")
        for compression in compressions:
            off = indexed.get((compression, device, "nocache"))
            on = indexed.get((compression, device, "per_process"))
            if not off or not on:
                continue

            warm = off["warm_ms"] / on["warm_ms"] if on["warm_ms"] else 0
            cold = off["cold_ms"] / on["cold_ms"] if on["cold_ms"] else 0

            note = ""
            if not on["cache_active"]:
                note = "cache never engaged (tile dims 0x0)"
                inactive.append((compression, device))
            elif on["warm_hits"] == 0:
                note = "no warm hits - cache not serving reads"
            elif warm < 1.0:
                note = "caching is a regression here"
            print(f"  {compression:<10} {warm:9.2f}x {cold:9.2f}x   {note}")

    if inactive:
        print(
            "\nThe cache path requires TileWidth/TileLength from nvImageCodec.\n"
            "Where tile dims came back 0x0 the cache was bypassed entirely, so\n"
            "those rows compare identical code paths and mean nothing."
        )

    if "cuda" in devices:
        print(
            "\nOn the CUDA rows, remember the comparison is not "
            "cache-vs-no-cache in\nisolation: the nocache path decodes straight "
            "to device memory, while the\ncached path decodes to host, "
            "assembles with host memcpy, then copies H2D.\nA warm ratio below "
            "1.0x there is the cost of that round-trip, and is the\ncase for a "
            "GPU-resident tile cache (issue #1069)."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--image-size",
        default="4096x4096",
        help="Generated image size (default: 4096x4096)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256, help="Tile size (default: 256)"
    )
    parser.add_argument(
        "--region",
        default="1024x1024",
        help="Region size to read (default: 1024x1024)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Warm reads to time per cell (default: 20)",
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=8,
        help="Distinct regions for the cold pattern (default: 8)",
    )
    parser.add_argument(
        "--compression",
        dest="compressions",
        action="append",
        choices=COMPRESSIONS,
        help="Restrict to this codec (repeatable; default: all)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip the CUDA output device",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the generated images instead of deleting them",
    )
    parser.add_argument("--report", help="Write a JSON report to this path")
    args = parser.parse_args()

    region = tuple(int(v) for v in args.region.split("x"))
    compressions = tuple(args.compressions or COMPRESSIONS)

    plugin_lib = setup_environment("cucim_cache_benchmark")
    try:
        from cucim.clara import _set_plugin_root

        _set_plugin_root(plugin_lib)
        from cucim import CuImage
    except ImportError as exc:
        print(
            f"cuCIM is not importable here ({exc}).\n"
            "Build/install cuCIM with the cuslide2 plugin first."
        )
        return 2

    devices = ["cpu"] if args.cpu_only else ["cpu", "cuda"]

    workdir = Path(tempfile.mkdtemp(prefix="cucim_cache_bench_"))
    try:
        print(f"Generating test images in {workdir}")
        paths = generate_images(workdir, compressions, args.image_size, args.tile_size)

        print(
            f"\nRegion {region[0]}x{region[1]}, {args.repeats} warm reads, "
            f"{args.regions} cold regions"
        )
        rows = run(paths, CuImage, devices, region, args.repeats, args.regions)
        report(rows, devices)

        if args.report:
            import json

            Path(args.report).write_text(json.dumps(rows, indent=2))
            print(f"\nJSON report written to {args.report}")
    finally:
        if args.keep:
            print(f"\nGenerated images kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
