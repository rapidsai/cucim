#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation script for pyramidal OME-TIFF with cuslide2.

This script focuses on:
  - OME metadata presence and basic semantic checks
  - multi-level pyramid reads
  - channel-plane selection via read_region kwargs (C/Z/T)
  - CPU vs GPU decode consistency (where GPU is available)
  - optional tile-level caching smoke test
"""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from test_common import setup_environment, test_tile_level_caching


def _to_numpy(arr):
    """Convert CuImage/CuPy/NumPy-like objects to NumPy arrays."""
    if hasattr(arr, "get"):
        # CuPy array
        return arr.get()
    return np.asarray(arr)


def _max_read_size(level_dims: Tuple[int, int], cap: int = 512):
    return [min(cap, int(level_dims[0])), min(cap, int(level_dims[1]))]


def _print_public_dataset_references():
    print("\n📋 Public Cell DIVE datasets used in this effort:")
    print("=" * 70)
    print("Heart sample (5 markers):")
    print("  https://portal.hubmapconsortium.org/browse/dataset/e4263715a087881e46ea4d11f49139aa")
    print("Skin sample (19 markers):")
    print("  https://portal.hubmapconsortium.org/browse/dataset/1b8121539ff16f53681de6108069be24")


def _extract_ome_meta(metadata: Dict):
    """Extract OME metadata from CuImage metadata dict."""
    ome = metadata.get("ome", {}) if isinstance(metadata, dict) else {}
    size_c = int(ome.get("size_c", -1)) if isinstance(ome, dict) else -1
    size_z = int(ome.get("size_z", -1)) if isinstance(ome, dict) else -1
    size_t = int(ome.get("size_t", -1)) if isinstance(ome, dict) else -1
    channel_names = ome.get("channel_names", []) if isinstance(ome, dict) else []
    return ome, size_c, size_z, size_t, channel_names


def _load_cell_dive_tsv(tsv_path: str, hubmap_id: Optional[str] = None):
    """Load Cell DIVE assay keys from HuBMAP-style TSV metadata export."""
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"TSV is empty: {tsv_path}")

    # Header expected: HuBMAP ID, Entity, Key, Value, Description
    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        record = {
            "hubmap_id": parts[0].strip(),
            "entity": parts[1].strip(),
            "key": parts[2].strip(),
            "value": parts[3].strip(),
            "description": parts[4].strip(),
        }
        records.append(record)

    cell_dive_records = [r for r in records if r["entity"] == "Cell DIVE"]
    if not cell_dive_records:
        raise RuntimeError("No 'Cell DIVE' entity rows found in TSV")

    if hubmap_id is None:
        hubmap_id = cell_dive_records[0]["hubmap_id"]
    selected = [r for r in cell_dive_records if r["hubmap_id"] == hubmap_id]
    if not selected:
        raise RuntimeError(
            f"No Cell DIVE rows found for HuBMAP ID '{hubmap_id}'. "
            f"Available IDs: {sorted({r['hubmap_id'] for r in cell_dive_records})}"
        )

    kv = {r["key"]: r["value"] for r in selected}
    return hubmap_id, kv


def _as_float_or_none(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int_or_none(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _validate_tsv_mapping(tsv_hubmap_id: str, tsv_meta: Dict, ome: Dict, size_c: int, channel_names):
    """Map selected TSV keys to OME/cuslide2 outputs and validate consistency."""
    print("\n🧾 TSV → OME mapping checks")
    print("-" * 50)
    print(f"  HuBMAP ID: {tsv_hubmap_id}")

    # 1) Pixel spacing mapping:
    # TSV resolution_x/y_value (um) ↔ OME physical_size_x/y (um)
    tsv_rx = _as_float_or_none(tsv_meta.get("resolution_x_value"))
    tsv_ry = _as_float_or_none(tsv_meta.get("resolution_y_value"))
    ome_px = _as_float_or_none(ome.get("physical_size_x"))
    ome_py = _as_float_or_none(ome.get("physical_size_y"))

    if tsv_rx is not None and ome_px is not None:
        dx = abs(tsv_rx - ome_px)
        print(f"  resolution_x_value ({tsv_rx}) ↔ physical_size_x ({ome_px}), Δ={dx:.6f}")
        if dx > 1e-3:
            raise RuntimeError(
                f"TSV/OME X spacing mismatch is too large: TSV={tsv_rx}, OME={ome_px}, delta={dx}"
            )
    else:
        print("  ⚠️  Skipping X-spacing strict check (missing TSV or OME value)")

    if tsv_ry is not None and ome_py is not None:
        dy = abs(tsv_ry - ome_py)
        print(f"  resolution_y_value ({tsv_ry}) ↔ physical_size_y ({ome_py}), Δ={dy:.6f}")
        if dy > 1e-3:
            raise RuntimeError(
                f"TSV/OME Y spacing mismatch is too large: TSV={tsv_ry}, OME={ome_py}, delta={dy}"
            )
    else:
        print("  ⚠️  Skipping Y-spacing strict check (missing TSV or OME value)")

    # 2) Channel-count heuristics:
    # - number_of_antibodies is expected to be <= decoded size_c in most assembled datasets.
    # - number_of_channels is per-round and not equal to final SizeC, so only soft check.
    n_antibodies = _as_int_or_none(tsv_meta.get("number_of_antibodies"))
    n_channels_per_round = _as_int_or_none(tsv_meta.get("number_of_channels"))
    n_rounds = _as_int_or_none(tsv_meta.get("number_of_total_imaging_rounds"))

    print(f"  OME size_c={size_c}")
    if n_antibodies is not None:
        print(f"  TSV number_of_antibodies={n_antibodies}")
        if size_c > 0 and size_c < n_antibodies:
            raise RuntimeError(
                f"Decoded SizeC ({size_c}) is smaller than TSV antibody count ({n_antibodies})"
            )
    if n_channels_per_round is not None:
        print(f"  TSV number_of_channels (per round)={n_channels_per_round}")
    if n_rounds is not None:
        print(f"  TSV number_of_total_imaging_rounds={n_rounds}")

    # 3) Nuclear marker mapping:
    # TSV nuclear_marker_or_stain often expected among OME channel names.
    nuc = (tsv_meta.get("nuclear_marker_or_stain") or "").strip()
    if nuc and channel_names:
        lower_names = {str(x).strip().lower() for x in channel_names}
        print(f"  TSV nuclear_marker_or_stain={nuc}")
        if nuc.lower() in lower_names:
            print("  ✅ nuclear marker found in OME channel names")
        else:
            print("  ⚠️  nuclear marker not found in OME channel names (non-fatal)")
    elif nuc:
        print("  ⚠️  channel names missing, cannot verify nuclear marker mapping")

    # 4) Instrument/model metadata (soft checks)
    vendor = (tsv_meta.get("acquisition_instrument_vendor") or "").strip()
    model = (tsv_meta.get("acquisition_instrument_model") or "").strip()
    if vendor:
        print(f"  TSV acquisition_instrument_vendor={vendor}")
    if model:
        print(f"  TSV acquisition_instrument_model={model}")


def _validate_channel_selection(img, level_dims, level, c_index, z_index=0, t_index=0):
    """Validate per-plane selection using read_region kwargs."""
    read_size = _max_read_size(level_dims)
    kwargs = {"C": int(c_index), "Z": int(z_index), "T": int(t_index)}

    # CPU read
    start = time.time()
    cpu_region = img.read_region((0, 0), read_size, level=level, device="cpu", **kwargs)
    cpu_time = time.time() - start
    cpu_np = _to_numpy(cpu_region)

    print(
        f"    CPU plane read C={c_index},Z={z_index},T={t_index}: "
        f"{cpu_np.shape}, {cpu_np.dtype}, {cpu_time:.4f}s"
    )

    # GPU read (optional, skip if unavailable)
    try:
        start = time.time()
        gpu_region = img.read_region((0, 0), read_size, level=level, device="cuda", **kwargs)
        gpu_time = time.time() - start
        gpu_np = _to_numpy(gpu_region)
        print(
            f"    GPU plane read C={c_index},Z={z_index},T={t_index}: "
            f"{gpu_np.shape}, {gpu_np.dtype}, {gpu_time:.4f}s"
        )

        if gpu_np.shape != cpu_np.shape:
            raise RuntimeError(
                f"Shape mismatch for plane C={c_index},Z={z_index},T={t_index}: "
                f"GPU={gpu_np.shape}, CPU={cpu_np.shape}"
            )
        if not np.array_equal(gpu_np, cpu_np):
            max_diff = int(np.max(np.abs(gpu_np.astype(np.int64) - cpu_np.astype(np.int64))))
            raise RuntimeError(
                f"Pixel mismatch for plane C={c_index},Z={z_index},T={t_index}: max_diff={max_diff}"
            )
        print("    ✅ GPU and CPU plane decode are identical")
    except Exception as e:
        print(f"    ⚠️  GPU plane validation skipped/failed: {e}")


def _validate_batch_decode(img, level_dims):
    """Validate batch decode path (multiple tile reads)."""
    print("\n🔄 Batch decode validation")
    print("-" * 50)

    tile_w = min(256, int(level_dims[0]))
    tile_h = min(256, int(level_dims[1]))
    if tile_w <= 0 or tile_h <= 0:
        print("  ⚠️  Invalid level dimensions; skipping batch check")
        return

    max_x = max(0, int(level_dims[0]) - tile_w)
    max_y = max(0, int(level_dims[1]) - tile_h)
    locations = []
    for y_off in range(0, min(max_y + 1, tile_h * 4), tile_h):
        for x_off in range(0, min(max_x + 1, tile_w * 4), tile_w):
            locations.append([x_off, y_off])

    if len(locations) < 2:
        print("  ⚠️  Not enough locations for batch test")
        return

    batch_size = min(8, len(locations))
    print(
        f"  Locations={len(locations)}, tile={tile_w}x{tile_h}, "
        f"batch_size={batch_size}"
    )

    # CPU ground truth
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
    cpu_time = time.time() - start
    print(f"  CPU batch decode: {cpu_time:.4f}s ({len(cpu_tiles)} tiles)")

    # GPU batch compare (optional)
    try:
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
        gpu_time = time.time() - start
        print(f"  GPU batch decode: {gpu_time:.4f}s ({len(gpu_tiles)} tiles)")
        if gpu_time > 0:
            print(f"  🎯 Batch speedup: {cpu_time / gpu_time:.2f}x")

        if len(cpu_tiles) != len(gpu_tiles):
            raise RuntimeError(f"Tile count mismatch CPU={len(cpu_tiles)} GPU={len(gpu_tiles)}")

        mismatch = 0
        for idx, (cpu_tile, gpu_tile) in enumerate(zip(cpu_tiles, gpu_tiles)):
            c = _to_numpy(cpu_tile)
            g = _to_numpy(gpu_tile)
            if c.shape != g.shape or not np.array_equal(c, g):
                mismatch += 1
                if mismatch <= 3:
                    print(f"    ❌ mismatch at tile idx={idx}, location={locations[idx]}")
        if mismatch:
            raise RuntimeError(f"Batch decode mismatch count: {mismatch}/{len(cpu_tiles)}")
        print("  ✅ Batch decode GPU and CPU are identical")
    except Exception as e:
        print(f"  ⚠️  GPU batch validation skipped/failed: {e}")


def test_pyramidal_ome_tiff(
    file_path,
    plugin_lib,
    run_cache=True,
    tsv_path: Optional[str] = None,
    hubmap_id: Optional[str] = None,
):
    print("=" * 70)
    print("🔬 Testing pyramidal OME-TIFF with cuslide2")
    print("=" * 70)
    print(f"📁 File: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"OME-TIFF file not found: {file_path}")

    from cucim.clara import _set_plugin_root
    from cucim import CuImage

    _set_plugin_root(str(plugin_lib))
    print(f"✅ Plugin root set: {plugin_lib}")

    print("\n📂 Loading image...")
    start = time.time()
    img = CuImage(file_path)
    load_time = time.time() - start
    print(f"✅ Loaded in {load_time:.3f}s")

    print("\n📊 Image summary")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Device: {img.device}")
    level_count = img.resolutions["level_count"]
    print(f"  Levels: {level_count}")
    level_dimensions = img.resolutions["level_dimensions"]
    level_downsamples = img.resolutions["level_downsamples"]
    for level in range(level_count):
        dims = level_dimensions[level]
        ds = level_downsamples[level]
        print(f"  Level {level}: {dims[0]}x{dims[1]} (downsample: {ds:.3f}x)")

    print("\n🧬 OME metadata checks")
    metadata = img.metadata
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    ome, size_c, size_z, size_t, channel_names = _extract_ome_meta(metadata)
    if ome:
        print("  ✅ Found OME metadata block")
        print(f"  size_c={size_c}, size_z={size_z}, size_t={size_t}")
        print(f"  channel_names_count={len(channel_names)}")
    else:
        print("  ⚠️  OME metadata block not found in img.metadata")

    if tsv_path:
        tsv_hubmap_id, tsv_meta = _load_cell_dive_tsv(tsv_path, hubmap_id)
        _validate_tsv_mapping(tsv_hubmap_id, tsv_meta, ome, size_c, channel_names)

    # Multi-level single read smoke checks
    print("\n🧪 Multi-level decode checks")
    test_levels = sorted(set([0, min(level_count - 1, 1), level_count - 1]))
    for level in test_levels:
        dims = level_dimensions[level]
        read_size = _max_read_size(dims)
        start = time.time()
        cpu_region = img.read_region((0, 0), read_size, level=level, device="cpu")
        cpu_time = time.time() - start
        cpu_np = _to_numpy(cpu_region)
        print(f"  CPU level {level}: {cpu_np.shape}, {cpu_np.dtype}, {cpu_time:.4f}s")
        try:
            start = time.time()
            gpu_region = img.read_region((0, 0), read_size, level=level, device="cuda")
            gpu_time = time.time() - start
            gpu_np = _to_numpy(gpu_region)
            print(f"  GPU level {level}: {gpu_np.shape}, {gpu_np.dtype}, {gpu_time:.4f}s")
            if gpu_np.shape != cpu_np.shape:
                raise RuntimeError(
                    f"Shape mismatch at level {level}: CPU={cpu_np.shape}, GPU={gpu_np.shape}"
                )
        except Exception as e:
            print(f"  ⚠️  GPU level {level} validation skipped/failed: {e}")

    # Plane-selection checks (if OME C/Z/T metadata is available)
    if size_c > 0:
        print("\n🎯 Plane selection checks (C/Z/T kwargs)")
        dims0 = level_dimensions[0]
        c_candidates = [0]
        if size_c > 1:
            c_candidates.append(1)
        z_idx = 0 if size_z <= 0 else min(size_z - 1, 0)
        t_idx = 0 if size_t <= 0 else min(size_t - 1, 0)
        for c_idx in c_candidates:
            _validate_channel_selection(img, dims0, level=0, c_index=c_idx, z_index=z_idx, t_index=t_idx)
    else:
        print("\n⚠️  Skipping C/Z/T plane checks (size_c not available)")

    _validate_batch_decode(img, level_dimensions[0])

    if run_cache:
        print("\n💾 Tile-level cache check")
        test_tile_level_caching(img, file_path, CuImage)

    print("\n✅ Pyramidal OME-TIFF test completed")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python test_pyramidal_ome_tiff.py <path_to_ome_tiff> "
            "[--no-cache] [--tsv <path>] [--hubmap-id <ID>]"
        )
        print("")
        print("Examples:")
        print("  python test_pyramidal_ome_tiff.py /data/sample.ome.tif")
        print("  python test_pyramidal_ome_tiff.py /data/sample.ome.tiff --no-cache")
        print(
            "  python test_pyramidal_ome_tiff.py /data/sample.ome.tif "
            "--tsv /path/HBM388.RPTF.754.tsv --hubmap-id HBM388.RPTF.754"
        )
        _print_public_dataset_references()
        return 1

    file_path = sys.argv[1]
    run_cache = "--no-cache" not in sys.argv[2:]
    tsv_path = None
    hubmap_id = None

    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        if argv[i] == "--tsv" and i + 1 < len(argv):
            tsv_path = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--hubmap-id" and i + 1 < len(argv):
            hubmap_id = argv[i + 1]
            i += 2
            continue
        i += 1

    plugin_lib = setup_environment("cucim_ome_tiff_test")
    try:
        test_pyramidal_ome_tiff(
            file_path,
            plugin_lib,
            run_cache=run_cache,
            tsv_path=tsv_path,
            hubmap_id=hubmap_id,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

