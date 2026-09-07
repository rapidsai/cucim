#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate cuslide2 against every pyramid format it claims to accept.

PR #1085 added ``.ndpi``, ``.scn``, ``.bif`` and ``.qptiff`` to the cuslide2
extension gate plus vendor detection for Hamamatsu, Leica, Ventana, Trestle and
OME-TIFF, but shipped no tests for any of them.  This script is the
fact-finding pass for issue #1094: it reports what works today rather than
fixing anything.

Two phases:

``open``
    Open every candidate image and record geometry, pyramid levels, tile
    dimensions, pixel spacing and whether a small region decodes.

``refs``
    Diff NDPI reads against the golden reference crops in ``hamamatsu/ref``,
    driven by that directory's ``manifest.json``.  The crop plan there
    deliberately covers aligned corners, centers, deliberately misaligned odd
    sizes (1025x1031), large regions up to 8192x8192, and wide/tall strips
    (4096x256, 256x4096) -- the strips matter most for NDPI, whose restart
    markers yield long thin addressable slivers rather than square tiles.

Test images come from the internal corpus at
gitlab-master.nvidia.com/cuda-hpc-libraries/ImageCodecs/imagecodecs-test-images-large.
Point ``CUCIM_LARGE_TESTDATA_FOLDER`` at a local clone; the script exits
cleanly when it is unset so it stays safe to invoke unconditionally.

That corpus is >100 GB behind git-LFS, so most files are usually absent or
unmaterialized pointers.  Both are reported separately from genuine failures --
conflating them would drown the real signal.

On pass criteria: the reference crops were produced by ``generate_refs.py``
using tifffile, which applies its own YCbCr-to-RGB conversion.  nvImageCodec
need not round chroma identically, so small per-channel deltas are expected and
are reported as ``close`` rather than ``mismatch``.  A genuinely wrong read
(wrong offset, wrong level, wrong sliver) shows up as large deltas across most
of the region, which no tolerance will mask.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from test_common import setup_environment

ENV_VAR = "CUCIM_LARGE_TESTDATA_FOLDER"

# Directory -> (expected vendor flavor, extensions to consider).  Keyed on
# directory because extension alone cannot separate Philips, Trestle, OME-TIFF
# and generic TIFF, which all use `.tif`/`.tiff`.
CORPUS_LAYOUT = {
    "aperio": ("Aperio SVS", (".svs",)),
    "philips": ("Philips TIFF", (".tif", ".tiff")),
    "hamamatsu": ("Hamamatsu NDPI", (".ndpi",)),
    "leica-scn": ("Leica SCN", (".scn",)),
    "ventana": ("Ventana BIF", (".bif",)),
    "vectra-qptiff": ("Vectra QPTIFF", (".qptiff", ".tif")),
    "ome-tiff": ("OME-TIFF", (".tif", ".tiff")),
    "trestle": ("Trestle TIFF", (".tif",)),
    "misc-tiff": ("Generic TIFF", (".tif", ".tiff")),
    "geotiff": ("Generic TIFF", (".tif", ".tiff")),
    "planetlabs": ("Generic TIFF", (".tif", ".tiff")),
    "grundium": ("Aperio SVS", (".svs",)),
}

# Fallback only; `manifest.json` is authoritative when present.
REF_PATTERN = re.compile(
    r"^(?P<slide>.+)_ifd(?P<ifd>\d+)_x(?P<x>\d+)_y(?P<y>\d+)"
    r"_(?P<w>\d+)x(?P<h>\d+)\.raw$"
)

LFS_MAGIC = b"version https://git-lfs"


def file_state(path):
    """Classify *path* as ``ok``, ``missing`` or ``lfs_pointer``.

    An incomplete `GIT_LFS_SKIP_SMUDGE` clone can leave LFS entries either as
    small pointer files or absent entirely, and neither is a cuslide2 defect.
    """
    if not path.exists():
        return "missing"
    try:
        if path.stat().st_size <= 1024:
            with open(path, "rb") as f:
                if f.read(len(LFS_MAGIC)) == LFS_MAGIC:
                    return "lfs_pointer"
    except OSError:
        return "missing"
    return "ok"


def sha256_of(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_expected(root, path):
    """Return ``(oid, size)`` declared by *path*'s git-LFS pointer, or None.

    The committed blob for an LFS-tracked file *is* the pointer, so this
    recovers the authoritative size and hash for anything in the corpus
    without needing the per-directory ``index.yaml`` (which only a couple of
    directories have) or a YAML parser.
    """
    try:
        rel = path.relative_to(root).as_posix()
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{rel}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if blob.returncode != 0 or not blob.stdout.startswith(LFS_MAGIC):
        return None

    oid = size = None
    for line in blob.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split("oid sha256:", 1)[1].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split(None, 1)[1])
            except (IndexError, ValueError):
                pass
    return (oid, size) if size is not None else None


def check_integrity(root, path, verify_hash):
    """Compare *path* against its LFS pointer.

    Returns ``(status, detail)`` where status is ``ok`` or ``truncated``.  An
    interrupted LFS fetch leaves a short file that decodes into confusing
    partial failures, so this runs before any decode is blamed on the reader.
    The size comparison is free; hashing is opt-in because the corpus has
    multi-gigabyte slides.
    """
    expected = lfs_expected(root, path)
    if expected is None:
        return "ok", None
    oid, size = expected

    actual_size = path.stat().st_size
    if actual_size != size:
        return "truncated", (
            f"{actual_size} bytes on disk, pointer declares {size} "
            f"({actual_size / size:.0%} complete)"
        )

    if verify_hash and oid:
        actual = sha256_of(path)
        if actual != oid:
            return "truncated", (f"sha256 {actual[:16]}... != pointer {oid[:16]}...")
    return "ok", None


def discover(root, only_formats=None):
    """Collect candidate images grouped by expected format."""
    found = []
    for directory, (fmt, extensions) in sorted(CORPUS_LAYOUT.items()):
        if only_formats and fmt not in only_formats:
            continue
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            # `hamamatsu/ref` holds reference crops, not slides.
            if "ref" in path.relative_to(root).parts:
                continue
            found.append((fmt, path))
    return found


def probe(path, CuImage):
    """Open *path* and summarize what cuslide2 reports about it."""
    record = {"path": str(path), "size_bytes": path.stat().st_size}

    img = CuImage(str(path))
    resolutions = img.resolutions

    record.update(
        opened=True,
        shape=list(img.shape),
        dtype=str(img.dtype),
        level_count=resolutions.get("level_count", 0),
        level_dimensions=[list(d) for d in resolutions.get("level_dimensions", ())],
        level_tile_sizes=[list(t) for t in resolutions.get("level_tile_sizes", ())],
    )

    # Pixel spacing drives mpp-based resolution normalization downstream, so a
    # slide that decodes but reports no spacing is still a blocker for those
    # users.
    try:
        record["spacing"] = list(img.spacing())
    except Exception as exc:
        record["spacing_error"] = f"{type(exc).__name__}: {exc}"

    tiles = record["level_tile_sizes"]
    record["tiled"] = bool(tiles and tiles[0][0] > 0 and tiles[0][1] > 0)

    try:
        record["associated_images"] = sorted(img.associated_images)
    except Exception:
        record["associated_images"] = []

    # Smallest useful decode: proves the read path works at all.
    size = (min(256, img.shape[1]), min(256, img.shape[0]))
    region = np.asarray(img.read_region((0, 0), size, level=0))
    record["probe_read"] = {
        "requested": list(size),
        "shape": list(region.shape),
        "dtype": str(region.dtype),
        "all_zero": bool(not region.any()),
    }
    return record


def phase_open(root, CuImage, only_formats, limit, verify_checksums):
    """Open every candidate image and report per-format outcomes.

    A *CuImage* of ``None`` inventories the corpus without decoding, which is
    how ``--dry-run`` checks the harness without cuCIM or a GPU.
    """
    print("=" * 72)
    print("PHASE 1 - open every format cuslide2 accepts")
    print("=" * 72)

    candidates = discover(root, only_formats)
    if not candidates:
        print(f"No candidate images under {root}")
        return []

    if limit:
        candidates = candidates[:limit]

    results = []
    current_format = None
    for fmt, path in candidates:
        if fmt != current_format:
            print(f"\n--- {fmt} ---")
            current_format = fmt

        rel = path.relative_to(root)
        state = file_state(path)
        if state != "ok":
            note = (
                "not checked out"
                if state == "missing"
                else "git lfs pull to materialize"
            )
            print(f"  [{state:<8}] {rel}   ({note})")
            results.append({"format": fmt, "path": str(path), "status": state})
            continue

        record = {"format": fmt, "path": str(path)}

        integrity, detail = check_integrity(root, path, verify_checksums)
        if integrity != "ok":
            print(f"  [TRUNCATE] {rel}\n              {detail}")
            record.update(status="truncated", detail=detail)
            results.append(record)
            continue

        if CuImage is None:
            print(f"  [present ] {rel}   ({path.stat().st_size / 2**20:.1f} MB)")
            record["status"] = "not_decoded"
            results.append(record)
            continue

        try:
            record.update(probe(path, CuImage))
        except Exception as exc:
            print(f"  [FAIL    ] {rel}\n              {type(exc).__name__}: {exc}")
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            results.append(record)
            continue

        record["status"] = "ok"
        results.append(record)

        dims = record["level_dimensions"][0] if record["level_dimensions"] else "?"
        tile = record["level_tile_sizes"][0] if record["level_tile_sizes"] else "?"
        warn = "  <- decoded all zeros" if record["probe_read"]["all_zero"] else ""
        print(
            f"  [ok      ] {rel}\n"
            f"              {dims} px, {record['level_count']} levels, "
            f"tile {tile}, {record['dtype']}{warn}"
        )
    return results


def load_ref_plan(ref_dir):
    """Return the reference crop plan, preferring ``manifest.json``.

    The manifest states slide, IFD, origin, size, shape and dtype outright.
    Filename parsing is a fallback for corpora predating it, and cannot
    recover the exact slide name (the manifest maps ``test3_DAPI_2_387_`` back
    to ``test3-DAPI 2 (387) .ndpi``, which no normalization reliably does).
    """
    manifest = ref_dir / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text()), "manifest.json"

    plan = {}
    for ref in sorted(ref_dir.glob("*.raw")):
        match = REF_PATTERN.match(ref.name)
        if not match:
            continue
        info = match.groupdict()
        w, h = int(info["w"]), int(info["h"])
        plan[ref.name] = {
            "slide": info["slide"].replace("_", "-") + ".ndpi",
            "ifd": int(info["ifd"]),
            "x": int(info["x"]),
            "y": int(info["y"]),
            "w": w,
            "h": h,
            "shape": [h, w, 3],
            "dtype": "uint8",
        }
    return plan, "filename parsing (no manifest.json)"


def phase_refs(root, CuImage, limit, max_ref_bytes, tolerance):
    """Diff NDPI reads against the golden reference crops."""
    print()
    print("=" * 72)
    print("PHASE 2 - NDPI reads vs golden reference crops")
    print("=" * 72)

    ref_dir = root / "hamamatsu" / "ref"
    if not ref_dir.is_dir():
        print(f"No reference crops at {ref_dir}")
        return []

    plan, source = load_ref_plan(ref_dir)
    if not plan:
        print(f"No reference crop plan found in {ref_dir}")
        return []
    print(f"{len(plan)} reference crops from {source}")
    print(f"Tolerance: max abs delta <= {tolerance} counts as 'close'\n")

    results = []
    open_slides = {}
    for name in sorted(plan) if not limit else sorted(plan)[:limit]:
        spec = plan[name]
        ifd, x, y = spec["ifd"], spec["x"], spec["y"]
        w, h = spec["w"], spec["h"]
        label = f"{spec['slide']} ifd{ifd} ({x},{y}) {w}x{h}"

        ref_path = ref_dir / name
        slide_path = root / "hamamatsu" / spec["slide"]
        record = {
            "ref": name,
            "slide": spec["slide"],
            "ifd": ifd,
            "location": [x, y],
            "size": [w, h],
        }

        states = {file_state(ref_path), file_state(slide_path)}
        if "missing" in states or "lfs_pointer" in states:
            state = "missing" if "missing" in states else "lfs_pointer"
            print(f"  [{state:<8}] {label}")
            record["status"] = state
            results.append(record)
            continue

        # A short slide or crop yields bogus diffs, so rule that out first.
        # Size-only here; --verify-checksums hashing happens in the open phase.
        for candidate in (slide_path, ref_path):
            integrity, detail = check_integrity(root, candidate, False)
            if integrity != "ok":
                print(f"  [TRUNCATE] {label}\n              {candidate.name}: {detail}")
                record.update(status="truncated", detail=detail)
                break
        if record.get("status") == "truncated":
            results.append(record)
            continue

        if max_ref_bytes and ref_path.stat().st_size > max_ref_bytes:
            print(
                f"  [skip    ] {label}   "
                f"({ref_path.stat().st_size / 2**20:.0f} MB > --max-ref-mb)"
            )
            record["status"] = "too_large"
            results.append(record)
            continue

        if CuImage is None:
            print(f"  [mapped  ] {label} -> {name}")
            record["status"] = "not_decoded"
            results.append(record)
            continue

        try:
            if slide_path not in open_slides:
                open_slides[slide_path] = CuImage(str(slide_path))
            img = open_slides[slide_path]

            # cuslide2 indexes read_region() location in the target level's own
            # coordinate space (ifd.cpp maps `location` straight onto that
            # IFD's tile grid).  generate_refs.py crops the same way, slicing
            # `pages[ifd].aszarr()` directly, so the origins line up as-is.
            region = np.asarray(img.read_region((x, y), (w, h), level=ifd))
            expected = np.frombuffer(
                ref_path.read_bytes(), dtype=np.dtype(spec["dtype"])
            ).reshape(spec["shape"])

            record["actual_shape"] = list(region.shape)
            record["expected_shape"] = list(expected.shape)

            if region.shape != expected.shape:
                record["status"] = "shape_mismatch"
                print(
                    f"  [SHAPE   ] {label}\n"
                    f"              got {region.shape}, want {expected.shape}"
                )
                results.append(record)
                continue

            if np.array_equal(region, expected):
                record["status"] = "exact"
                print(f"  [exact   ] {label}")
                results.append(record)
                continue

            diff = np.abs(region.astype(np.int32) - expected.astype(np.int32))
            differing = int(np.count_nonzero(diff))
            record.update(
                max_abs_diff=int(diff.max()),
                mean_abs_diff=float(diff.mean()),
                differing_values=differing,
                differing_fraction=differing / diff.size,
            )
            # Small deltas are chroma rounding; large ones mean the wrong
            # pixels came back.
            if diff.max() <= tolerance:
                record["status"] = "close"
                print(
                    f"  [close   ] {label}   max {diff.max()}, "
                    f"{differing / diff.size:.1%} of values differ"
                )
            else:
                record["status"] = "mismatch"
                print(
                    f"  [DIFF    ] {label}\n"
                    f"              max {diff.max()}, mean {diff.mean():.3f}, "
                    f"{differing / diff.size:.1%} of values differ"
                )
        except Exception as exc:
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            print(f"  [FAIL    ] {label}\n              {type(exc).__name__}: {exc}")

        results.append(record)
    return results


FAILURE_STATUSES = {"error", "mismatch", "shape_mismatch", "truncated"}


def summarize(open_results, ref_results):
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if open_results:
        by_format = {}
        for r in open_results:
            bucket = by_format.setdefault(r.get("format", "?"), {})
            bucket[r["status"]] = bucket.get(r["status"], 0) + 1
        print("\nOpen phase, files by status:")
        for fmt in sorted(by_format):
            counts = ", ".join(
                f"{status}={n}" for status, n in sorted(by_format[fmt].items())
            )
            print(f"  {fmt:<18} {counts}")

    if ref_results:
        counts = {}
        for r in ref_results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print("\nReference phase, crops by status:")
        for status in sorted(counts):
            print(f"  {status:<18} {counts[status]}")

    unavailable = sum(
        1
        for r in open_results + ref_results
        if r["status"] in {"missing", "lfs_pointer"}
    )
    if unavailable:
        print(
            f"\n{unavailable} item(s) were not present locally and were not tested.\n"
            "Fetch them in the corpus, e.g.:\n"
            '  git lfs pull --include="hamamatsu/CMU-1.ndpi,hamamatsu/ref/CMU_1_*"'
        )

    return [r for r in open_results + ref_results if r["status"] in FAILURE_STATUSES]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=os.getenv(ENV_VAR),
        help=f"Corpus root (default: ${ENV_VAR})",
    )
    parser.add_argument(
        "--phase",
        choices=("open", "refs", "all"),
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        help="Restrict the open phase to this format label (repeatable)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap items per phase (0 = no cap)"
    )
    parser.add_argument(
        "--max-ref-mb",
        type=int,
        default=64,
        help="Skip reference crops larger than this (default: 64, 0 = no cap)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Max per-value delta still counted as 'close' (default: 2)",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Also sha256 every slide against its LFS pointer, not just "
        "compare sizes (slow on multi-GB slides)",
    )
    parser.add_argument("--report", help="Write a JSON report to this path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory the corpus and resolve reference crops without "
        "decoding, so the harness can be checked without cuCIM or a GPU",
    )
    args = parser.parse_args()

    if not args.root:
        print(
            f"{ENV_VAR} is not set, so there is nothing to validate.\n\n"
            "Clone the corpus and point the variable at it:\n"
            "  GIT_LFS_SKIP_SMUDGE=1 git clone \\\n"
            "    ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/"
            "ImageCodecs/imagecodecs-test-images-large.git\n"
            f"  export {ENV_VAR}=$PWD/imagecodecs-test-images-large"
        )
        return 0

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"Corpus root does not exist: {root}")
        return 2

    CuImage = None
    if args.dry_run:
        print("Dry run: inventorying only, nothing will be decoded.\n")
    else:
        plugin_lib = setup_environment("cucim_pyramid_validation")
        try:
            from cucim.clara import _set_plugin_root

            _set_plugin_root(plugin_lib)
            from cucim import CuImage
        except ImportError as exc:
            print(
                f"cuCIM is not importable here ({exc}).\n"
                "Build/install cuCIM with the cuslide2 plugin, or pass "
                "--dry-run to check the harness without decoding."
            )
            return 2

    print(f"Corpus: {root}\n")

    open_results = []
    ref_results = []
    if args.phase in ("open", "all"):
        open_results = phase_open(
            root, CuImage, args.formats, args.limit, args.verify_checksums
        )
    if args.phase in ("refs", "all"):
        ref_results = phase_refs(
            root, CuImage, args.limit, args.max_ref_mb * 2**20, args.tolerance
        )

    failures = summarize(open_results, ref_results)

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {"root": str(root), "open": open_results, "refs": ref_results},
                indent=2,
            )
        )
        print(f"\nJSON report written to {args.report}")

    # Fact-finding pass: nonzero means "found something to look at", not
    # "the script broke".
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
