#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Basic tests of the cuslide2 plugin on synthetic TIFF files.

The images are written with tifffile, so no sample data has to be downloaded.
The plugin is selected with ``ENABLE_CUSLIDE2=1``, which is read when the
plugins are loaded, so every check runs in a new Python process.

The tests are skipped when the package has no cuslide2 plugin. If it is
present, it is expected to work: a missing nvImageCodec, nvJPEG2000 or nvTIFF
library is a failure, not a reason to skip.
"""

import glob
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
imagecodecs = pytest.importorskip("imagecodecs")

TILE = 128
SHAPE = (512, 640, 3)  # (height, width, channels) of the base level


def _has_cuslide2_plugin():
    from cucim.clara import _get_plugin_root

    return bool(
        glob.glob(os.path.join(_get_plugin_root(), "cucim.kit.cuslide2@*.so"))
    )


pytestmark = pytest.mark.skipif(
    not _has_cuslide2_plugin(), reason="cucim.kit.cuslide2 is not installed"
)


def _make_image(shape=SHAPE):
    """A smooth, non-constant image (lossy codecs keep it within a few levels)."""
    h, w, c = shape
    y, x = np.mgrid[:h, :w]
    chans = [
        128 + 100 * np.sin(x / 37.0 + k) * np.cos(y / 53.0 - k)
        for k in range(c)
    ]
    return np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)


def _write_pyramid(path, image, compression, compressionargs=None, n_levels=3):
    """Write a tiled, pyramidal (one reduced-resolution IFD per level) RGB TIFF."""
    with tifffile.TiffWriter(path) as tif:
        for level in range(n_levels):
            data = image[:: 2**level, :: 2**level]
            tif.write(
                data,
                photometric="rgb",
                tile=(TILE, TILE),
                compression=compression,
                compressionargs=compressionargs,
                subfiletype=0 if level == 0 else 1,
            )


# (id, compression, compressionargs, max mean absolute error)
CODECS = [
    ("deflate", "adobe_deflate", None, 0),
    ("lzw", "lzw", None, 0),
    ("jpeg", "jpeg", {"level": 95}, 4),
    ("jpeg2000", "jpeg2000", {"level": 0}, 1),
]

_CHECK = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from cucim import CuImage

    path = sys.argv[1]
    img = CuImage(path)
    res = img.resolutions
    out = {
        "level_count": res["level_count"],
        "level_dimensions": [list(d) for d in res["level_dimensions"]],
        "level_tile_sizes": [list(d) for d in res["level_tile_sizes"]],
        "regions": {},
    }
    for device in ("cpu", "cuda"):
        region = img.read_region((100, 70), (200, 150), level=0, device=device)
        if device == "cuda":
            import cupy as cp
            arr = cp.asnumpy(cp.asarray(region))
        else:
            arr = np.asarray(region)
        out["regions"][device] = arr.tolist()
    print("RESULT" + json.dumps(out))
    """
)


def _run_cuslide2(path):
    env = dict(os.environ, ENABLE_CUSLIDE2="1")
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT") :])


@pytest.mark.parametrize(
    "compression,compressionargs,max_mae",
    [c[1:] for c in CODECS],
    ids=[c[0] for c in CODECS],
)
def test_cuslide2_reads_synthetic_pyramid(
    tmp_path, compression, compressionargs, max_mae
):
    image = _make_image()
    path = tmp_path / "pyramid.tif"
    _write_pyramid(path, image, compression, compressionargs)

    result = _run_cuslide2(path)

    # The pyramid, with the tile size of each level (not zeros: the tile size
    # is only known when the TIFF tags were read).
    assert result["level_count"] == 3
    assert result["level_dimensions"] == [
        [image.shape[1] // 2**k, image.shape[0] // 2**k] for k in range(3)
    ]
    assert result["level_tile_sizes"] == [[TILE, TILE]] * 3

    # The pixels, on the CPU and on the GPU. A decoder that is not available
    # must not give an all-zero region without an error.
    expected = image[70:220, 100:300]
    for device, region in result["regions"].items():
        region = np.asarray(region)
        assert region.shape == expected.shape, device
        assert region.any(), f"all-zero region returned on {device}"
        mae = np.abs(region.astype(int) - expected.astype(int)).mean()
        assert mae <= max_mae, (device, mae)


_CHECK_READ = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from cucim import CuImage

    img = CuImage(sys.argv[1])
    out = {}
    for device in ("cpu", "cuda"):
        try:
            region = img.read_region((20, 30), (100, 80), level=0, device=device)
        except RuntimeError as e:
            out[device] = {"error": str(e)}
            continue
        if device == "cuda":
            import cupy as cp
            region = cp.asnumpy(cp.asarray(region))
        out[device] = {"region": np.asarray(region).tolist()}
    print("RESULT" + json.dumps(out))
    """
)


@pytest.mark.parametrize(
    "compression",
    [33003, 33005, 34712],
    ids=["aperio_33003", "aperio_33005", "jpeg2000_34712"],
)
def test_cuslide2_jpeg2000_is_decoded_or_an_error(tmp_path, compression):
    """A JPEG 2000 TIFF is decoded correctly or the read raises an error.

    If the nvJPEG2000 or nvTIFF library cannot be loaded, nvImageCodec falls
    back to CPU decoders that cannot decode JPEG 2000 in TIFF but report
    success. The plugin must not return an all-zero region then.
    """
    image = _make_image((256, 320, 3))
    path = tmp_path / "jpeg2000.tif"
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(TILE, TILE),
        compression=compression,
    )

    env = dict(os.environ, ENABLE_CUSLIDE2="1")
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK_READ, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    result = json.loads(lines[-1][len("RESULT") :])

    expected = image[30:110, 20:120]
    for device, outcome in result.items():
        if "error" in outcome:
            assert "nvImageCodec" in outcome["error"], outcome["error"]
            continue
        region = np.asarray(outcome["region"])
        assert region.shape == expected.shape, device
        mae = np.abs(region.astype(int) - expected.astype(int)).mean()
        assert region.any() and mae <= 1, (
            f"{device}: region is blank or wrong (mean abs error {mae})"
        )
