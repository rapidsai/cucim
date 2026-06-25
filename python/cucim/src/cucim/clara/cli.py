#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mcucim` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``cucim.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``cucim.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import logging
import os
from pathlib import Path

import click


@click.group()
def main():
    """nothing for now"""
    pass


@main.command()
@click.argument("src_file", type=click.Path(exists=True))
@click.argument(
    "dest_folder",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=Path("."),
)
@click.option("--tile-size", type=int, default=256)
@click.option("--overlap", type=int, default=0)
@click.option("--num-workers", type=int, default=os.cpu_count())
@click.option("--compression", type=str, default="jpeg")
@click.option("--output-filename", type=str, default="image.tif")
def convert(
    src_file,
    dest_folder,
    tile_size,
    overlap,
    num_workers,
    compression,
    output_filename,
):
    """Convert file format"""
    from .converter import tiff

    logging.basicConfig(level=logging.INFO)

    compression = compression.lower()
    if compression in ["raw", "none"]:
        compression = None

    tiff.svs2tif(
        src_file,
        Path(dest_folder),
        tile_size,
        overlap,
        num_workers,
        compression,
        output_filename,
    )
