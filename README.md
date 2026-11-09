# NVIDIA cuCIM

NVIDIA cuCIM (pronounced "koo-sim", see [here]( https://ipa-reader.com/?text=ku%CB%90%CB%88s%C9%AAm&voice=Joey )) is an open-source, accelerated computer vision and image processing software library for multidimensional images used in biomedical, geospatial, material and life science, and remote sensing use cases.

cuCIM offers:

- Enhanced Image Processing Capabilities for large and n-dimensional tag image file format (TIFF) files
- Accelerated performance through Graphics Processing Unit (GPU)-based image processing and computer vision primitives
- A Straightforward Pythonic Interface with Matching Application Programming Interface (API) for Openslide

cuCIM supports the following formats:

- Aperio ScanScope Virtual Slide (SVS)
- Philips TIFF
- Generic Tiled, Multi-resolution RGB TIFF files with the following compression schemes:
  - No Compression
  - JPEG
  - JPEG2000
  - Lempel-Ziv-Welch (LZW)
  - Deflate

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cucim/blob/main/README.md) ensure you are on the `main` branch.

- [GTC 2022 Accelerating Storage IO to GPUs with Magnum IO [S41347]](https://events.rainfocus.com/widget/nvidia/gtcspring2022/sessioncatalog/session/1634960000577001Etxp)
  - cuCIM's GDS API examples: <https://github.com/NVIDIA/MagnumIO/tree/main/gds/readers/cucim-gds>
- [SciPy 2021 cuCIM - A GPU image I/O and processing library](https://www.scipy2021.scipy.org/)
  - [video](https://youtu.be/G46kOOM9xbQ)
- [GTC 2021 cuCIM: A GPU Image I/O and Processing Toolkit [S32194]](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=cucim&sort=date)
  - [video](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32194/)

**[Developer Page](https://developer.nvidia.com/multidimensional-image-processing)**

**Blogs**
- [Enhanced Image Analysis with Multidimensional Image Processing](https://developer.nvidia.com/blog/enhanced-image-analysis-with-multidimensional-image-processing/)
- [Accelerating Scikit-Image API with cuCIM: n-Dimensional Image Processing and IO on GPUs](https://developer.nvidia.com/blog/cucim-rapid-n-dimensional-image-processing-and-i-o-on-gpus/)
- [Accelerating Digital Pathology Pipelines with NVIDIA Clara™ Deploy](https://developer.nvidia.com/blog/accelerating-digital-pathology-pipelines-with-nvidia-clara-deploy-2/)

**Webinars**

- [cuCIM: a GPU Image IO and Processing Library](https://www.youtube.com/watch?v=G46kOOM9xbQ)

**[Documentation](https://docs.nvidia.com/cucim)**

**Release notes** are available on our [wiki page](https://github.com/rapidsai/cucim/wiki/Release-Notes).

## Install cuCIM

### Conda

#### [Conda (stable)](https://anaconda.org/rapidsai/cucim)

```bash
conda create -n cucim -c rapidsai -c conda-forge cucim cuda-version=`<CUDA version>`
```

`<CUDA version>` should be 12.0+ (e.g., `12.0`, etc.)

#### [Conda (nightlies)](https://anaconda.org/rapidsai-nightly/cucim)

```bash
conda create -n cucim -c rapidsai-nightly -c conda-forge cucim cuda-version=`<CUDA version>`
```

`<CUDA version>` should be 12.0+ (e.g., `12.0`, etc.)

### [PyPI](https://pypi.org/project/cucim/)

Install for CUDA 12:

```bash
pip install cucim-cu12
```

Install for CUDA 13:

```bash
pip install cucim-cu13
```

### Notebooks

Please check out our [Welcome](notebooks/Welcome.ipynb) notebook ([NBViewer](https://nbviewer.org/github/rapidsai/cucim/blob/main/notebooks/Welcome.ipynb))

#### Downloading sample images

To download images used in the notebooks, please execute the following commands from the repository root folder to copy sample input images into `notebooks/input` folder:

(You will need [Docker](https://www.docker.com/) installed in your system)

```bash
./run download_testdata
```
or

```bash
mkdir -p notebooks/input
tmp_id=$(docker create gigony/svs-testdata:little-big)
docker cp $tmp_id:/input notebooks
docker rm -v ${tmp_id}
```

## cuslide2 Plugin

> **Note:** `cuslide` is still the default plugin. `cuslide2` is opt-in (see below)
> and is included in the pip wheels and in the conda packages (`libcucim`). It uses
> [nvImageCodec](https://developer.nvidia.com/nvimagecodec) for GPU-accelerated
> TIFF decoding and adds tile-level caching, async batch decoding, and broader
> format support. The original `cuslide` plugin is planned to be deprecated in favor
> of `cuslide2`, so please plan to migrate existing workflows to `cuslide2`.

### Enabling cuslide2

To enable the cuslide2 plugin, set the `ENABLE_CUSLIDE2` environment variable:

```bash
ENABLE_CUSLIDE2=1 python your_script.py
```

### Requirements

The plugin decodes with [nvImageCodec](https://developer.nvidia.com/nvimagecodec)
(`>=0.9.0,<0.10.0`) and needs its nvJPEG2000 and nvTIFF libraries to read JPEG 2000
and TIFF files, in addition to nvJPEG for JPEG.

- **pip:** `pip install cucim-cu13` (or `cucim-cu12`) installs
  `nvidia-nvimgcodec-cu13[nvjpeg,nvjpeg2k,nvtiff]` (`-cu12` for CUDA 12). The plugin
  finds the nvImageCodec library in the `nvidia/nvimgcodec` directory of
  `site-packages`, and `LD_LIBRARY_PATH` does not have to be set. If you
  installed `nvidia-nvimgcodec-cu13` without these extras, install them with
  `pip install "nvidia-nvimgcodec-cu13[nvjpeg,nvjpeg2k,nvtiff]"`. Without them,
  nvImageCodec prints `Failed to load nvtiff library!` or
  `Failed to load nvjpeg2k library!`, and cuslide2 reports the wrong number of
  levels and tile sizes of `(0, 0)` or fails to read JPEG 2000 files.

### Test Scripts

The test scripts are in the cuCIM repository (`scripts/`) and are not installed
with the packages. They use the cuslide2 plugin of a source build in the checkout
if there is one, and the plugin of the installed package otherwise.
They fail if the tile sizes of the image are `(0, 0)` or if all decoded tiles are zero.

cuCIM provides test scripts to verify cuslide2 functionality with different TIFF formats:

**Aperio SVS files:**

```bash
# Download a sample SVS file and run the test
python scripts/test_aperio_svs.py --download

# Or test with your own SVS file
ENABLE_CUSLIDE2=1 python scripts/test_aperio_svs.py /path/to/your/file.svs
```

**Philips TIFF files:**

```bash
# Test with a Philips TIFF file
ENABLE_CUSLIDE2=1 python scripts/test_philips_tiff.py /path/to/your/philips.tiff

# List available test options
python scripts/test_philips_tiff.py --help
```

## Build/Install from Source

See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing Guide

Contributions to cuCIM are more than welcome!
Please review the [CONTRIBUTING.md](https://github.com/rapidsai/cucim/blob/main/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.

## Acknowledgments

Without awesome third-party open source software, this project wouldn't exist.

Please find [LICENSE-3rdparty.md](LICENSE-3rdparty.md) to see which third-party open source software
is used in this project.

## License

Apache-2.0 License (see [LICENSE](LICENSE) file).

Copyright (c) 2020-2026, NVIDIA CORPORATION.
