# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuCIM</div>


[RAPIDS](https://rapids.ai) cuCIM is an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cucim/blob/main/README.md) ensure you are on the `main` branch.

- [GTC 2021 cuCIM: A GPU Image I/O and Processing Toolkit [S32194]](https://www.nvidia.com/en-us/gtc/catalog/?search=cuCIM#/)
  - [video](https://gtc21.event.nvidia.com/media/cuCIM%3A%20A%20GPU%20Image%20I_O%20and%20Processing%20Toolkit%20%5BS32194%5D/1_fwfxd0iu)
- [SciPy 2021 cuCIM - A GPU image I/O and processing library](https://www.scipy2021.scipy.org/)
  - [video](https://youtu.be/G46kOOM9xbQ)

**Release notes** are available on our [wiki page](https://github.com/rapidsai/cucim/wiki/Release-Notes).

## Install cuCIM

### Conda

#### Conda (stable)

> conda create -n cucim -c rapidsai -c conda-forge cucim cudatoolkit=`<CUDA version>`

`<CUDA version>` should be 11.0+ (e.g., `11.0`, `11.2`, etc.)

#### Conda (nightlies)

> conda create -n cucim -c rapidsai-nightly -c conda-forge cucim cudatoolkit=`<CUDA version>`

`<CUDA version>` should be 11.0+ (e.g., `11.0`, `11.2`, etc)

### Notebooks

Please check out our [Welcome](notebooks/Welcome.ipynb) notebook ([NBViewer](https://nbviewer.jupyter.org/github/rapidsai/cucim/blob/branch-21.06/notebooks/Welcome.ipynb))

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

Copyright (c) 2020-2021, NVIDIA CORPORATION.
