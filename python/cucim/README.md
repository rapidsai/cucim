# [cuCIM](https://github.com/rapidsai/cucim)

<!-- start-include-here -->

[RAPIDS](https://rapids.ai) [cuCIM](https://github.com/rapidsai/cucim) is an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cucim/blob/main/README.md) ensure you are on the `main` branch.

- [GTC 2022 Accelerating Storage IO to GPUs with Magnum IO [S41347]](https://events.rainfocus.com/widget/nvidia/gtcspring2022/sessioncatalog/session/1634960000577001Etxp)
  - cuCIM's GDS API examples: <https://github.com/NVIDIA/MagnumIO/tree/main/gds/readers/cucim-gds>
- [SciPy 2021 cuCIM - A GPU image I/O and processing library](https://www.scipy2021.scipy.org/)
  - [video](https://youtu.be/G46kOOM9xbQ)
- [GTC 2021 cuCIM: A GPU Image I/O and Processing Toolkit [S32194]](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=cucim&sort=date)
  - [video](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32194/)


## Quick Start

### Install cuCIM

```bash
pip install cucim

# Install dependencies for `cucim.skimage` (assuming that CUDA 11.0 is used for CuPy)
pip install scipy scikit-image cupy-cuda110
```

### Jupyter Notebooks

Please check out our [Welcome](https://github.com/rapidsai/cucim/blob/branch-22.10/notebooks/Welcome.ipynb) notebook.

### Open Image

cuCIM's dataloader(`cucim.CuImage` class) is currently supporting Digital-pathology specific image formats (tiled multi-resolution raw/deflate/lzw/JPEG/JPEG2000-compressed TIFF-like RGB images such as Generic TIFF, Philips TIFF, Aperio SVS format).

```python
from cucim import CuImage
img = CuImage('image.tif')
```

### See Metadata

```python
import json
print(img.is_loaded)        # True if image data is loaded & available.
print(img.device)           # A device type.
print(img.ndim)             # The number of dimensions.
print(img.dims)             # A string containing a list of dimensions being requested.
print(img.shape)            # A tuple of dimension sizes (in the order of `dims`).
print(img.size('XYC'))      # Returns size as a tuple for the given dimension order.
print(img.dtype)            # The data type of the image.
print(img.channel_names)    # A channel name list.
print(img.spacing())        # Returns physical size in tuple.
print(img.spacing_units())  # Units for each spacing element (size is same with `ndim`).
print(img.origin)           # Physical location of (0, 0, 0) (size is always 3).
print(img.direction)        # Direction cosines (size is always 3x3).
print(img.coord_sys)        # Coordinate frame in which the direction cosines are
                            # measured. Available Coordinate frame is not finalized yet.

# Returns a set of associated image names.
print(img.associated_images)
# Returns a dict that includes resolution information.
print(json.dumps(img.resolutions, indent=2))
# A metadata object as `dict`
print(json.dumps(img.metadata, indent=2))
# A raw metadata string.
print(img.raw_metadata)
```

### Read Region

```python
# Install matplotlib (`pip install matplotlib`) if not installed before.
from matplotlib import pyplot as plt
def visualize(image):
    dpi = 80.0
    height, width, _ = image.shape
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.axis('off')
    plt.imshow(image)

```

```python
import numpy as np

# Read whole slide at the highest resolution
resolutions = img.resolutions
level_count = resolutions['level_count']  # level: 0 ~ (level_count - 1)

# Note: ‘level’ is at 3rd parameter (OpenSlide has it at 2nd parameter)
#   `location` is level-0 based coordinates (using the level-0 reference frame)
#   If `size` is not specified, size would be (width, height) of the image at the specified `level`.
region = img.read_region(location=(5000, 5000), size=(512, 512), level=0)

visualize(region)
#from PIL import Image
#Image.fromarray(np.asarray(region))
```

Aperio SVS (.svs) image format is supported since cuCIM [v21.10.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.10.01#support-aperio-svs-svs).

Please check [this notebook](https://nbviewer.org/github/rapidsai/cucim/blob/branch-22.10/notebooks/Supporting_Aperio_SVS_Format.ipynb) to see how to use the feature.

### Using Cache

Please look at this [notebook](https://nbviewer.jupyter.org/github/rapidsai/cucim/blob/branch-22.10/notebooks/Using_Cache.ipynb).

### Accessing File with GDS

Please look at this [notebook](https://nbviewer.jupyter.org/github/rapidsai/cucim/blob/branch-22.10/notebooks/Accessing_File_with_GDS.ipynb).

### NVTX Support for Performance Analysis

Please look at [this release note](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.00#2-add-nvtx-support-for-performance-analysis).

### Supporting Multithreading and Batch Processing

Please look at [this release note](https://github.com/rapidsai/cucim/wiki/release_notes_v22.02.00#2-supporting-multithreading-and-batch-processing).

### Using scikit-image API

Import `cucim.skimage` instead of `skimage`.

```python
# The following code is modified from https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py
#
import cupy as cp  # modified from: `import numpy as np`
import matplotlib.pyplot as plt

# from skimage import data
from cucim.skimage.color import rgb2hed, hed2rgb  # modified from: `from skimage.color import rgb2hed, hed2rgb`

# Load image (You can download sample images from https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/)
from cucim import CuImage
img = CuImage("CMU-1.svs")
region = img.read_region((30000, 10000), (256, 256))

# Example IHC image
ihc_rgb = cp.asarray(region)  # modified from: `ihc_rgb = data.immunohistochemistry()`

# Separate the stains from the IHC image
ihc_hed = rgb2hed(ihc_rgb)

# Create an RGB image for each of the stains
null = cp.zeros_like(ihc_hed[:, :, 0])  # np -> cp
ihc_h = hed2rgb(cp.stack((ihc_hed[:, :, 0], null, null), axis=-1))  # np -> cp
ihc_e = hed2rgb(cp.stack((null, ihc_hed[:, :, 1], null), axis=-1))  # np -> cp
ihc_d = hed2rgb(cp.stack((null, null, ihc_hed[:, :, 2]), axis=-1))  # np -> cp

# Display
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb.get())  # appended `.get()`
ax[0].set_title("Original image")

ax[1].imshow(ihc_h.get())  # appended `.get()`
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_e.get())  # appended `.get()`
ax[2].set_title("Eosin")

ax[3].imshow(ihc_d.get())  # appended `.get()`
ax[3].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
```

## Acknowledgments

Without awesome third-party open source software, this project wouldn't exist.

Please find `LICENSE-3rdparty.md` to see which third-party open source software
is used in this project.

## License

Apache-2.0 License (see `LICENSE` file).

Copyright (c) 2020-2022, NVIDIA CORPORATION.
