# [cuCIM](https://github.com/rapidsai/cucim)

<!-- start-include-here -->

The [RAPIDS](https://rapids.ai) [cuCIM](https://github.com/rapidsai/cucim) is an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging.

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cucim/blob/main/README.md) ensure you are on the `main` branch.

## Quick Start

### Install cuCIM

```
pip install cucim
```

### Open Image

```python
from cucim import CuImage
img = CuImage('image.tif')
```

### See Metadata

```python
import json
print(img.is_loaded)        # True if image data is loaded & available.
print(img.device)           # A device type.
print(img.ndim)             # The number of dimensions.
print(img.dims)             # A string containing a list of dimensions being requested.
print(img.shape)            # A tuple of dimension sizes (in the order of `dims`).
print(img.size('XYC'))      # Returns size as a tuple for the given dimension order.
print(img.dtype)            # The data type of the image.
print(img.channel_names)    # A channel name list.
print(img.spacing())        # Returns physical size in tuple.
print(img.spacing_units())  # Units for each spacing element (size is same with `ndim`).
print(img.origin)           # Physical location of (0, 0, 0) (size is always 3).
print(img.direction)        # Direction cosines (size is always 3x3).
print(img.coord_sys)        # Coordinate frame in which the direction cosines are 
                            # measured. Available Coordinate frame is not finalized yet.

# Returns a set of associated image names.
print(img.associated_images)
# Returns a dict that includes resolution information.
print(json.dumps(img.resolutions, indent=2))
# A metadata object as `dict`
print(json.dumps(img.metadata, indent=2))
# A raw metadata string.
print(img.raw_metadata) 
```

### Read Region

```python
from matplotlib import pyplot as plt
def visualize(image):
    dpi = 80.0
    height, width, _ = image.shape
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.axis('off')
    plt.imshow(image)

```

```python
import numpy as np

# Read whole slide at the lowest resolution
resolutions = img.resolutions
level_count = resolutions["level_count"]

# Note: ‘level’ is at 3rd parameter (OpenSlide has it at 2nd parameter)
#   `location` is level-0 based coordinates (using the level-0 reference frame)
#   If `size` is not specified, size would be (width, height) of the image at the specified `level`.
region = img.read_region(location=(10000, 10000), size=(512, 512), level=level_count-1)

visualize(region)
#from PIL import Image
#Image.fromarray(np.asarray(region))
```

## Acknowledgments

Without awesome third-party open source software, this project wouldn't exist.

Please find `LICENSE-3rdparty.md` to see which third-party open source software
is used in this project.

## License

Apache-2.0 License (see `LICENSE` file).

Copyright (c) 2020-2021, NVIDIA CORPORATION.
