import math
import cupy as cp
import numpy as np

import colorcet
import matplotlib.pyplot as plt

from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.color import label2rgb
from cucim.skimage.segmentation import relabel_sequential


def coords_to_labels(coords):
    """
    Convert coordinate output of distance_transform_edt to unique region
    labels.
    """
    if coords.shape[0] != 2:
        raise ValueError("this utility function assumes 2D coordinates")
    # create a set of unique integer labels based on coordinates
    labels = coords[1] + (coords[0].max() + 1)
    labels += coords[0]
    # convert to sequential labels
    return relabel_sequential(labels)[0]


def hexstr_to_rgb(hex):
    """Convert a color specified as a hex strings to a normalized RGB 3-tuple.

    For example, "#8c3bff" -> (0.5490, 0.2314, 1.0)
    """
    hex = hex.lstrip("#")
    return tuple(int(hex[i:i + 2], 16)/255. for i in (0, 2, 4))


shape = (200, 200)
size = math.prod(shape)
ntrue = .001 * size
p_true = ntrue / size
p_false = 1 - p_true

# generate a sparse set of background points
cp.random.seed(123)
image = cp.random.choice([0, 1], size=shape, p=(p_false, p_true))

distances, coords = distance_transform_edt(
    image == 0, return_distances=True, return_indices=True
)
# plt.figure(); plt.show(distances.get()); plt.show()
# create "labels" image based on locations of unique coordinates

labels = coords_to_labels(coords)

# Note: The code above this point should be fast on the GPU, but the
#       code below for visualizing the colored Voronoi cells has not been
#       optimized and may run slowly for larger image sizes.

# create a suitable RGB colormap (using method of Glasbey et. al.)
n_labels = int(labels.max())
color_list = colorcet.glasbey[:n_labels]
color_list = [hexstr_to_rgb(c) for c in color_list]

# colorize the labels image
rgb_labels = label2rgb(labels, colors=color_list)

# copy to host and visualize results
image, distances, coords, rgb_labels = map(
    cp.asnumpy, (image, distances, coords, rgb_labels)
)

# set original point locations in rgb_labels to white
xx, yy = np.where(image)
for x, y in zip(xx, yy):
    rgb_labels[x, y, :] = 1

fig, axes = plt.subplots(2, 3, figsize=(8, 7))
axes[0][0].imshow(image, cmap=plt.cm.gray)
axes[0][0].set_title('seed points')
axes[0][1].imshow(distances, cmap=plt.cm.gray)
axes[0][1].set_title('Euclidean distance\n(to nearest seed)')
axes[1][0].imshow(coords[0], cmap=plt.cm.gray)
axes[1][0].set_title('y coordindate\nof neareset seed')
axes[1][1].imshow(coords[1], cmap=plt.cm.gray)
axes[1][1].set_title('x coordindate\nof neareset seed')
axes[1][2].imshow(rgb_labels)
axes[1][2].set_title('discrete Voronoi')
for ax in axes.ravel():
    ax.set_axis_off()
# overlay larger markers at the seed points for better visibility
for x, y in zip(xx, yy):
    # overlay in image
    axes[0, 0].plot(y, x, 'w.')
    # overlay in rgb_labels
    axes[1, 2].plot(y, x, 'w.')
plt.tight_layout()
plt.show()
