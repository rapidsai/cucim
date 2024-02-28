import math

import cupy as cp
import numpy as np

try:
    import colorcet
    import matplotlib.pyplot as plt
except ImportError as e:
    print("This demo requires the matplotlib and colorcet packages.")
    raise (e)

from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.color import label2rgb
from cucim.skimage.segmentation import relabel_sequential
from skimage import data


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


shape = (200, 200)
size = math.prod(shape)
ntrue = 0.001 * size
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

# Colorize the labels image, using a suitable categorical colormap
rgb_labels = label2rgb(labels, colors=colorcet.cm.glasbey.colors)

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
axes[0][0].set_title("seed points")
axes[0][1].imshow(distances, cmap=plt.cm.gray)
axes[0][1].set_title("Euclidean distance\n(to nearest seed)")
axes[1][0].imshow(coords[0], cmap=plt.cm.gray)
axes[1][0].set_title("y coordinate\nof neareset seed")
axes[1][1].imshow(coords[1], cmap=plt.cm.gray)
axes[1][1].set_title("x coordinate\nof neareset seed")
axes[1][2].imshow(rgb_labels)
axes[1][2].set_title("discrete Voronoi")
for ax in axes.ravel():
    ax.set_axis_off()
# overlay larger markers at the seed points for better visibility
for x, y in zip(xx, yy):
    # overlay in image
    axes[0, 0].plot(y, x, "w.")
    # overlay in rgb_labels
    axes[1, 2].plot(y, x, "w.")
plt.tight_layout()


"""
As a second demo, we apply the distance transform to a binary image of a
horse (and its inverse). The distance transform computes the Euclidean distance
from each foreground point to the nearest background point.
"""

horse = data.horse()
horse_inv = ~horse

distances = distance_transform_edt(
    cp.asarray(horse), return_distances=True, return_indices=False
)
distances_inv = distance_transform_edt(
    cp.asarray(horse_inv), return_distances=True, return_indices=False
)

distances = cp.asnumpy(distances)
distances_inv = cp.asnumpy(distances_inv)

fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes[0][0].imshow(horse_inv, cmap=plt.cm.gray)
axes[0][0].set_title("Foreground horse")
axes[0][1].imshow(horse, cmap=plt.cm.gray)
axes[0][1].set_title("Background horse")
axes[1][0].imshow(distances_inv)
axes[1][0].set_title("Distance\n(foreground horse)")
axes[1][1].imshow(distances)
axes[1][1].set_title("Distance\n(background horse)")
for ax in axes.ravel():
    ax.set_axis_off()
plt.tight_layout()
plt.show()
