import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pickle as pkl
import os


def _imshow_pytorch(rgb, ax=None):
    from utils.transforms import im_to_numpy

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(im_to_numpy(rgb * 255).astype(np.uint8))
    ax.axis("off")


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf
