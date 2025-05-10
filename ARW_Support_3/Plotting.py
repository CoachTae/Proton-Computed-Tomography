import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d(Image, title='',
            figsize = (10,7),
            axisfontsize = 14,
            titlefontsize = 18,
            ticksize = 12,
            labelpad = 10,
            xstart = None,
            xend = None,
            ystart = None,
            yend = None):

    '''
    Plots image as a 3D surface using its spatial map.

    Parameters:
        title: Title put above the graph
        figsize: Tuple defining figure size in inches
        axisfontsize: Font size for axis labels
        titlefontsize: Font size for title
        ticksize: Font size for tick marks
        labelpad: Padding between axis and label
        xstart, xend, ystart, yend: Optional cropping window for axes
    '''


    if Image.image is None:
        print("No image data found. Cannot plot.")
        return

    # Ensure spatial map exists
    if Image.X is None or Image.Y is None:
        Image.get_spatial_map()

    X, Y, Z = Image.X, Image.Y, Image.image

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labels and title
    if Image.is_pixelspace:
        ax.set_xlabel('X Pixel', fontsize=axisfontsize, labelpad=labelpad)
        ax.set_ylabel('Y Pixel', fontsize=axisfontsize, labelpad=labelpad)
    else:
        ax.set_xlabel('X Distance (mm)', fontsize=axisfontsize, labelpad=labelpad)
        ax.set_ylabel('Y Distance (mm)', fontsize=axisfontsize, labelpad=labelpad)

    ax.set_zlabel('Brightness (Arbitrary Units)', fontsize=axisfontsize, labelpad=labelpad)
    ax.set_title(title, fontsize=titlefontsize)

    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.tick_params(axis='z', labelsize=ticksize)

    # Axis limits
    if xstart is not None:
        ax.set_xlim(left=xstart)
    if xend is not None:
        ax.set_xlim(right=xend)
    if ystart is not None:
        ax.set_ylim(bottom=ystart)
    if yend is not None:
        ax.set_ylim(top=yend)

    plt.show()
