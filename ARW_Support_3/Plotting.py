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
    if xstart is not None or xend is not None or ystart is not None or yend is not None:
        Image.crop_image(xstart, xend, ystart, yend)

    plt.show()



def plot_gaussian(Image,
                  axis='x' -> str, # Which axis will be plotted
                  title='' -> str,
                  show='True' -> bool, # Whether to display plot or not
                  save='False' -> bool, # Whether to save plot or not
                  file_name='' -> str, # If saving, what the file name will be
                  fit='False' -> bool, # Whether or not to curve fit
                  pixelspace=None -> bool,
                  fontsize=14 -> float,
                  ticksize=12 -> float,
                  titlesize=20 -> float,
                  pointsize=12 -> float,
                  center=True -> bool, # Center Gaussian at 0?
                  xleft=None -> float, # Crop left side of plot
                  xright=None -> float, # Crop right side of plot
                  ylabel='Brightness' -> str):
    '''
    Creates 2D plot of your Gaussian.

    Parameters:
        Image: Image object defined in ARW3 package.
        title: Title displayed above graph
        show: If set False, it will not show the graph on screen (used for saving en-mass)
        save: If True, Python will automatically save the image once it's created.
        file_name: Name of file (only used if saving)
        fit: If True, it will curve fit a Gaussian to the data
        pixelspace: Allows user to specify pixelspace (True) or distance space (False)
        fontsize: Font of x and y labels
        ticksize: Font size of numbers on the axes
        titlesize: Font size of title
        center: If False, Gaussian is plotted as is without self-centering.
        xleft: Cuts graph from -inf to this value
        xright: Cuts graph from this value to +inf
        ylabel: Changes label on y-axis

    returns:
        Nothing. Just plots
    '''

    # Assign gaussian variable to either the x or y Gaussians
    if axis.lower() == 'x':
        if Image.x_Gaussian is None:
            Image.gaussian_2d(axis=axis)
        gaussian = Image.x_Gaussian
        distances = Image.X[0,:]
    elif axis.lower() == 'y':
        if Image.y_Gaussian is None:
            Image.gaussian_2d(axis=axis)
        gaussian = Image.y_Gaussian
        distances = Image.Y[:,0]
    else:
        print("\n\n\nERROR IN PLOTTING GAUSSIAN.")
        print(f"Parameter 'axis' was given a value of {axis} when 'x' or 'y' was expected.")
        print("Continuing with plot using x-axis.\n\n\n")
        axis = 'x'
        gaussian = Image.x_Gaussian
        distances = Image.X[0,:]


    # Matplotlib takes a list of point sizes for each point in the graph
    size = [pointsize] * len(gaussian)

    fig, ax = plt.subplots()


    # Center if needed
    if center:
        x_cen, y_cen = Image.find_center()

        if axis.lower() == 'x':
            distances -= x_cen
        else:
            distances -= y_cen


    if fit:
        
