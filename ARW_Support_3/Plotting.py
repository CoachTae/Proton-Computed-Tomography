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
    
def plot_2d(Image):
    '''
    Image should be a Image object

    Prints a picture of the image
    '''
    image = Image.image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def gaussian_func(x, amplitude, mean, std_dev) -> float:
    '''
    Evaluates value of a Gaussian at x given its parameters.

    Parameters:
        x: Location to evaluate Gaussian (int, float, or np array)
        amplitude: Amplitude of Gaussian
        mean: Mean of Gaussian
        std_dev: Standard deviation of Gaussian

    returns:
        Value of Gaussian at x (float)
    '''
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)

def plot_gaussian(Image,
                  axis=0,
                  fit = False, 
                  fontsize = 14, ticksize = 12, titlesize=20, pointsize=12, 
                  ylabel='Brightness',
                  xleft=None, xright=None,
                  title = None,
                  show = False,
                  save = False,
                  file_name = ''):
    '''
    Plots 2D gaussian fit for the specified axis.  
    
    Parameters
    ----------
    axis : str
        0 or 1 — which Gaussian profile to fit. 0 is 'x' and 1 is 'y'
    fit: bool 
        Graphs a best-fit gaussian curve over the data if True
    include_errors : bool
        If True, returns covariance information.
    pcov_list : bool
        If True, return sqrt(diagonal) of covariance matrix (1σ errors).
    corr : bool
        If True, compute R² correlation coefficient.
    minSD : float
        Minimum allowed standard deviation.
    fontsize, ticksize, titlesize, pointsize : int (s)
        Plot elements
    ylabel : str
        Plot element
    xleft : int
        Crops the image to start at this x-value (recommended value is -10)
    xright : int 
        Crops the image to end at this x-value (recommended value is 10)
    shift  : float
        Optional manual shift of x-values before fitting.

    show : bool
        Option to show plot
    save : bool
        Option to save plot. Defaults to False 
    file_name : str 
        Name for the file to be saved under. Only relevant if save = True
    Returns
    -------
    None.

    '''
    # Check which axis we are plotting
    if axis not in [0, 1]:
        print('Invalid axis entry')
        return
    
    if axis==0:
        # Checking if x_Gaussian is populated
        if Image.x_Gaussian == 0: 
            Image.gaussian_2d(axis = 0)
        # If fit, then checking if the fit params have been calculated
        if fit and Image.x_fit_params is None:
            Image.gaussian_curve_fit(axis=axis)
        fitted = Image.x_fit_params
        vertical = Image.x_Gaussian
        horizontal = Image.X[0,:]
    else:
        # Checking if y_Gaussian is populated
        if Image.y_Gaussian == 0: 
            Image.gaussian_2d(axis = 0)
        # If fit, then checking if the fit params have been calculated
        if fit and Image.y_fit_params is None:
            Image.gaussian_curve_fit(axis=axis)
        fitted = Image.y_fit_params
        vertical = Image.y_Gaussian
        horizontal = Image.Y[:,0]
    
    # Matplotlib takes a list of point sizes for each point
    size = pointsize
    
    # Initalizing plot
    fig, ax = plt.subplots()
    
    # Plot data and fit
    if fit:
        gaussian_vals = gaussian_func(horizontal, fitted[0] , fitted[1], fitted[2])
        plt.plot(horizontal, gaussian_vals, label='Gaussian Fit', color='blue')
        plt.legend()
    plt.scatter(horizontal, vertical, color='red', s=size)    
    
    # Handling the x axis in terms of pixles or mm
    if not Image.is_pixelspace:
        ax.set_xlabel('Distance (mm)', fontsize=fontsize)
    else:
        ax.set_xlabel('Pixel Number', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    # For cropping the x-axis manually
    if xright is None and xleft is None:
        pass
    elif xright is not None and xleft is None:
        ax.set_xlim(right=xright)
    elif xright is None and xleft is not None:
        ax.set_xlim(left = xleft)
    else:
        ax.set_xlim(xleft, xright)
    # Plot elements
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax.set_title(title, fontsize=titlesize)
    
    if show:
        plt.show()

    if save:
        plt.savefig(file_name, dpi=800)
        plt.close()
    