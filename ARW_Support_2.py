import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import json
import threading
import time
import sys

accepted_mm_per_pixel = 0.05487

def open_bayer(file: str):
    '''
    Opens the bayer layer of a given ARW file.
    
    file should be a string filename ending in .ARW

    return: 2D numpy array of pixel values
    '''

    base_dir = os.getcwd()  # Base directory where program is located

    # Path to ARW Files folder
    arw_files_dir = os.path.join(base_dir, 'Images', 'ARW Files')

    # Initialize as False in case a file is never found
    found_file_path = False
    
    for root, dirs, files in os.walk(arw_files_dir):
        if file in files:
            found_file_path = os.path.join(root, file)
            break

    if found_file_path:
        with rawpy.imread(found_file_path) as raw:
            image = raw.raw_image.copy()

        return image

    else:
        print("File Not Found.")
        sys.exit()


def process(file: str, autosubtract=False):
    '''
    Does common file processing techniques used in PCT.

    return: 2D image array
    '''

    image = open_bayer(file)
    image = apply_median_filter(image)
    image = subtract_background(image, autosubtract=autosubtract)

    return image

def apply_median_filter(image, size=(3,3)):
    return median_filter(image, size)

def subtract_background(image, subtract=575, autosubtract=False):
    '''
    Takes down the value in all pixels in an image by some amount.

    image = A 2D array from which the background will be subtracted from

    subtract = Integer amount that will be subtracted from each pixel

    autosubtract: Will use a simple algorithm to find the subtraction value
        that yields the best curve fit, overriding the "subtract" parameter
    '''

    image = image.astype(np.int32)
    
    if autosubtract:
        best_subtraction = 0

        for exponent in range(3):
            corrs = []
            for i in range(10):
                subtraction = i * (10 ** (2 - exponent))
                image_adjusted = image - subtraction
                # Set lower bound of 0. Anything below 0 gets set to 0.
                image_adjusted = np.clip(image, 0 , None)
                gaussian = gaussian_2d(image_adjusted, 0)
                _, corr = gaussian_curve_fit(gaussian, corr=True)
                corrs.append(corr)
            best_subtraction += np.argmax(corrs) * (10 ** (2 - exponent))

        image -= best_subtraction
        image = np.clip(image, 0, None)
        return image

    else:
        image -= subtract
        # Set lower bound of 0. Anything below 0 gets set to 0
        image = np.clip(image, 0, None)
        return image

def convert(pixel, mm_per_pixel = None): # Units are in mm
    '''
    Converts pixel number into mm
    '''
    if mm_per_pixel is None:
        mm_per_pixel = accepted_mm_per_pixel
        
    return pixel * mm_per_pixel

def get_distances(pixels, mm_per_pixel=None): # Units are in mm
    '''
    Normally, when we plot, we have pixel values. The indices containing a given
        value are the pixel number. That isn't always useful.

    This function creates a list of distances from those pixel numbers. Your
        original list can act as the pixel values (y-axis on graphs), and this
        newly generated list can act as your distance (x-axis on graphs).

    pixels: List or 1D numpy array of pixel values

    return: List of distances where len(distances) = len(pixels)
        Units of distances are in mm
        Example:
            pixels = [400, 450, 473, 800, ...]
            distances = [0.051, 0.102, 0.153, 204, ...]
    '''
    if mm_per_pixel is None:
        mm_per_pixel = accepted_mm_per_pixel
        
    return np.arange(len(pixels)) * mm_per_pixel


def plot_3d(image, title='', figsize = (10,7), pixelspace = False,
            axisfontsize = 14, titlefontsize = 18, ticksize = 12,
            labelpad = 10):
    '''
    Creates a 3D plot of the provided image.

    image: A 2D numpy array
    
    title: Optional string that titles the graph
    
    figsize: Optional tuple of (width, height) in inches that
        determines the size of the created image
        
    pixelspace: True means x and y are pixel numbers.
                False means x and y are distances
                
    axisfontsize: Determines the font size of the names of the axes

    titlefontsize: Determines the font size of the title of the graph

    ticksize: Size of the numbers that appear on the axes

    labelpad: Determines spacing of the axis tick markers and the axis labels
    
    NO RETURN. It just plots.
    '''

    # Create meshgrid for the x and y coordinates
    if pixelspace:
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        X, Y = np.meshgrid(x, y)
    else:
        x = get_distances(image[0,:])
        y = get_distances(image[:,0])
        X, Y = np.meshgrid(x, y)
        

    # Set up the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d') # 111 is basically just (give me 1 plot). Long story...

    # Plot the surface
    surf = ax.plot_surface(X, Y, image, cmap='viridis', edgecolor='none')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    
    # Labels and title
    if pixelspace:
        ax.set_xlabel('X Pixel', fontsize=axisfontsize, labelpad=labelpad)
        ax.set_ylabel('Y Pixel', fontsize=axisfontsize, labelpad=labelpad)
    else:
        ax.set_xlabel('X Distance (mm)', fontsize=axisfontsize, labelpad=labelpad)
        ax.set_ylabel('Y Distance (mm)', fontsize=axisfontsize, labelpad=labelpad)

    ax.set_zlabel('Brightness (Arbitrary Units)', fontsize=axisfontsize, labelpad=labelpad)
    ax.set_title(title, fontsize=titlefontsize)

    # Adjust tick parameters
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.tick_params(axis='z', labelsize=ticksize)

    plt.show()


def full_sum(image, integration=True, include_error=False, gaussians=None):
    '''
    Sums all pixel values in a 2D array.

    integration: If true, fits gaussians to both x and y gaussians and
        integrates them for the "full sum". This also doubles as a 3D version
        of the integrate_gaussian method below.

    gaussians: Can provide a list of 2 1D gaussians that make up the full 2D
        gaussian and it will find the area under them rather than pulling
        them from a 2D image array.

    Might work with a list if integration is false?

    return: Area if not include_error
            Area, Area_error if include_error
    '''

    if not integration:
        return np.sum(image)

    elif integration:
        if gaussians is None:
            # Create gaussians
            xgauss = gaussian_2d(image, 0)
            ygauss = gaussian_2d(image, 1)
        else:
            xgauss = gaussians[0]
            ygauss = gaussians[1]

        # Get curve fit parameters
        xpopt, xpcov = gaussian_curve_fit(xgauss, include_errors=True)
        ypopt, ypcov = gaussian_curve_fit(ygauss, include_errors=True)

        # Extract parameters into reasonable variable names
        xamp, xmean, xsd = xpopt
        xamp_error, xmean_error, xsd_error = xpcov
        yamp, ymean, ysd = ypopt
        yamp_error, ymean_error, ysd_error = ypcov

        # NOTE!!! WE USE XAMP AND XAMP_ERROR FOR AMPLITUDE AND AMPLITUDE ERROR FROM HERE ON OUT
        # XAMP AND YAMP SHOULD BE ABOUT THE SAME ANYWAY
        
        # Calculate Area...it's really a volume but :shrug:
        Area = 2 * np.pi * xamp * xsd * ysd

        # Calculate 1sigma error in Area
        term1 = xsd * ysd * xamp_error
        term2 = xamp * ysd * xsd_error
        term3 = xamp * xsd * ysd_error

        Area_error = 2 * np.pi * np.sqrt(term1**2 + term2**2 + term3**2)

        if include_error:
            return Area, Area_error
        else:
            return Area
        

        
        

def plot_image(image):
    '''
    image should be a 2D numpy array

    Prints a picture of the image
    '''
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def gaussian_2d(image, axis=0):
    '''
    Creates a side-profile from a 2D image.
    Side profile is made by taking the silhouette (max values) as seen from 1 side

    image: A 2D numpy array

    axis: 0 or 1. Determines which direction we'll see the gaussian's
        side-profile from

    returns: 1D List of pixel values
    '''

    data = []
    if axis == 0:

        # For each row, find the brightest pixel to emulate a side-on view
        # Values are turned to float (from int64) to allow JSON serialization
        for i in range(len(image[:,0])):
            max_brightness = np.max(image[i,:])
            data.append(float(max_brightness))
        return data
    elif axis == 1:
        for i in range(len(image[0,:])):
            max_brightness = np.max(image[:,i])
            data.append(float(max_brightness))
        return data


def gaussian_func(x, amplitude, mean, std_dev):
    '''
    Allows the ability to quickly evaluate the value of a gaussian given
        its parameters and which x value you want it to be evaluated at.

    returns: Float value given by the gaussian formula
    '''
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)


   
def gaussian_curve_fit(gaussian, x_values=None, include_errors = False,
                       shift = None, pcov_list = True, Debug=False,
                       corr = False, minSD = 1, pixelspace = False,
                       mm_per_pixel = None):
    '''
    Provides the parameters of the gaussian function that best fits the data.
    You may notice that a good chunk of this is copy/pasted in the above function
    on finding the peak. It is. There wasn't an attractive way to keep the code
    to a single usage. I could use this function to call the above and allow
    the above function to return popt and pcov, but that's weird to have in a
    peak-finding algorithm. So I'll just leave it as is.

    gaussian: A 1D list of pixel values to which the gaussian will be fit

    x_values: Optional ability to give the x_values correlating to the gaussian you'll fit to

    include_errors: Whether or not the function will also return the covariance
        matrix. np.sqrt(np.diag(pcov)) gives the standard deviation of the
        fitted values.

    pcov_list: Returns the sqrt of the diagonals of pcov rather than the full
        matrix. These values are the 1sigma errors of [Amplitude, Mean, SD].
        If False, you get a 3x3 matrix of covariance values.

    corr: Whether or not the function will also return the correlation
        coefficient.
        0 = The fit doesn't fit the data at all.
        1 = The fit perfectly describes the data.

    minSD: The smallest standard deviation allowed. Any SDs below this
        will be marked as noise and have those pixels set to 0. The default
        value for this is in mm. Must be changed if working in pixel space.
        The equivalent for pixel space is about 19 pixels.

    pixelspace: If True, x-axis will be in pixels, not mm

    popt: A list of fitted parameters [Amplitude, Mean, SD]

    pcov: A 3x3 array of covariance values. np.sqrt(np.diag(pcov)) gives SD
        of [Amplitude, Mean, and SD] errors. np.diag(pcov) gives variances.

    R2: Correlation value between 0 and 1
    '''

    if Debug:
        print("\n\n\nSTARTING CURVE FIT FUNCTION:\n\n")
    
    # If a different conversion factor is not give, use the default
    if mm_per_pixel is None:
        mm_per_pixel = accepted_mm_per_pixel
        
    if Debug:
        print("mm_per_pixel: ", mm_per_pixel, '\n')

    # Convert things to pixel space if needed.
    if pixelspace and x_values is None:
        x_values = np.arange(len(gaussian))
        mean_guess = np.mean(gaussian)    # Initial guess for scipy fit function
        SD_guess = np.std(gaussian) / mm_per_pixel
    elif not pixelspace and x_values is None:
        x_values = get_distances(gaussian)
        mean_guess = convert(np.mean(gaussian))
        SD_guess = convert(np.std(gaussian))
    elif pixelspace and x_values is not None:
        x_values = x_values
        mean_guess = np.mean(gaussian)
        SD_guess = np.std(gaussian) / mm_per_pixel
    else:
        x_values = x_values
        mean_guess = convert(np.argmax(gaussian))
        SD_guess = convert(np.std(gaussian))

    if Debug: 
        print("x_values: ", x_values[:5])
        print("mean_guess: ", mean_guess)
        print("SD_guess: ", SD_guess)

    
    # Get initial fit. p0 is guessed values to help the fit function
        # converge quicker
    bounds = ([0, -50, 0], [10000, 200, 50])
    
    popt, pcov = curve_fit(gaussian_func, x_values, gaussian,
                           p0=[max(gaussian), mean_guess, SD_guess],
                           bounds=bounds)

    if Debug:
        print("\nCurve Fit Popt: ", popt)


    # Decide what values to return
    if not include_errors and not corr:
        return popt
    elif include_errors and not corr:
        if pcov_list:
            pcov = np.sqrt(np.diag(pcov))
        return popt, pcov
    elif corr:
        SSR = sum((gaussian_values - gaussian)**2)
        SST = sum((gaussian - np.mean(gaussian))**2)
        R2 = 1 - (SSR/SST)

        if not include_errors:
            return popt, R2
        else:
            if pcov_list:
                pcov = np.sqrt(np.diag(pcov))
            return popt, pcov, R2


def integrate_gaussian(popt, pcov = None, include_error = False,
                       pixelspace = False, start_limit = 0, end_limit = 220):
    '''
    Provides analytical solution to the integral of a gaussian.

    popt: List of gaussian parameters [Amplitude, Mean, SD]

    pcov: List of gaussian parameter 1sigma errors [Amplitude, Mean, SD]

    include_error: Determines whether or not the 1sigma error of the integral
        should be included. Requires pcov if True.

    pixelspace: x-axis is in pixel number if True, or in mm if False.

    limits: Limits of integration (they default to the size of the image in mm)
    '''

    # Throw an error if user wants errors but doesn't give necessary values
    if include_error and pcov is None:
        print("ERROR: pcov was not given as an argument to the integrate_gaussian function.")
        sys.exit()

    # Convert limit of integration to pixel space if user chooses pixel space
    # Make an exception for user giving a custom end_limit input
    if pixelspace and end_limit == 220:
        end_limit=4000

    # Extract values from popt
    Amplitude = popt[0]
    Mean = popt[1]
    SD = popt[2]

    # If pcov is given, assign the values
    if pcov is not None:
        Amplitude_error = pcov[0]
        Mean_error = pcov[1]
        SD_error = pcov[2]

    Area = Amplitude * np.sqrt(2 * np.pi) * SD

    if include_error:
        # d(Area) = sqrt(2pi * (sigma^2(dA)^2 + A^2(dsigma)^2))
        first_term = SD**2 * Amplitude_error**2
        second_term = Amplitude**2 * SD_error**2

        Area_error = np.sqrt(2*np.pi * (first_term + second_term))

        return Area, Area_error

    else:
        return Area

  
    
def plot_gaussian(gaussian, distances = None, popt = None, shift = None,
                  title = '', show = True, save = False,
                  file_name = '', fit = False, pixelspace = False,
                  mm_per_pixel = None, fontsize = 14, Debug=False,
                  ticksize = 12, titlesize=20, pointsize=12, multiple=False,
                  graphs=None, xleft=None, xright=None):
    '''
    Creates a 2D plot of an expected gaussian.

    gaussian: A List of pixel values that, when plotted consecutively, should form a gaussian

    distances: Optional list to use as the x-axis. If not given, it will find them for you.

    popt: List of 3 fitting parameters. If not given and fit=True, it will calculate it

    shift: How far to shift the peak of the gaussian to align it with 0.
        If not given, it will calculate it.

    show: Shows the graph to the user if True. Turn to false to automate the saving of images

    save: Saves the graph under file_name (another input parameter) if True.

    file_name: Name for the file to be saved under. Only relevant if save = True

    fit: Graphs a best-fit gaussian curve over the data if True

    pixelspace: x-axis of the graph is in pixels if True. In mm if False

    mm_per_pixel: Allows to change the conversion factor between pixel number and distance

    fontsize: Size of the font used in the axes labels

    Debug: Decides whether or not to run print statements and any other
        relevant data used for debugging.

    ticksize: Size of the font used for each tick in the graph's grid

    pointsize: Size of points in scatterplot of data

    xleft: Crops the image to start at this x-value (recommended value is -10)

    xright: Crops the image to end at this x-value (recommended value is 10)

    test: Can be given integer values to test the code up to certain points.
        Used for troubleshooting hard-to-find bugs.

    PLEASE READ:
        multiple: If True, pass in a list of popt's for the "gaussian" argument.
            It will plot the popt's on top of each other.

        graphs: a list of names correlating to each of the popt's provided

        shift: This is expected to be a single value. It is the amount that each value in
            distance should be shifted left in order to center distances at 0. You can
            usually pull this from the database:
                distances = Database[folder][image]['Subtracted']['X Gaussian Distances']
                shift = Database[folder][image]['Subtracted']['X Gaussian Shift']
            The exact image you choose from the set of multiple images isn't relevant.
            Therefore, an alternative could be a list of any distance values where
                the distance for each entry increments by {mm_per_pixel}
    
    No return value. This function simply plots the graph.
    '''

    if Debug:
        print("\n\n\nSTARTING PLOTTING FUNCTION\n\n")
        
    size = [pointsize] * len(gaussian) # Matplotlib takes a list of point sizes for each point
    
    fig, ax = plt.subplots()

    if not multiple:
        if distances is None:
            x_vals = np.arange(len(gaussian)) if pixelspace else get_distances(np.arange(len(gaussian)), mm_per_pixel)
        else:
            x_vals = np.array(distances) if not pixelspace else np.arange(len(gaussian))

        if Debug:
            print("x_vals: ", x_vals[:5])
            
        # Find popt and/or shift for data
        shift_popt = gaussian_curve_fit(gaussian, x_values=x_vals, pixelspace=pixelspace, mm_per_pixel=mm_per_pixel)

        if Debug and shift is None:
            print("\nShift Popt: ", shift_popt)

        # Find appropriate shift
        if shift is None:
            shift = shift_popt[1]
        else:
            shift = shift

        if Debug:
            print("\nShift being used: ", shift)
        
        shifted_x_vals = x_vals - shift

        
        if fit or popt is not None:
            if popt is None or pixelspace:
                popt = shift_popt
            else:
                popt = popt

            if Debug:
                print("\nPopt used in plot: ", popt)
            
            gaussian_vals = gaussian_func(shifted_x_vals, popt[0], 0, popt[2])

            # Plot data and fit
            plt.plot(shifted_x_vals, gaussian_vals, label='Gaussian Fit', color='blue')
            plt.legend()
            plt.scatter(shifted_x_vals, gaussian, color='red', s=size)

        else:
            plt.scatter(shifted_x_vals, gaussian, color='red', s=size)


    elif multiple and distances is None:
        print("To call 'multiple', you need to provide a single distances array to calculate values at")
        sys.exit()

    else:
        x_vals = np.array(distances) if not pixelspace else np.arange(len(distances))

        if not isinstance(shift, int) and not isinstance(shift, float):
            print("\n\n\n")
            print("ERROR! PLEASE READ THE MESSAGE BELOW\n\n")
            print("Shift given was not of type int or float.")
            print("Shift given is of data type ", type(shift))
            print("Please read the docstring below for additional help.")
            print("\n\n", plot_gaussian.__doc__)
            sys.exit()
            
        x_vals = x_vals - shift
        
        # Center at 0
        for i, popt in enumerate(gaussian):
            popt[1] = 0
            #print("Popt: ", popt, '\n')

            gaussian_vals = gaussian_func(x_vals, popt[0], popt[1], popt[2])

            ax.plot(x_vals, gaussian_vals, label=graphs[i])

        plt.legend()

            

    if not pixelspace:
        ax.set_xlabel('Distance (mm)', fontsize=fontsize)
    elif pixelspace:
        ax.set_xlabel('Pixel Number', fontsize=fontsize)
    ax.set_ylabel('Brightness', fontsize=fontsize)


    # For cropping the x-axis manually
    if xright is None and xleft is None:
        pass
    elif xright is not None and xleft is None:
        ax.set_xlim(right=xright)
    elif xright is None and xleft is not None:
        ax.set_xlim(left = xleft)
    else:
        ax.set_xlim(xleft, xright)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax.set_title(title, fontsize=titlesize)


    if show:
        plt.show()

    if save:
        plt.savefig(file_name, dpi=800)
        plt.close()
        
    

def plot_run_sums(folder_name, title='', autosubtract=True,
                  fontsize=14, ticksize=12):
    # Expects Images folder -> ARW Files folder -> folder_name folder
    folder_path = './Images/ARW Files/'+ folder_name + '/'

    # This says that we'll choose all ARW files
    pattern = '*.ARW'

    # This will store how many files there for plotting purposes
    num_files = 0

    # This will store the sums for each file
    full_sums = []

    print(folder_path)
    # For each ARW file in folder...
    for file_path in glob.glob(folder_path + pattern):
        # Open the raw data
        with rawpy.imread(file_path) as raw:
            image = raw.raw_image.copy()

        # Apply median filter
        image = apply_median_filter(image)
        image = subtract_background(image, autosubtract=autosubtract)

        # Increment num_files
        num_files += 1
        
        # Get the sum
        image_sum = full_sum(image)

        # Append sum to list of sums
        full_sums.append(image_sum)

    # Plot the sums
    fig, ax = plt.subplots()
    ax.scatter(range(num_files), full_sums)
    ax.set_title(title)
    ax.xticks(fontsize=ticksize)
    ax.yticks(fontsize=ticksize)
    ax.set_xlabel('"File Number"', fontsize=fontsize)
    ax.set_ylabel("Full Sum", fontsize=fontsize)
    plt.show()


def get_all_images(folder_name, include_names=False):
    '''
    Creates a list of all 2D image arrays for every ARW file in a given folder

    include_names: Decides whether or not to include the image name (str) in the results

    return: List of 2D image arrays if include_names == False
            List of (name, image) if include_names == True
    '''
    # Expects Images folder -> ARW Files folder -> folder_name folder
    folder_path = './Images/ARW Files/'+ folder_name + '/'

    # This says that we'll choose all ARW files
    pattern = '*.ARW'

    # This will store the array for each file
    images = []

    print(folder_path)
    # For each ARW file in folder...
    for file_path in glob.glob(folder_path + pattern):
        # Open the raw data
        with rawpy.imread(file_path) as raw:
            image = raw.raw_image.copy()

        if not include_names:
            images.append(image)

        elif include_names:
            images.append([file_path[-11:], image])
    
    return images


def get_useful_images(folder_name, include_names=False):
    '''
    Pulls all images from a folder that meet a certain brightness (full_sum) threshold

    return: List of tuples
        if include_names == False: items are of shape (image, image_sum)
        if include_names == True: items are of shape (image, image_sum, file_name)
    '''

    if not include_names:
        images = get_all_images(folder_name)
        image_sums = [(image, full_sum(image, integration=False)) for image in images]

    if include_names:
        images = get_all_images(folder_name, include_names=True)
        image_sums = [(image[1], full_sum(image[1], integration=False), image[0]) for image in images]

    # Sort by sum
    image_sums.sort(key=lambda x: x[1])

    lowest_sums = [image_sum[1] for image_sum in image_sums[:5]]
    highest_sums = [image_sum[1] for image_sum in image_sums[-5:]]


    lowest_avg = sum(lowest_sums) / len(lowest_sums)
    highest_avg = sum(highest_sums) / len(highest_sums)

    cutoff_line = (highest_avg - lowest_avg) / 2

    filtered_image_sums = [tup for tup in image_sums if tup[1] >= (lowest_avg + cutoff_line)]

    return filtered_image_sums

def compare_with_geant_2d(gaussian, amplitude, std_dev, title='',
                          mm_per_pixel = None, fontsize=14):
    '''
    Graphs data with an overlayed gaussian whose parameters are given by Geant.

    gaussian: A List of pixel values

    amplitude: Amplitude given by Geant

    std_dev: Standard Deviation given by Geant

    title: Title for the whole graph.

    No return value. Just graphs.
    '''

    if mm_per_pixel is None:
        mm_per_pixel = accepted_mm_per_pixel
    
    # Get the fit for the data
    data_gaussian_popt = gaussian_curve_fit(gaussian)

    # Extract the parameters needed for comparison
    data_amp = data_gaussian_popt[0]
    data_std_dev = data_gaussian_popt[2]

    # Create a list of x-values
    x_values = get_distances(np.arange(len(gaussian)), mm_per_pixel)

    # Shift values so that we can center the gaussians at 0
    shift = peak_finder(gaussian, mm_per_pixel=mm_per_pixel)
    x_values -= shift

    # Get y-values for each gaussian
    data_y = []
    geant_y = []

    for x in x_values:
        data_y.append(gaussian_func(x, data_amp, 0, data_std_dev))
        geant_y.append(gaussian_func(x, amplitude, 0, std_dev))

    fig, ax = plt.subplots()
    ax.plot(x_values, data_y, label='Data')
    ax.plot(x_values, geant_y, label='Geant')
    ax.set_xlabel("Distance (mm)", fontsize=fontsize)
    ax.set_ylabel("Intensity", fontsize=fontsize)
    ax.set_title(title)
    ax.legend()
    plt.show()

