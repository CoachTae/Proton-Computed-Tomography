import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.integrate import quad

# open_file(run) opens a file and returns an (m, n, 3) matrix for the image
    # run is expected to be an integer
    
# open_bayer(run, file=None) opens a file and returns an (m, n) matrix matrix of the bayer layer
    # run is expected to be an integer
    # If file is given, it's expected as a string
    # If file is given, run can take on any value as it's not used
        # run is only mandatory to keep old code functioning without edits
    
# bin_image(image, bin_size) takes in an image and bins it by (bin_size) pixels
    # image is expected to be either of shape (m, n, 3) or (m, n)
    # bin_size should be an integer
    
# pull_dimensions(image) takes an image and returns the height and width
    # image is expected to be either of shape (m, n, 3) or (m, n)

# hotspot(image, bin_size, brightness_limit)
    # image is expected to be of shape (m, n, 3)
    # bin_size determines how big each bin of pixels will be when finding hotspot
    # brightness_limit is an optional parameter, default at 300 (it was made for RGB images originally)
        # Determines how bright a pixel must be to be considered part of the hotspot
    # hopefully finds the hotspot of an image
    # returns start_row, end_row, start_column, end_column
        # Returned values are in original image coordinates

# plot_rgb(image, title) returns nothing. Just does a brightness plot.
    # Expects an (m, n, 3) matrix for "image"
    
# plot_3d(image, title) returns nothing. Does brightness plot
    # Expects an (m, n) matrix for "image"
    
# full_sum(image) takes an image and sums ALL rgb values. Returns total_sum
    # image is any array
    
# plot_image(image) takes any matrix (2 or 3 dimensions) and prints the image

# gaussian_2d(image, axis) takes an (m, n) array and creates a list of brightness values
    # Axis = 0 means our list is m entries of brightness levels
    # Axis = 1 means our list is n entries of brightness levels
    # For each entry, it takes that row/column and grabs the highest brightness level

# gaussian_func(x, amplitude, mean, std_dev) returns the value of a gaussian at x whose parameters are amplitude, mean, std_dev

# gaussian_curve_fit(gaussian, include_errors=False) returns parameters of best fit to the data given
    # parameter "gaussian" is expected to be a 1D list of brightness values
    # if include_errors is False, we return popt
        # popt is [amplitude, mean, std_dev]
    # if include_errors is True, we return popt, pcov
        # pcov is a matrix of errors
        # np.diag(pcov) gives variance of each parameter's errors

# integrate_gaussian(popt, start_limit=0, end_limit=4000) returns the area under a gaussian
    # popt is a list of gaussian parameters (usually found with the returned values from gaussian_curve_fit
    # the "limit" parameters are your limits of integration and are optional

# plot_gaussian(gaussian, title='', show=True, save=False, file_name='', fit=False) does a 2d plot of a gaussian
    # gaussian is expected to be a 1D list of brightness values (1 value per pixel)
    # title should be a string to title the graph
    # show being True display the image on the screen
        # set show to False if the goal is to just save the image without viewing it
    # save being True saves the graph as is and closes it
    # if show and save are both True, save/close should happen after you close the displayed graph
    # file_name should be a string in which the saved image will be named
    # fit being set to True will plot a line to fit over the gaussian

# find_gaussian_peak(gaussian, min_bright=0) finds the slice of gaussian in which which the peak exists
    # gaussian is expected to be a 1D list of brightness values (like the output of gaussian_2d function)
    # min_bright is the brightness in which we consider it "out of the gaussian"
    # min_bright is optional and is 0 if nothing else is given
        # if 0 isn't working (other pixels are slightly above 0, try 0.0001 of max brightness observed)

# apply_median_filter(image) takes in a 2D image and returns the same image with a median filter applied
    # image is an (m, n) array

# subtract_background(image) takes all pixel values down by 535
    # returns an image of the same dimensions as the input
    # any pixel that would fall below 0 is instead set to 0

# plot_run_sums(folder_name) graphs uncropped sums of all ARW files in a folder
    # folder_name is a string located in ./Images/ARW Files/

# get_all_images(folder_name) returns a list of image arrays for every file in a folder
    # folder_name is a string located in ./Images/ARW Files/

# get_useful_images(folder_name) filters out runs with low sums (beam off)
    # returned list of tuples of shape (image, sum)
        # So each object in list is an array of the image and the corresponding sum
        
# get_3d_sum(image) gets a 3D gaussian fit + sum for an image
    # returns the sum for the 3D gaussian
    # DOES NOT APPLY MEDIAN FILTER OR SHIFT DOWN BY 535!!

def open_file(run): #Opens the postprocessed image
    # Base directory where program is located
    base_dir = os.getcwd()

    # Path to ARW Files folder
    arw_files_dir = os.path.join(base_dir, 'Images', 'ARW Files')
    
    # Name of file to be examined
    file_name = 'DSC00' + str(run) + '.ARW'

    # Construct full path to file
    file_path = os.path.join(arw_files_dir, file_name)
    
    with rawpy.imread(file_path) as raw:
        image = raw.postprocess()

    # Pull variables for the height and width of the image
    height, width, _ = image.shape

    return image

def open_bayer(run, file=None): # Opens the bayer layer of a given file
    # Base directory where program is located
    base_dir = os.getcwd()

    # Path to ARW Files folder
    arw_files_dir = os.path.join(base_dir, 'Images', 'ARW Files')
        
    if not file:
        # Name of file to be examined
        file_name = 'DSC00' + str(run) + '.ARW'

        # Construct full path to file
        file_path = os.path.join(arw_files_dir, file_name)
        
        with rawpy.imread(file_path) as raw:
            image = raw.raw_image.copy()

        return image

    if file:
        # Variable to hold path of found file
        found_file_path = None

        # Walk through ARW Files directory and subdirectories
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
            return None

def bin_image(image, bin_size):
    try:
        # Pull variables for the height and width of the image
        height, width, _ = image.shape

        # This is the size of the image after binning
        binned_height = height // bin_size
        binned_width = width // bin_size

        # Create a blank image that is to be filled in
        binned_image = np.zeros((binned_height, binned_width, 3), dtype = np.uint8)

        # Fill in the binned_image with associated rgb values
        for i in range(binned_height):
            for j in range(binned_width):
                # Extract the nxn block
                block = image[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size, :]
                
                # Calculate the average RGB values of the block
                average_color = np.mean(block, axis=(0,1)).astype(np.uint8)
                
                # Assign the average color to the corresponding pixel in the binned image
                binned_image[i, j, :] = average_color
                
        return binned_image
    except:
        # Pull variables for the height and width of the image
        height, width = image.shape

        # This is the size of the image after binning
        binned_height = height // bin_size
        binned_width = width // bin_size

        # Create a blank image that is to be filled in
        binned_image = np.zeros((binned_height, binned_width))

        # Fill in the binned_image with associated brightness values
        for i in range(binned_height):
            for j in range(binned_width):
                # Extract the nxn block
                block = image[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]
                
                # Calculate the average RGB values of the block
                average_color = np.mean(block)
                
                # Assign the average color to the corresponding pixel in the binned image
                binned_image[i, j] = average_color
                
        return binned_image

def pull_dimensions(image):
    try:
        height, width, _ = image.shape

    except:
        height, width = image.shape

    return height, width

def hotspot(image, bin_size, brightness_limit=300):

    # Bin the image
    binned_image = bin_image(image, bin_size)

    # Get binned dimensions
    binned_height, binned_width = pull_dimensions(binned_image)
    
    # These 4 lines will determine the box that the hot spot lies in
    # Start by having them encapsulate the whole image
    start_row = 0
    end_row = binned_width
    start_column = 0
    end_column = binned_height

    # If it's an RGB array, we run the "try" block.
    try:
        # For each row, search that row for the first brightness peak
        for i in range(binned_height):

            # When the first brightness peak is found, mark the index for that row
            if np.max(np.sum(binned_image[i, :, :], axis=-1)) > brightness_limit and start_row == 0:
                start_row = i

            # After the first peak is found, find the first row to NOT peak >= brightness_limit
            # We assume there are no more rows after who can peak >= brightness_limit
            if start_row != 0 and end_row == binned_width and np.max(np.sum(binned_image[i, :, :], axis=-1)) < brightness_limit:
                end_row = i
                break

        # Repeat the process above but for columns instead
        for i in range(binned_width):
            if np.max(np.sum(binned_image[:, i, :], axis=-1)) > brightness_limit and start_column == 0:
                start_column = i

            if start_column != 0 and end_column == binned_height and np.max(np.sum(binned_image[:, i, :], axis=-1)) < brightness_limit:
                end_column = i
                break

    # If array is NOT an RGB array (probably a bayer layer)
    except:
        # For each row, search that row for the first brightness peak
        for i in range(binned_height):

            # When the first brightness peak is found, mark the index for that row
            if np.max(binned_image[i, :]) > brightness_limit and start_row == 0:
                start_row = i

            # After the first peak is found, find the first row to NOT peak >= brightness_limit
            # We assume there are no more rows after who can peak >= brightness_limit
            if start_row != 0 and end_row == binned_width and np.max(binned_image[i, :]) < brightness_limit:
                end_row = i
                break

        # Repeat the process above but for columns instead
        for i in range(binned_width):
            if np.max(binned_image[:, i]) > brightness_limit and start_column == 0:
                start_column = i

            if start_column != 0 and end_column == binned_height and np.max(binned_image[:, i]) < brightness_limit:
                end_column = i
                break

    start_row = start_row*bin_size
    end_row = end_row*bin_size
    start_column = start_column*bin_size
    end_column = end_column*bin_size
    
    return start_row, end_row, start_column, end_column

def plot_rgb(image, title):
    # Calculate the RGB sums for each pixel
    brightness_matrix = np.sum(image, axis=2)

    # Create meshgrid for the x and y coords
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)

    # Set up the plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, brightness_matrix, cmap='viridis', edgecolor='none')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labels and title
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_zlabel('Brightness')
    ax.set_title(title)

    # Show plot
    plt.show()

def plot_3d(image, title):
    # Create meshgrid for the x and y coords
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)

    # Set up the plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, image, cmap='viridis', edgecolor='none')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labels and title
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_zlabel('Brightness')
    ax.set_title(title)

    # Show plot
    plt.show()

def full_sum(image):
    total_sum = np.sum(image)

    return total_sum

def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def gaussian_2d(image, axis):
    data = []
    if axis == 0:
        for i in range(len(image[:,0])):
            max_brightness = np.max(image[i,:])
            data.append(max_brightness)
        return data
    elif axis == 1:
        for i in range(len(image[0,:])):
            max_brightness = np.max(image[:,i])
            data.append(max_brightness)
        return data

def gaussian_func(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) / std_dev)**2 / 2)


# This variable needs to be used in the plotting function, but is defined in the curve_fit function
# I did not want to edit my return values to include this because it might be too big of a change given the current infrastructure of the code
offset = None
def gaussian_curve_fit(gaussian, include_errors=False, corr=False):
        # The window created below "offset" creates a new x-axis from 0 to 470
        # This messes with our centering of the gaussian and the fit at 0
        # We need to compensate by adding in the offset that we lost
        global offset
        offset = np.argmax(gaussian) - 235

        # We will do our fit on this window to keep our fit within the scintillator
        # This fits the gaussian between x = 0 and x = 470 (pixel units)
        # Data is from x = 0 to x = 4000 (pixel units)
        gaussian = gaussian[np.argmax(gaussian)-235:np.argmax(gaussian)+235]

        x_values = np.arange(len(gaussian)) * 0.051
        popt, pcov = curve_fit(gaussian_func, x_values, gaussian, p0=[max(gaussian), np.argmax(gaussian) * 0.051, 10])
        
        # This loop looks for near-delta functions caused by noise that was not filtered out
        # It sets these noise values to 0 and refits another gaussian until we get a satisfactory one
        while True:
            # If SD is less than 1 (arbitrary value that I eyeballed), consider it noise
            if popt[2] <= 1:
                if popt[2] < 0:
                    popt[2] = abs(popt[2])
                    
                max_value = np.argmax(gaussian)
                gaussian[max_value] = 0
                try:
                    popt, pcov = curve_fit(gaussian_func, x_values, gaussian, p0=[max(gaussian), np.argmax(gaussian) * 0.051, 10])
                    continue
                except:
                    continue
            break
                
        gaussian_values = gaussian_func(x_values, *popt)
        if not include_errors and not corr:
            return popt
        if include_errors and not corr:
            return popt, pcov

        if corr:
            variance = np.var(gaussian)
            
            mse = np.mean((gaussian - gaussian_func(x_values, *popt))**2)
            R2 = 1 - (mse/variance)

            if not include_errors:
                return popt, R2
            elif include_errors:
                return popt, pcov, R2
        

def integrate_gaussian(popt, start_limit=0, end_limit=204):
    amplitude, mean, stddev = popt
    area, _ = quad(gaussian_func, start_limit, end_limit, args=(amplitude, mean, stddev))
    return area
    

def plot_gaussian(gaussian, title='', show=True, save=False, file_name='', fit=False):
    fig, ax = plt.subplots()
    x_vals = np.arange(len(gaussian)) #* 0.051

    
    if fit:
        popt = gaussian_curve_fit(gaussian)
        x_values = np.arange(len(gaussian)) * 0.051

        # Shift x_values to center Gaussian at x=0
        shifted_x_values = x_values - popt[1] - offset*0.051
        gaussian_values = gaussian_func(shifted_x_values, popt[0], 0, popt[2])
        ax.plot(shifted_x_values, gaussian_values, label='Gaussian Fit', color='blue')
        plt.legend()
        ax.scatter(x_vals - np.argmax(gaussian)*0.051, gaussian, color='red')

    else:
        ax.scatter(x_vals, gaussian, color='red')
        
    ax.set_xlabel('Distance (mm)', fontsize=14)
    ax.set_ylabel('Brightness', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_title(title)

    if show == True:
        plt.show()

    if save == True:
        plt.savefig(file_name, dpi = 800)
        plt.close()

def find_gaussian_peak(gaussian, threshold=0.1):
    
    # Determine the index associated with the brightest value
    max_index = np.argmax(gaussian)

    last_index = len(gaussian) - 1
    # Find the spot to the right of it that hits min_bright
    for i in range(len(gaussian) - max_index):
        if gaussian[max_index + i] <= threshold * gaussian[max_index]:
            last_index = max_index + i
            break

    first_index = 0
    # Find the spot to the left of it that hit min_bright
    for i in range(max_index):
        if gaussian[max_index - i] <= threshold * gaussian[max_index]:
            first_index = max_index - i
            break

    return first_index, last_index

def apply_median_filter(image, size=(3, 3)):
    filtered_image = median_filter(image, size)
    return filtered_image


def subtract_background(image, subtract=575):
    image = image.astype(np.int32)
    image -= subtract
    # Set lower bound of 0. Anything below this bound gets set to the bound value
    image = np.clip(image, 0, None)
    return image
    

def plot_run_sums(folder_name, title=''):
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
        image = subtract_background(image)

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
    ax.xticks(fontsize=14)
    ax.yticks(fontsize=14)
    ax.set_xlabel('"File Number"', fontsize=14)
    ax.set_ylabel("Full Sum", fontsize=14)
    plt.show()

def get_all_images(folder_name):
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

        images.append(image)

    return images

def get_useful_images(folder_name):
    images = get_all_images(folder_name)
    image_sums = [(image, full_sum(image)) for image in images]

    # Sort by sum
    image_sums.sort(key=lambda x: x[1])

    lowest_sums = [image_sum[1] for image_sum in image_sums[:5]]
    highest_sums = [image_sum[1] for image_sum in image_sums[-5:]]


    lowest_avg = sum(lowest_sums) / len(lowest_sums)
    highest_avg = sum(highest_sums) / len(highest_sums)

    cutoff_line = (highest_avg - lowest_avg) / 2

    filtered_image_sums = [tup for tup in image_sums if tup[1] >= (lowest_avg + cutoff_line)]

    return filtered_image_sums


def get_3d_sum(image):
    # Get x and y gaussian slices of the beam
    xgauss = gaussian_2d(image, 0)
    ygauss = gaussian_2d(image, 1)

    # Get fitted gaussian information
    xpopt = gaussian_curve_fit(xgauss)
    ypopt = gaussian_curve_fit(ygauss)

    # Get volume
    Amplitude = (xpopt[0] + ypopt[0])/2 # Get the average amplitude
    xsd = xpopt[2]
    ysd = ypopt[2]

    Volume = Amplitude * 2 * np.pi * xsd * ysd

    return Volume

def compare_with_geant_2d(gaussian, amplitude, std_dev, title=''):
    # Get the fit for the data
    data_gaussian_popt = gaussian_curve_fit(gaussian)

    # Extract the parameters needed for comparison
    data_amp = data_gaussian_popt[0]
    data_std_dev = data_gaussian_popt[2]

    # Create a list of x-values
    x_values = np.arange(len(gaussian))

    # Shift values so that we can center the gaussians at 0
    x_values -= len(gaussian) // 2

    # Convert to mm
    x_values = x_values * 0.051

    # Get y-values for each gaussian
    data_y = []
    geant_y = []

    for x in x_values:
        data_y.append(gaussian_func(x, data_amp, 0, data_std_dev))
        geant_y.append(gaussian_func(x, amplitude, 0, std_dev))

    fig, ax = plt.subplots()
    ax.plot(x_values, data_y, label='Data')
    ax.plot(x_values, geant_y, label='Geant')
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend()
    plt.show()
