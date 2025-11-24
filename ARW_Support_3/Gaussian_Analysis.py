import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def find_center(Image) -> tuple[float, float]:
    '''
    Intended to find the center of the Gaussian beam in the current representation
    space.This method is intended as a rough estimate of the center. 
    For more accurate center, use paramaters found from gaussian_curve_fit()
    Parameters:
        Image: Image object as defined in Image_Class.py

    Returns:
        x_center, y_center
    '''

    # Make sure we have a space to operate in
    if Image.X is None or Image.Y is None:
        Image.get_spatial_map()

    image = Image.image.astype(np.float64)
    X = Image.X.astype(np.float64)
    Y = Image.Y.astype(np.float64)

    total = np.sum(image)
    x_center = np.sum(image * X) / total
    y_center = np.sum(image * Y) / total

    return x_center, y_center

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

def weighted_mean(x, y):
    return np.sum(x * y) / np.sum(y)

def weighted_variance(x, y, mu):
    return np.sqrt(np.sum(y * (x - mu)**2) / np.sum(y))

def estimate_amplitude(x, y, mu, sigma):
    
    # Convert the mean to an int
    mean = int(mu)
    # Estimate amplitude as the value of y at the mean
    A_estimate = x[mean]
    
    return A_estimate

def gaussian_2d(Image, axis=0):
    '''
    Creates a side-profile from a 2D image.
    Side profile is made by taking the silhouette (max values) as seen from 1 side

    Image: Image_Class object

    axis: 0 or 1. Determines which direction we'll see the gaussian's
        side-profile from

    returns: 1D List of pixel values
    '''
    image = Image.image #pulls 2D numpy array attribute for object Image.
    
    data = []
    if axis == 1:

        # For each row, find the brightest pixel to emulate a side-on view
        # Values are turned to float (from int64) to allow JSON serialization
        for i in range(image.shape[0]):  # image.shape[0] gives number of rows
            max_brightness = np.max(image[i, :])  # Max value across columns
            data.append(float(max_brightness))  # Convert to float for consistency
        return data
    elif axis == 0:
        for i in range(image.shape[1]):  # image.shape[1] gives number of columns
            max_brightness = np.max(image[:, i])  # Max value across rows
            data.append(float(max_brightness))  # Convert to float for consistency
        return data

def gaussian_curve_fit(Image, axis=0):
    '''
    Fits a 1D Gaussian to the side-profile of the image along the specified axis.

    Parameters
    ----------
    Image: image object
    axis : str
        'x' or 'y' — which Gaussian profile to fit.
    Returns
    -------
    None. Populates fit paramaters, cov, and R2 for each axis.
    '''

    # Select which axis to analyze; x=0 y=1
    
    if axis == 0:
        if Image.x_Gaussian == 0: 
            Image.gaussian_2d(axis = 0)
        vertical = Image.x_Gaussian
        horizontal = Image.X[0,:]
    elif axis == 1:
        if Image.y_Gaussian == 0: 
            Image.gaussian_2d(axis = 1)
        vertical = Image.y_Gaussian 
        horizontal = Image.Y[:,0]
    else:
        print("invalid axis input")
        return 
    
    # Obtain initial guesses for fit
    
    mu = weighted_mean(horizontal, vertical)
    
    sigma = weighted_variance(horizontal, vertical, mu)

    amp = estimate_amplitude(horizontal, vertical, mu, sigma)
    
    # Fit curve:
    popt, pcov = curve_fit(
        gaussian_func, horizontal, vertical,
        p0=[amp, mu, sigma] , bounds=([0,0,0],[np.inf, np.inf, np.inf])
    )

    # Compute fitted Gaussian
    fitted = gaussian_func(horizontal, *popt)

    # Compute correlation coefficient

    SSR = np.sum((fitted - vertical) ** 2)
    SST = np.sum((vertical - np.mean(vertical)) ** 2)
    R2 = 1 - SSR / SST

    # Handle covariance output
    pcov_out = np.sqrt(np.diag(pcov))

    # Store results in Image object
    if axis == 0:
        Image.x_fit_params = popt
        Image.x_fit_cov = pcov_out
        Image.x_fit_R2 = R2
    else:
        Image.y_fit_params = popt
        Image.y_fit_cov = pcov_out
        Image.y_fit_R2 = R2

    
    
    

def integrate_gaussian(Image, axis = 0):
    '''
    Provides analytical solution to the integral of a gaussian.
    
    Parameters
    ----------
    Image: image object
    axis : str
        'x' or 'y' — which Gaussian profile to fit.
    Returns
    -------
    None. Populates Area and error for each axis
    
    '''
    
    if axis == 0:
        popt = Image.x_fit_params
        pcov = Image.x_fit_cov
    elif axis == 1:
        popt = Image.y_fit_params
        pcov = Image.y_fit_cov
    else:
        print("invalid axis input")
        return 

    # Extract values from popt
    Amplitude = popt[0]
    Mean = popt[1]
    SD = popt[2]

    # If pcov is given, assign the values
    Amplitude_error = pcov[0]
    Mean_error = pcov[1]
    SD_error = pcov[2]

    # ----- Integration section -----
    Area = Amplitude * np.sqrt(2 * np.pi) * SD

    first_term = SD**2 * Amplitude_error**2
    second_term = Amplitude**2 * SD_error**2

    Area_error = np.sqrt(2*np.pi * (first_term + second_term))
    
    if axis == 0:
        Image.x_Area= Area
        Image.x_Area_error = Area_error
    else:
        Image.y_Area= Area
        Image.y_Area_error = Area_error
    

    
    
    
    
    
    