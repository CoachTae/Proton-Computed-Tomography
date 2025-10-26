import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def find_center(Image) -> tuple[float, float]:
    '''
    Intended to find the center of the Gaussian beam in the current representation
    space.

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
    # Find a local window around the mean, e.g., ±3 standard deviations
    window_size = 3 * sigma
    mask = (x >= (mu - window_size)) & (x <= (mu + window_size))
    # Local window in x and y
    x_window = x[mask]
    y_window = y[x_window[0]:x_window[-1]]
    
    # Estimate amplitude as the maximum value in the window
    A_estimate = np.max(y_window)
    
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

def gaussian_curve_fit(Image,
                       axis=0,
                       include_errors=False,
                       pcov_list=True, 
                       corr=False, 
                       minSD=1,
                       shift=None):
    '''
    Fits a 1D Gaussian to the side-profile of the image along the specified axis.

    Parameters
    ----------
    axis : str
        'x' or 'y' — which Gaussian profile to fit.
    include_errors : bool
        If True, returns covariance information.
    pcov_list : bool
        If True, return sqrt(diagonal) of covariance matrix (1σ errors).
    corr : bool
        If True, compute R² correlation coefficient.
    minSD : float
        Minimum allowed standard deviation.
    pixelspace : bool
        Override for using pixel or mm space.
    shift : float
        Optional manual shift of x-values before fitting.
    Returns
    -------
    popt : list
        [Amplitude, Mean, SD]
    pcov : array or list (optional)
        Covariance or 1σ errors of parameters.
    R2 : float (optional)
        Coefficient of determination (goodness of fit).
    '''
    # Ensure we are in pixel space   
    if not Image.is_pixelspace:
        Image.pixelspace_on()
    
    # Select which axis to analyze; x=0 y=1
    
    if axis == 0:
        vertical = Image.x_Gaussian
        horizontal = Image.X[0]
    elif axis == 1:
        vertical = Image.y_Gaussian 
        horizontal = Image.Y[:,0]
    else:
        print("invalid axis input")
        return 
    
    # Obtain initial guesses for fit
    
    mu = weighted_mean(np.arange(len(vertical)), vertical)
    
    sigma = weighted_variance(np.arange(len(vertical)), vertical, mu)

    amp = estimate_amplitude(np.arange(len(vertical)), vertical, mu, sigma)

    
    # Handle x-axis shifting
    '''
    if shift is not None:
        horizontal -= shift
    else:
        x_cen, y_cen = Image.find_center()
        shift = x_cen if axis == 0 else y_cen
        horizontal -= shift
    '''
    
    # Fit curve
    popt, pcov = curve_fit(
        gaussian_func, horizontal, vertical,
        p0=[amp, mu, max(sigma, minSD)]
    )

    # Ensure SD positive and re-center mean
    '''
    popt[2] = abs(popt[2])
    popt[1] = 0.0
    '''
    # Compute fitted Gaussian
    fitted = gaussian_func(horizontal, *popt)

    # Optionally compute correlation coefficient
    R2 = None
    if corr:
        SSR = np.sum((fitted - vertical) ** 2)
        SST = np.sum((vertical - np.mean(vertical)) ** 2)
        R2 = 1 - SSR / SST

    # Handle covariance output
    if include_errors:
        pcov_out = np.sqrt(np.diag(pcov)) if pcov_list else pcov
    else:
        pcov_out = None

    # Store results dynamically in Image object
    if axis == 0:
        Image.x_fit_params = popt
        Image.x_fit_cov = pcov_out
        Image.x_fit_R2 = R2
    else:
        Image.y_fit_params = popt
        Image.y_fit_cov = pcov_out
        Image.y_fit_R2 = R2


    # Return requested outputs
    if include_errors and corr:
        return popt, pcov_out, R2
    elif include_errors:
        return popt, pcov_out
    elif corr:
        return popt, R2
    else:
        return popt
        
    

    
    
    
    
    