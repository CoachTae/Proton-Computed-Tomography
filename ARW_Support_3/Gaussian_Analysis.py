import sys
import numpy as np
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
    if axis == 1:

        # For each row, find the brightest pixel to emulate a side-on view
        # Values are turned to float (from int64) to allow JSON serialization
        for i in range(len(image[:,0])):
            max_brightness = np.max(image[i,:])
            data.append(float(max_brightness))
        return data
    elif axis == 0:
        for i in range(len(image[0,:])):
            max_brightness = np.max(image[:,i])
            data.append(float(max_brightness))
        return data


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


def gaussian_curve_fit(gaussian

