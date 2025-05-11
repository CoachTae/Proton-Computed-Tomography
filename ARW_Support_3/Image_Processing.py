import os
import rawpy
import numpy as np
import sys
from scipy.ndimage import median_filter

def open_bayer(file: str) -> np.ndarray:
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
        print("Error occured in 'open_bayer' function.")
        sys.exit()








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
        print("Autosubtraction is not currently supported. System will now close.")
        sys.exit()

    else:
        image -= subtract
        # Set lower bound of 0. Anything below 0 gets set to 0
        image = np.clip(image, 0, None)
        return image


    

def crop_image(Image, xstart, xend, ystart, yend):
    '''
    Crops the image and spatial maps (X and Y) to a specified rectangular region.
    Units of parameters should be consistent with whatever spatial units you're using (mm or pixels)

    Parameters:
        Image: Image object from the ARW3 package.
        xstart, xend: Horizontal (column) pixel/mm limits
        ystart, yend: Vertical (row) pixel/mm limits

    This method modifies the current image in-place.
    '''

    if Image.X is None or Image.Y is None:
        Image.get_spatial_map()

    # Find closest indices corresponding to the provided value
    x_axis = Image.X[0] # Take one row
    y_axis = Image.Y[:,0] # Take on column

    # Find index closest to each spatial bound
    if xstart is None:
        xstart_idx = 0
    else:
        xstart_idx = np.argmin(np.abs(x_axis - xstart))


    if xend is None:
        xend_idx = len(x_axis)
    else:
        xend_idx = np.argmin(np.abs(x_axis - xend)) + 1 # +1 to be inclusive



    if ystart is None:
        ystart_idx = 0
    else:
        ystart_idx = np.argmin(np.abs(y_axis - ystart))


    if yend is None:
        yend_idx = len(y_axis)
    else:
        yend_idx = np.argmin(np.abs(y_axis - yend)) + 1


    # Apply crop
    Image.image = Image.image[ystart_idx:yend_idx, xstart_idx:xend_idx]

    # Also crop the spatial maps if they exist
    if Image.X is not None and Image.Y is not None:
        Image.X = Image.X[ystart_idx:yend_idx, xstart_idx:xend_idx]
        Image.Y = Image.Y[ystart_idx:yend_idx, xstart_idx:xend_idx]



