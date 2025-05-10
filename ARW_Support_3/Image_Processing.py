import os
import rawpy
import numpy as np
import sys

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
