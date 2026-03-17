import os
import rawpy
import numpy as np
import glob
import sys
from scipy.ndimage import median_filter

# --- Basler ace2 a2A2448-75ucPRO native resolution ---
BASLER_WIDTH  = 2448
BASLER_HEIGHT = 2048


def open_bayer(file: str, normalize=False, verbose=False) -> np.ndarray:
    """
    Opens Bayer-layer data from Sony ARW or Basler RAW files.

    Sony ARW  -> rawpy
    Basler RAW -> auto-detected binary loader

    Returns
    -------
    2D numpy array of Bayer values
    """

    file_ext = os.path.splitext(file)[1].lower()

    # ==========================================================
    # Sony ARW files (DSLR RAW)
    # ==========================================================
    if file_ext == ".arw":

        base_dir = os.getcwd()
        arw_files_dir = os.path.join(base_dir, 'Images', 'ARW Files')

        found_file_path = None
        for root, dirs, files in os.walk(arw_files_dir):
            if file in files:
                found_file_path = os.path.join(root, file)
                break

        if not found_file_path:
            raise FileNotFoundError(f"{file} not found under {arw_files_dir}")

        with rawpy.imread(found_file_path) as raw:
            image = raw.raw_image.copy()

        if normalize:
            image = image.astype(np.float32)
            image /= image.max()

        if verbose:
            print(f"[ARW] Loaded {file} | shape={image.shape} dtype={image.dtype}")

        return image

    # ==========================================================
    # Basler RAW files (industrial camera)
    # ==========================================================
    elif file_ext == ".raw":

        if not os.path.exists(file):
            raise FileNotFoundError(file)

        file_size = os.path.getsize(file)
        n_pixels = BASLER_WIDTH * BASLER_HEIGHT

        # --- Detect storage format ---
        if file_size == n_pixels:
            dtype = np.uint8
            container = "8-bit"
        elif file_size == n_pixels * 2:
            dtype = np.uint16
            container = "16-bit container"
        else:
            raise ValueError(
                f"[Basler RAW] Unexpected file size: {file_size} bytes "
                f"(expected {n_pixels} or {n_pixels*2})"
            )

        data = np.fromfile(file, dtype=dtype)

        if data.size != n_pixels:
            raise ValueError("[Basler RAW] Pixel count mismatch.")

        image = data.reshape((BASLER_HEIGHT, BASLER_WIDTH))

        # --- Infer likely sensor bit depth ---
        max_val = int(image.max())
        if max_val <= 1023:
            bit_depth = "10-bit"
        elif max_val <= 4095:
            bit_depth = "12-bit"
        else:
            bit_depth = "16-bit"

        if normalize:
            image = image.astype(np.float32)
            image /= image.max()

        if verbose:
            print(f"[Basler RAW] Loaded {file}")
            print(f"             {container}, likely {bit_depth}, "
                  f"shape={image.shape}, dtype={image.dtype}")

        return image

    # ==========================================================
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
        


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


    # Crop spatial maps
    if Image.X is not None and Image.Y is not None:
        Image.X = Image.X[ystart_idx:yend_idx, xstart_idx:xend_idx]
        Image.Y = Image.Y[ystart_idx:yend_idx, xstart_idx:xend_idx]

    # Crop Gaussian side profiles
    if isinstance(Image.x_Gaussian, list):
        Image.x_Gaussian = Image.x_Gaussian[xstart_idx:xend_idx]

    if isinstance(Image.y_Gaussian, list):
        Image.y_Gaussian = Image.y_Gaussian[ystart_idx:yend_idx]


