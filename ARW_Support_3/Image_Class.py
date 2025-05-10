import numpy as np
from . import Image_Processing as IP
from . import Plotting as Plot

class Image:
    def __init__(self,
                 filename: str,
                 process: bool =False,
                 mm_per_pixel = 0.05487):
        self.filename = filename
        self.image = IP.open_bayer(self.filename)
        self.median_filter_applied = False
        self.background_subtracted = 0
        self.mm_per_pixel = mm_per_pixel
        self.is_pixelspace = True
        self.X = None
        self.Y = None
        self.X_Shift = 0
        self.Y_Shift = 0

        if process:
            # We can choose to automatically filter and subtract background
            self.apply_median_filter()
            self.subtract_background()




    def pixelspace_off(self):
        '''
        pixelspace is a boolean variable primarily used to convert the pixels
        we see into distances so that we can properly measure the Gaussian
        parameters. If pixelspace is True, distances will be in pixel. If False,
        distances will be in mm.
        '''
        if self.is_pixelspace:
            self.is_pixelspace = False

            # If the map isn't complete or doesn't exist, create it.
            if self.X is None or self.Y is None:            
                self.get_spatial_map()
                print("Spatial map created in distance space.")
            # Otherwise, just shift the current map and shifts to distance space.
            else:
                # Change data type from int array to float array
                self.X = self.X.astype(np.float64)
                self.Y = self.Y.astype(np.float64)

                self.X *= self.mm_per_pixel
                self.Y *= self.mm_per_pixel
                self.X_Shift *= self.mm_per_pixel
                self.Y_Shift *= self.mm_per_pixel
                print("Spatial map converted into distance space.")
        else:
            print("You are already in distance space.")
            print("No changes have been made.")


    def pixelspace_on(self):
        if self.is_pixelspace:
            print("You are already in pixelspace.")
            print("No changes have been made.")
        else:
            self.is_pixelspace = True

            # If the map isn't complete or doesn't exist, create it.
            if self.X is None or self.Y is None:
                self.get_spatial_map()
                print("Spatial map created in pixelspace.")
            # Otherwise, undo the distance conversion
            else:
                self.X /= self.mm_per_pixel
                self.Y /= self.mm_per_pixel
                self.X_Shift /= self.mm_per_pixel
                self.Y_Shift /= self.mm_per_pixel
                print("Spatial map converted into pixelspace.")



    def full_sum(self) -> int:
        return np.sum(Image.image)



    def get_spatial_map(self, pixelspace: bool = None) -> None:
        '''
        Creates 2D arrays representing space.

        If we just plot an image, the origin is at the top left corner.
        This map allows us to shift the origin around.
        Rather than the plotting function using indices from our image to determine locations, we can give it these arrays instead.

        X is a 2D grid where each number tells you that pixels position in x. (If you travel vertically, numbers are identical)
        Y is a 2D grid where each number tells you that pixels position in y. (If you travel horizontally, numbers are identical)
        '''
        if pixelspace is None:
            # User may decide to use pixelspace or not explicitly, otherwise, default will be used.
            pixelspace = self.is_pixelspace
            
        height, width = self.image.shape

        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x,y)


        if not pixelspace:
            X = X * self.mm_per_pixel
            Y = Y * self.mm_per_pixel
            self.is_pixelspace = False

        self.X = X
        self.Y = Y


    def shift_spatial_map(self, direction: str, amount: float) -> None:
        '''
        Allows us to shift the origin around. For example, we can center the image
        on the origin, rather than the origin always being the top-left corner.

        Parameters:
            direction (str): 'x', 'X', 'y', or 'Y'.
                There is no functional difference between 'x' and 'X'. It's just to prevent silly errors.
            amount (float): The amount in which you want to shift. Positive numbers shift in +x and +y, negative shifts in -x and -y.
        '''
        
        if self.X is None or self.Y is None:
            self.get_spatial_map()
        
        if direction.lower() == 'x':
            if self.is_pixelspace:
                self.X += int(amount)
                self.X_Shift += int(amount)
            else:
                self.X += amount
                self.X_Shift += amount

        elif direction.lower() == 'y':
            if self.is_pixelspace:
                self.Y += int(amount)
                self.X_Shift += int(amount)
            else:
                self.Y += amount
                self.Y_Shift += amount

        else:
            print("Not a valid direction.")
        
    

    def apply_median_filter(self):
        '''
        Applies a median filter to the image to help filter out noise.

        WARNING!!! Median filter is not perfect. Some images may have surviving noise
            whose pixel yield is greater than that of our Gaussian's peak.
        '''
        self.image = IP.apply_median_filter(self.image)

        # Take note that a filter has been applied
        self.median_filter_applied = True

        

    def subtract_background(self,
                            subtract = 575,
                            autosubtract = False):
        '''
        Defines how much pixel value we subtract from every pixel in the image.

        Pixels have a lower bound of 0, meaning that if a pixel is reduced to 0
            by background subtraction, we lose information about what its value
            was prior to subtraction. (i.e. If pixel is at 523 but we subtract
            575, the pixel is just at 0, so we can't simply add 575 to get the
            original image back.)

        Autosubtract, if True, will hopefully be able to calculate the optimal
            background subtraction needed such that a Gaussian fit will yield
            the highest correlation coefficient (R^2). Currently this is under
            development and does not work.
        '''
        
        # Initial value of the brightest pixel. Used to determine amount subtracted in case autosubtract was used
        if autosubtract:
            peak_init = np.max(self.image)

        self.image = IP.subtract_background(self.image, subtract, autosubtract)

        # Final value of brightest pixel. Calculate subtraction value. Record it.
        if autosubtract:
            peak_final = np.max(self.image)
            delta_peak = peak_init - peak_final
            self.background_subtracted = delta_peak

        else:
            self.background_subtracted = subtract



    def find_center(self):
        '''Finds the center of the beam using the same concept as center of mass calculation.

            Returns: x_center, y_center
        '''

        x_center, y_center = IP.find_center(self)

        return x_center, y_center
        



    def crop_image(self, xstart, xend, ystart, yend):
        '''
        Crops the image and spatial maps (X and Y) to a specified rectangular region.
        Units of parameters should be consistent with whatever spatial units you're using (mm or pixels)

        Parameters:
            xstart, xend: Horizontal (column) pixel/mm limits
            ystart, yend: Vertical (row) pixel/mm limits

        This method modifies the current image in-place.
        '''

        if self.X is None or self.Y is None:
            self.get_spatial_map()

        # Find closest indices corresponding to the provided value
        x_axis = self.X[0] # Take one row
        y_axis = self.Y[:,0] # Take on column

        # Find index closest to each spatial bound
        xstart_idx = np.argmin(np.abs(x_axis - xstart))
        xend_idx = np.argmin(np.abs(x_axis - xend)) + 1 # +1 to be inclusive

        ystart_idx = np.argmin(np.abs(y_axis - ystart))
        yend_idx = np.argmin(np.abs(y_axis - yend)) + 1


        # Apply crop
        self.image = self.image[ystart_idx:yend_idx, xstart_idx:xend_idx]

        # Also crop the spatial maps if they exist
        if self.X is not None and self.Y is not None:
            self.X = self.X[ystart_idx:yend_idx, xstart_idx:xend_idx]
            self.Y = self.Y[ystart_idx:yend_idx, xstart_idx:xend_idx]



    def plot_3d(self):
        Plot.plot_3d(self)

