import numpy as np
from . import Image_Processing as IP
from . import Plotting as Plot
from . import Gaussian_Analysis as GA

class Image:
    def __init__(self,
                 filename: str,
                 process: bool = False,
                 mm_per_pixel: float = 0.05487):
        self.filename = filename
        self.image = IP.open_bayer(self.filename)
        self.median_filter_applied = False
        self.background_subtracted = 0
        self.mm_per_pixel = mm_per_pixel
        self.is_pixelspace = False
        self.X = None
        self.Y = None
        self.X_Shift = 0
        self.Y_Shift = 0
        self.x_Gaussian = 0
        self.y_Gaussian = 0
        self.x_fit_params = None
        self.x_fit_cov = None
        self.x_fit_R2 = None
        self.y_fit_params = None
        self.y_fit_cov = None
        self.y_fit_R2 = None
        self.x_Area= 0
        self.y_Area= 0
        self.x_Area_error = 0
        self.y_Area_error = 0
        self.x_centered = False
        self.y_centered = False
        
        self.is_valid = True
        self.failure_reason = None

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
                if self.x_fit_params is not None:
                    self.x_fit_params[1:] *= self.mm_per_pixel 
                if self.y_fit_params is not None:
                    self.y_fit_params[1:] *= self.mm_per_pixel         
                self.x_Area *= self.mm_per_pixel
                self.y_Area *= self.mm_per_pixel
                self.x_Area_error *= self.mm_per_pixel
                self.y_Area_error *= self.mm_per_pixel
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
                if self.x_fit_params is not None:
                    self.x_fit_params[1:] /= self.mm_per_pixel 
                if self.y_fit_params is not None:
                    self.y_fit_params[1:] /= self.mm_per_pixel   
                self.x_Area /= self.mm_per_pixel
                self.y_Area /= self.mm_per_pixel
                self.x_Area_error /= self.mm_per_pixel
                self.y_Area_error /= self.mm_per_pixel
                print("Spatial map converted into pixelspace.")


    def full_sum(self) -> int:
        return np.sum(self.image)



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
                if self.x_fit_params is not None:
                    self.x_fit_params[1] += int(amount)       
            else:
                self.X += amount
                self.X_Shift += amount
                if self.x_fit_params is not None:
                    self.x_fit_params[1] += amount

        elif direction.lower() == 'y':
            if self.is_pixelspace:
                self.Y += int(amount)
                self.X_Shift += int(amount)
                if self.y_fit_params is not None:
                    self.y_fit_params[1] += int(amount)
            else:
                self.Y += amount
                self.Y_Shift += amount
                if self.y_fit_params is not None:
                    self.y_fit_params[1] += amount
        else:
            print("Not a valid direction.")
        
    def center_plot(self, axis = 0):
        '''
        Centers Image at 0 for future use in plotting.  
        '''    
        if axis not in [0, 1]:
            print('Invalid axis entry')
            return
        
        if axis==0:
            ax = 'x'
            # Checking if x_Gaussian is populated
            if self.x_Gaussian == 0: 
                self.gaussian_2d(axis = 0)
            if self.x_fit_params is None:
                self.gaussian_curve_fit(axis=axis)
            shift = -self.x_fit_params[1]
            self.x_centered = True

        else:
            ax = 'y'
            # Checking if y_Gaussian is populated
            if self.y_Gaussian == 0: 
                self.gaussian_2d(axis = 0)
            if self.y_fit_params is None:
                self.gaussian_curve_fit(axis=axis)
            shift = -self.y_fit_params[1]
            self.y_centered = True
        self.shift_spatial_map(direction = ax, amount = shift)
        print("Spatial map in " , ax, " direction centered at 0")

    def apply_median_filter(self):
        '''
        Applies a median filter to the image to help filter out noise.

        WARNING!!! Median filter is not perfect. Some images may have surviving noise
            whose pixel yield is greater than that of our Gaussian's peak.
        '''
        if not self.median_filter_applied:
            self.image = IP.apply_median_filter(self.image)

            # Take note that a filter has been applied
            self.median_filter_applied = True
        else:
            print("Median filter is already applied.")
            print("Skipping median filter application call.")

        

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
            self.background_subtracted -= subtract



    def find_center(self):
        '''Finds the center of the beam using the same concept as center of mass calculation.

            Returns: x_center, y_center
        '''

        x_center, y_center = GA.find_center(self)

        return x_center, y_center
        



    def crop_image(self, xstart=None, xend=None, ystart=None, yend=None):
        '''
        Crops the image and spatial maps (X and Y) to a specified rectangular region.
        Units of parameters should be consistent with whatever spatial units you're using (mm or pixels)

        Parameters:
            xstart, xend: Horizontal (column) pixel/mm limits
            ystart, yend: Vertical (row) pixel/mm limits

        This method modifies the current image in-place.
        '''

        IP.crop_image(self, xstart, xend, ystart, yend)

    

    def plot_3d(self, title='',
                figsize=(10,7),
                axisfontsize=14,
                titlefontsize=18,
                ticksize=12,
                labelpad=10, # Spacing between axis labels/title and graph
                xstart=None,
                xend=None,
                ystart=None,
                yend=None):
        Plot.plot_3d(self)


    def plot_2d(self):
        Plot.plot_2d(self)

    def gaussian_2d(self, axis=0):
        sideimg = GA.gaussian_2d(self, axis = axis)
        if axis == 0:
            self.x_Gaussian = sideimg
        elif axis == 1:
            self.y_Gaussian = sideimg
        else:
            print("ERROR")
        return sideimg


    def gaussian_curve_fit(self, axis= 0):
        return GA.gaussian_curve_fit(self, axis=axis)
    
    def plot_gaussian(self,
                      axis=0,
                      fit = False,
                      fontsize = 14, ticksize = 12, titlesize=20, pointsize=12, 
                      ylabel='Brightness',

                      xleft=None, xright=None,
                      title = None,
                      show = False,
                      save = False,
                      file_name = ''):
        return Plot.plot_gaussian(self, axis=axis, fit = fit, fontsize = fontsize,
                          ticksize = ticksize, titlesize=titlesize, pointsize=pointsize,
                          ylabel=ylabel, xleft=xleft, 
                          xright=xright, title = title,show = show, save = save, 
                          file_name = file_name)
        
        
    def integrate_gaussian(self, axis = 0):
        return GA.integrate_gaussian(self, axis=axis)
        
    #----------------------------------------------------------------------#
    # The next two functions relate to saving and reloading Image objects. # 
    #----------------------------------------------------------------------# 
    
    def to_dict(self):
        """Convert all attributes to JSON-safe form."""
        return {
            "filename": self.filename,
            "median_filter_applied": self.median_filter_applied,
            "background_subtracted": self.background_subtracted,
            "mm_per_pixel": self.mm_per_pixel,
            "is_pixelspace": self.is_pixelspace,
            "X_Shift": self.X_Shift,
            "Y_Shift": self.Y_Shift,
            "x_Area": self.x_Area,
            "y_Area": self.y_Area,
            "x_Area_error": self.x_Area_error,
            "y_Area_error": self.y_Area_error,
            "x_centered": self.x_centered,
            "y_centered": self.y_centered,
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
            
            # Arrays -> lists
            "image": self.image.tolist() if self.image is not None else None,
            "X": self.X.tolist() if isinstance(self.X, np.ndarray) else None,
            "Y": self.Y.tolist() if isinstance(self.Y, np.ndarray) else None,
            "x_Gaussian": self.x_Gaussian if isinstance(self.x_Gaussian, list) else None,
            "y_Gaussian": self.y_Gaussian if isinstance(self.y_Gaussian, list) else None,


            # Fit params and covariance (lists or np arrays)
            "x_fit_params": self.x_fit_params.tolist() if isinstance(self.x_fit_params, np.ndarray) else self.x_fit_params,
            "y_fit_params": self.y_fit_params.tolist() if isinstance(self.y_fit_params, np.ndarray) else self.y_fit_params,
            "x_fit_cov": self.x_fit_cov.tolist() if isinstance(self.x_fit_cov, np.ndarray) else self.x_fit_cov,
            "y_fit_cov": self.y_fit_cov.tolist() if isinstance(self.y_fit_cov, np.ndarray) else self.y_fit_cov,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Rebuild Image object from JSON-safe dictionary."""
        img = cls(filename=data["filename"],
                  process=False,
                  mm_per_pixel=data["mm_per_pixel"])

        # Basic flags  and data
        img.median_filter_applied = data["median_filter_applied"]
        img.background_subtracted = data["background_subtracted"]
        img.is_pixelspace = data["is_pixelspace"]
        img.X_Shift = data["X_Shift"]
        img.Y_Shift = data["Y_Shift"]
        img.x_Area = data["x_Area"]
        img.y_Area = data["y_Area"]
        img.x_Area_error = data["x_Area_error"]
        img.y_Area_error = data["y_Area_error"]
        img.x_centered = data["x_centered"]
        img.y_centered = data["y_centered"]
        img.is_valid = data["is_valid"]
        img.failure_reason = data["failure_reason"]

        # Rebuilding Arrays
        img.image = np.array(data["image"]) if data["image"] is not None else None
        img.X = np.array(data["X"]) if data["X"] is not None else None
        img.Y = np.array(data["Y"]) if data["Y"] is not None else None

        # Gaussian 1D slices
        img.x_Gaussian = data["x_Gaussian"] if data["x_Gaussian"] is not None else None
        img.y_Gaussian = data["y_Gaussian"] if data["y_Gaussian"] is not None else None

        # Fit parameters
        img.x_fit_params = np.array(data["x_fit_params"]) if data["x_fit_params"] is not None else None
        img.y_fit_params = np.array(data["y_fit_params"]) if data["y_fit_params"] is not None else None

        # Fit covariances
        img.x_fit_cov = np.array(data["x_fit_cov"]) if data["x_fit_cov"] is not None else None
        img.y_fit_cov = np.array(data["y_fit_cov"]) if data["y_fit_cov"] is not None else None

        return img



        
        
        