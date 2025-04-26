import numpy as np
import Image_Processing as IP

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

        if process:
            # We can choose to automatically filter and subtract background
            self.apply_median_filter()
            self.subtract_background()



    def pixelspace_off(self):
        self.is_pixelspace = False
        self.get_spatial_map()
        print("Pixelspace has been turned off.")


    def pixelspace_on(self):
        self.is_pixelspace = True
        self.get_spatial_map()
        print("Pixelspace has been turned on.")



    def get_spatial_map(self, pixelspace: bool = None) -> None:
        '''
        Creates 2D arrays representing space.

        If we just plot an image, the origin is at the top left corner.
        This allows us to shift the origin around.
        Rather than the plotting function using indices from our image to determine locations, we can give it these arrays instead.

        X is a 2D grid where each number tells you that pixels position in x. (If you travel vertically, numbers are identical)
        Y is a 2D grid where each number tells you that pixels position in y. (If you travel horizontally, numbers are identical)
        '''
        if pixelspace is None:
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

        if self.X is None or self.Y is None:
            self.get_spatial_map()
        
        if direction.lower() == 'x':
            self.X += amount

        elif direction.lower() == 'y':
            self.Y += amount

        else:
            print("Not a valid direction.")
        
    

    def apply_median_filter(self):
        self.image = IP.apply_median_filter(self.image)

        # Take note that a filter has been applied
        self.median_filter_applied = True

        

    def subtract_background(self,
                            subtract = 575,
                            autosubtract = False):
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

            Returns:
        

    
