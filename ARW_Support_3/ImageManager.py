import json

class ImageManager:
    
    @staticmethod
    def create_image(filename: str,
         process: bool =False,
         mm_per_pixel = 0.05487) -> Image:
        """Create an Image object."""
        return Image