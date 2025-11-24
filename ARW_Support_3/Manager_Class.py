import json
from pathlib import Path
try:
    from . import Image_Class as Image
except:
    import Image_Class as Image 

class Manager:
    def __init__(self):
        self.images = {}
    
    def create_image(self, name, filename, process=False, mm_per_pixel=0.05487):
        '''
        creates Image_Class Image object 
        '''
        img = Image(filename, process=process, mm_per_pixel=mm_per_pixel)
        self.images[name] = img
        return img

    def save_image(self, name: str, output_path: str):
        '''
        Saves Image object to JSON file.
        '''
        if name not in self.images:
            raise ValueError(f"No image named '{name}'")

        img = self.images[name]
        data = img.to_dict()

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved '{name}' → {output_path}")
    
    def load_image(self, name: str, filepath: str):
        '''
        Reloads existng Image object from JSON file.
        '''
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        img = Image.from_dict(data)
        self.images[name] = img
        print(f"Loaded '{name}' from {filepath}")

        return img