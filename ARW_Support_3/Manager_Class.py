import json
from pathlib import Path
try:
    from .Image_Class import Image
except ImportError:
    from Image_Class import Image
    
class Manager:
    def __init__(self):
        self.images = {}
        self.image_paths = {}
        self.review_images = {}
    
    def create_image(self, name, filename, process=False, mm_per_pixel=0.05487):
        '''
        creates Image_Class Image object 
        '''
        img = Image(filename, process=process, mm_per_pixel=mm_per_pixel)
        self.images[name] = img
        return img
    
    def create_complete_image(self, name, filename, process=True, crop = False, mm_per_pixel=0.05487):
        '''
        creates Image_Class Image object AND populates all saveable variables for an Image object.
        i.e. processed, spatial maps, side views, fit paramaters, centers plot, obtains area, etc. 
        '''
        img = self.create_image(name=name, filename=filename, process=process, mm_per_pixel=mm_per_pixel)
        
        try:
            img.pixelspace_on()
            
            # -------- X axis --------
            img.gaussian_2d(axis=0)
            img.gaussian_curve_fit(axis=0)
            #img.center_plot(axis=0)
            img.integrate_gaussian(axis=0)
            
            # -------- Y axis --------
            img.gaussian_2d(axis=1)
            img.gaussian_curve_fit(axis=1)
            #img.center_plot(axis=1)
            img.integrate_gaussian(axis=1)
            
            img.is_valid = True
        
        except ValueError as e:
            # Image is not useful hence is removed from images dictionary
            img.is_valid = False
            img.failure_reason = str(e)
            
            print(f"[REJECTED] {name}: {e}")
            self.review_images[name] = img
            self.images.pop(name, None)
            
            
        except Exception as e:
            # Real unexpected error
            print(f"[ERROR] {name}: {e}")
            self.review_images[name] = img
            self.images.pop(name, None)
            raise
        
        return img
        
        
        
    
    def save_image(self, name, output_path=None):
        if name not in self.images:
            raise ValueError(f"No image named '{name}'")
        
        img = self.images[name]
        data = img.to_dict()
    
        # If no path given, try to reuse old one
        if output_path is None:
            if name not in self.image_paths:
                print("No previous save path")
            output_path = self.image_paths[name]
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / f"{name}.json"
            
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        
        self.image_paths[name] = output_path
        print(f"Saved '{name}' → {output_path}")

    def save_images(self, output_dir=None):
        if not self.images:
            print("No images to save.")
            return
    
        for name, img in self.images.items():
            if output_dir is None:
                if name not in self.image_paths:
                    raise ValueError(f"No known save path for '{name}'.")
                path = self.image_paths[name]
            else:
                path = Path(output_dir)
                path.mkdir(parents=True, exist_ok=True)
                path = path / f"{name}.json"
            
            data = img.to_dict()
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            
            self.image_paths[name] = path
            print(f"Saved '{name}' → {path}")
        
        print("\nAll images saved.")

    
    def load_image(self, name: str, filepath: str):
        '''
        Reloads existng Image object from JSON file.
        '''
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        img = Image.from_dict(data)
        self.images[name] = img
        self.image_paths[name] = filepath
        print(f"Loaded '{name}' from {filepath}")

        return img

    def load_images(self, folder):
        folder = Path(folder)
        print(folder)
        if not folder.exists():
            raise FileNotFoundError(folder)
        
        for file in folder.glob("*.json"):
            try:
                name = file.stem
                self.load_image(name=name, filepath=file)
            except Exception as e:
                print(f"Skipped {file.name}: {e}")    
    