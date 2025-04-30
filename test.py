import ARW_Support as arwsupp
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import rawpy
import time
import ARW_Support_2 as supp2
from ARW_Support_3.Image_Class import Image as img
import json
import Constants as const
import math

start_time = time.time()


#------------------------Test Code----------------------------------------------

with open('Database.json', 'r') as file:
    database = json.load(file)

for folder in database.keys():
    for file in database[folder].keys():

        Image = img(file, process=True)
        x_cen, y_cen = Image.find_center()
        print(f'{folder} -> {file}:')
        print(x_cen)
        print(y_cen)
        #Image.crop_image(int(x_cen-450), int(x_cen+450), int(y_cen-450), int(y_cen+450))
        #Image.plot_3d()
sys.exit()



image = supp2.open_bayer("SC00161.ARW")
supp2.plot_3d(image, xstart=80, xend=180, ystart=80, yend=180)
sys.exit()


with open('Database.json', 'r') as file:
    database = json.load(file)

for folder in database.keys():
    amps = []
    amp_errors = []
    for image in database[folder].keys():
        amps.append(database[folder][image]['X Popt'][0])
        amp_errors.append(math.sqrt(database[folder][image]['X Pcov'][0]))

    print(f'For {folder}:')
    print(f'Amplitude = {round(np.mean(amps),3)} +/- {round(2.576*np.mean(amp_errors)/len(amp_errors), 3)}')
    
        
