import ARW_Support as arwsupp
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import rawpy
import time
import ARW_Support_2 as supp2
import json
import Constants as const
import math

start_time = time.time()


#------------------------Test Code----------------------------------------------


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
    
        
