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

image = supp2.open_bayer('MB00000.ARW')
image = supp2.apply_median_filter(image)
image = supp2.subtract_background(image, subtract=625)
gaussian = supp2.gaussian_2d(image)
for i, point in enumerate(gaussian):
    gaussian[i] = gaussian[i]/6.208131012
supp2.plot_gaussian(gaussian, fit=True, fontsize=20, ticksize=18, ylabel='Normalized Photon Yield')
sys.exit()

with open('Database.json', 'r') as file:
    database = json.load(file)



for folder in database.keys():
    if folder in const.tissue_runs:
        if const.meanmuscle in database[folder].keys():
            gaussian = database[folder][const.meanmuscle]['X Gaussian']
            gaussian = gaussian/6.208131012
            supp2.plot_gaussian(gaussian, fit=True, fontsize=20, ticksize=18, ylabel='Normalized Photon Yield')
