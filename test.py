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

#%%
#------------------------Test Code----------------------------------------------


with open('Database.json', 'r') as file:
    database = json.load(file)

for folder in database.keys():
    for file in database[folder].keys():

        Image = img(file, process=True)
        x_cen, y_cen = Image.find_center()
        print(f'{folder} -> {file}:')
        Image.gaussian_2d()
        plt.plot(Image.x_Gaussian)
        print('\n\n\n')
sys.exit()

with open('Database.json', 'r') as file:
    database = json.load(file)



for folder in database.keys():
    if folder in const.tissue_runs:
        if const.meanmuscle in database[folder].keys():
            gaussian = database[folder][const.meanmuscle]['X Gaussian']
            gaussian = gaussian/6.208131012
            supp2.plot_gaussian(gaussian, fit=True, fontsize=20, ticksize=18, ylabel='Normalized Photon Yield')
