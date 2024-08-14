import numpy as np
import ARW_Support as arwsupp
import rawpy
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import integrate
import sys
#-----------FIND MEDIAN SUM------------------------------------------------
# Expects Images folder -> ARW Files folder -> folder_name folder
folder_name = 'MB 200 MeV'
folder_path = 'Images/ARW Files/'+ folder_name + '/'

# This says that we'll choose all ARW files
pattern = '*.ARW'

# This will store the array for each file
images = []

print(folder_path)
# For each ARW file in folder...
for file_path in glob.glob(folder_path + pattern):
    # Open the raw data
    with rawpy.imread(file_path) as raw:
        image = raw.raw_image.copy()

    images.append((image, file_path))

image_sums = [(image, arwsupp.full_sum(image), file_path) for image, file_path in images]

# Sort by sum
image_sums.sort(key=lambda x: x[1])

lowest_sums = [image_sum[1] for image_sum in image_sums[:5]]
highest_sums = [image_sum[1] for image_sum in image_sums[-5:]]


lowest_avg = sum(lowest_sums) / len(lowest_sums)
highest_avg = sum(highest_sums) / len(highest_sums)

cutoff_line = (highest_avg - lowest_avg) / 2

filtered_image_sums = [tup for tup in image_sums if tup[1] >= (lowest_avg + cutoff_line)]

sums = []
for image_data in filtered_image_sums:
    image_sum = image_data[1]
    sums.append(image_sum)

sums.sort()

numsums = len(sums)

if numsums % 2 == 0:
    index = int((numsums + 1) / 2)
    median = sums[index]
elif numsums % 2 == 1:
    index = int(numsums / 2)
    median = sums[index]

print(filtered_image_sums[index][2])
sys.exit()
