import ARW_Support as arwsupp
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import rawpy
import time
import ARW_Support_2 as supp2
import json

start_time = time.time()


#----------------CONSTANTS------------------------------------------------------
runs = [80, 100, 120, 140, 160, 180, 200, 220]

# Define protons_per_frame
protons_per_frame = [29380029.8, 27616468, 24680895.2, 26138120, 19490765.8, 28561906, 32837711.8, 29007595.4]

real_energies = [80.3, 100.7, 119.7, 140.2, 159.9, 179.5, 200.4, 221.3]


# MEDIAN RUNS
mev80 = "SC00100.ARW"
mev100 = "SC00161.ARW"
mev120 = "SC00183.ARW"
mev140 = "SC00247.ARW"
mev160 = "SC00071.ARW"
mev180 = "SC00266.ARW"
mev200 = "SC00318.ARW"
No_Block = "SC00372.ARW"
Block5cm = "PH00012.ARW"
Block10cm = "PH00074.ARW"
Block15cm = "PH00107.ARW"
muscle = "MB00000.ARW"
bone = "MB00040.ARW"


#------------------------Test Code----------------------------------------------


with open('Database.json', 'r') as file:
    database = json.load(file)

medians = [mev80, mev100, mev120, mev140, mev160, mev180, mev200, No_Block, bone, muscle, Block5cm, Block10cm, Block15cm]

regular_runs = ['80 MeV', '100 MeV', '120 MeV', '140 MeV', '160 MeV', '180 MeV', '200 MeV', '220 MeV']

tissue_runs = ['Muscle', 'Bone']

acrylic_runs = ['2 Blocks', '4 Blocks', '6 Blocks']
acrylic_run_files = [Block5cm, Block10cm, Block15cm]
acrylic_runs_cm = ['5cm Acrylic', '10cm Acrylic', '15cm Acrylic']

for i, folder in enumerate(acrylic_runs):
    total_sum = 0
    num_files = 0
    for file in database[folder].keys():
        total_sum += database[folder][file]['Subtracted']['X Popt'][2]
        num_files += 1
    avg = total_sum / num_files
    print(f'Average: {avg}')
sys.exit()


images = supp2.get_useful_images('Ce 80 MeV', include_names=True)
for image in images:
    print(image[2])

sys.exit()

#-----------------------Background Subtraction Algorithm-----------------------------------------------
image = arwsupp.open_bayer(None, file=No_Block)
image = arwsupp.apply_median_filter(image)
image = arwsupp.subtract_background(image)
gaussian = arwsupp.gaussian_2d(image, 0)
_, corr = arwsupp.gaussian_curve_fit(gaussian, corr=True)
print("Corr: ", corr)
arwsupp.plot_gaussian(gaussian, fit=True)

best_subtraction = 0

for exponent in range(3):
    corrs = []
    for i in range(10):
        subtraction = i * (10 ** (2 - exponent))
        image_adjusted = arwsupp.subtract_background(image, subtract=subtraction)
        gaussian = arwsupp.gaussian_2d(image_adjusted, 0)
#gaussian = gaussian[np.argmax(gaussian)-235:np.argmax(gaussian)+235]
#arwsupp.plot_gaussian(gaussian, fit=True)

        _, corr = arwsupp.gaussian_curve_fit(gaussian, corr=True)
        corrs.append(corr)
    best_subtraction += np.argmax(corrs) * (10 ** (2 - exponent))

print("Best Subtraction: ", best_subtraction)

sys.exit()


#----------------------Find Gaussian Data----------------------------------------------
arwsupp.plot_run_sums("Ce 220 MeV", title='220 MeV (No Phantom)')
sys.exit()


image = arwsupp.open_bayer(None, file="MB00040.ARW")
image = arwsupp.apply_median_filter(image)
image = arwsupp.subtract_background(image)
gaussianx = arwsupp.gaussian_2d(image, 0)

files = [Block5cm, Block10cm, Block15cm, "MB00000.ARW", "MB00040.ARW"]
filenames = ['5cm Acrylic', '10cm Acrylic', '15cm Acrylic', '5cm Muscle', '5cm Bone']
for i, file in enumerate(files):
    image = arwsupp.open_bayer(None, file=file)
    image = arwsupp.apply_median_filter(image)
    image = arwsupp.subtract_background(image)
    gaussianx = arwsupp.gaussian_2d(image, 0)
    gaussian = gaussianx
    arwsupp.plot_gaussian(gaussian, fit=True, title=filenames[i])
sys.exit()

gaussiany = arwsupp.gaussian_2d(image, 1)
popt, pcov = arwsupp.gaussian_curve_fit(gaussianx, include_errors=True)
xSD = popt[2]
xSD_error = np.sqrt(pcov[2][2])
popt, pcov = arwsupp.gaussian_curve_fit(gaussiany, include_errors=True)
ySD = popt[2]
ySD_error = np.sqrt(pcov[2][2])

A = popt[0]
A_error = np.sqrt(pcov[0][0])
Area = arwsupp.get_3d_sum(image)
Area_sigma = 2*np.pi*np.sqrt((xSD * ySD * A_error)**2 + (A * ySD * xSD_error)**2 + (A * xSD * ySD_error)**2)

A = round(A, 3)
A_error = round(A_error, 3)
Area = round(Area, 3)
Area_sigma = round(Area_sigma, 3)
xSD = round(xSD, 3)
xSD_error = round(xSD_error, 3)
ySD = round(ySD, 3)
ySD_error = round(ySD_error, 3)




print("95% confidence intervals:")
print('Amplitude: ', A, " +/- ", 1.96*A_error)
print('Standard Deviation (x): ', xSD, " +/- ", 1.96*xSD_error)
print('Standard Deviation (y): ', ySD, " +/- ", 1.96*ySD_error)
print("Area: ", Area, " +/-", 1.96*Area_sigma)
print('\n\n\n\n')

print("99% confidence intervals:")
print('Amplitude: ', A, " +/- ", 2.58*A_error)
print('Standard Deviation (x): ', xSD, " +/- ", 2.58*xSD_error)
print('Standard Deviation (y): ', ySD, " +/- ", 2.58*ySD_error)
print("Area: ", Area, " +/-", 2.58*Area_sigma)



sys.exit()

amplitude = popt[0]
arwsupp.compare_with_geant_2d(gaussian, amplitude, std_dev, title='15cm Phantom')

sys.exit()

#------------------PLOT FITS---------------------------------------------------
'''x_values = []
for i in range(0, 40000):
    x_values.append(i/10)
x_values = np.array(x_values)

fig, ax = plt.subplots()

for i, popt in enumerate(median_fits):
    gaussian_values = arwsupp.gaussian_func(x_values, *popt)
    if i == 0:
        name = 'No Phantom'
    elif i == 1:
        name = '2 Blocks'
    elif i == 2:
        name = '4 Blocks'
    elif i == 3:
        name = '6 Blocks'

    ax.plot(x_values, gaussian_values, label=name)
plt.legend()
ax.set_xlabel('Pixel Number')
ax.set_ylabel('Brightness')
ax.set_title('Median Gaussian Behavior')
plt.show()

sys.exit()'''
#------------------GET FITS----------------------------------------------------

folder_name = 'Ce 220 MeV'


# Get images with decent beam
good_images = arwsupp.get_useful_images(folder_name)

# Extract image data
images = []
for image_data in good_images:
    images.append(image_data[0])

#Apply median filter and background shift
for i, image in enumerate(images):
    image = arwsupp.apply_median_filter(image)
    image = arwsupp.subtract_background(image)
    gaussian = arwsupp.gaussian_2d(image, 0)
    popt, pcov = arwsupp.gaussian_curve_fit(gaussian, include_errors=True)
    errors = np.sqrt(np.diag(pcov))
    area = arwsupp.integrate_gaussian(popt)
    print(popt[0], '\t', errors[0], '\t', popt[2], '\t', errors[2], '\t', area)
    images[i] = image
    arwsupp.plot_gaussian(gaussian, fit=True)


print("Time elapsed: ", time.time() - start_time)
