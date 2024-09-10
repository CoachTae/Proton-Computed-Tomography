import ARW_Support_2 as supp2
import json
import sys
import Constants as const


def create_database():
    # Main object to be saved as a JSON
        # Each entry is a folder and each key is the folder name
    database = {}

    '''
    database -> ['folder name' : folder_object]

    folder_object -> ['file name' : file_object]

    file_object -> ['file type' : file_data] file type is [raw, filtered, subtracted, autosubtracted]

    file_data -> everything else
    '''

    folders = ['Ce 80 MeV',
               'Ce 100 MeV',
               'Ce 120 MeV',
               'Ce 140 MeV',
               'Ce 160 MeV',
               'Ce 180 MeV',
               'Ce 200 MeV',
               'Ce 220 MeV',
               'MB 200 MeV (Bone Only)',
               'MB 200 MeV (Muscle Only)',
               'Ph 220 MeV (2 Blocks)',
               'Ph 220 MeV (4 Blocks)',
               'Ph 220 MeV (6 Blocks)']

    bad_files = []
    for folder in folders:
        # This dictionary will contain all images and their data for the folder
        folder_object = {}

        # Pull image name and data from every ARW file in the folder
        all_images = Support.get_useful_images(folder, include_names=True)

        for image_data in all_images:
            file_object = {}
            
            file_name = image_data[2]
            raw_image = image_data[0]

            file_types = ['Subtracted', 'Autosubtracted']

            for i, file_type in enumerate(file_types):
                file_data = {}

                if i == 0: # Subtracted
                    image = Support.apply_median_filter(raw_image)
                    image = Support.subtract_background(image)

                elif i == 1: # Autosubtracted
                    image = Support.apply_median_filter(raw_image)
                    image = Support.subtract_background(image, autosubtraction=True)

                Volume, Volume_error = Support.full_sum(image, include_error=True)

                xgaussian = Support.gaussian_2d(image, 0)
                ygaussian = Support.gaussian_2d(image, 1)

                xgaussian_distances = Support.get_distances(xgaussian)
                ygaussian_distances = Support.get_distances(ygaussian)

                xgaussian_shift = Support.peak_finder(xgaussian)
                ygaussian_shift = Support.peak_finder(ygaussian)

                try:
                    xpopt, xpcov, xR2 = Support.gaussian_curve_fit(xgaussian, include_errors=True, corr=True)
                    ypopt, ypcov, yR2 = Support.gaussian_curve_fit(ygaussian, include_errors=True, corr=True)

                    xArea, xArea_sigma = Support.integrate_gaussian(xpopt, xpcov, include_error=True)
                    yArea, yArea_sigma = Support.integrate_gaussian(ypopt, ypcov, include_error=True)
                except:
                    xpopt, xpcov, xR2 = Support.gaussian_curve_fit(xgaussian, include_errors=True, corr=True)
                    ypopt, ypcov, yR2 = Support.gaussian_curve_fit(ygaussian, include_errors=True, corr=True)

                    xArea, xArea_sigma = Support.integrate_gaussian(xpopt, xpcov, include_error=True)
                    yArea, yArea_sigma = Support.integrate_gaussian(ypopt, ypcov, include_error=True)

                    bad_files.append(file_name)
                

                

                #file_data['Image'] = image.tolist()
                file_data['Pixel Sum'] = int(Support.full_sum(image, integration=False))
                file_data['Volume'] = Volume
                file_data['Volume Sigma'] = Volume_error
                file_data['X Gaussian'] = xgaussian
                file_data['X Gaussian Distances'] = xgaussian_distances.tolist()
                file_data['X Gaussian Shift'] = xgaussian_shift
                file_data['X Popt'] = xpopt.tolist()
                file_data['X Pcov'] = xpcov.tolist()
                file_data['X R2'] = xR2
                file_data['X Area'] = xArea
                file_data['X Area Sigma'] = xArea_sigma
                
                file_data['Y Gaussian'] = ygaussian
                file_data['Y Gaussian Distances'] = ygaussian_distances.tolist()
                file_data['Y Gaussian Shift'] = ygaussian_shift
                file_data['Y Popt'] = ypopt.tolist()
                file_data['Y Pcov'] = ypcov.tolist()
                file_data['Y R2'] = yR2
                file_data['Y Area'] = yArea
                file_data['Y Area Sigma'] = yArea_sigma
                
                

                file_object[file_type] = file_data
                print('Finished ' + file_name + ' ' + file_type)
                

            folder_object[file_name] = file_object
            print('\n\n\n\n Finished all of ' + file_name + '.')
            
        
        database[folder] = folder_object
        print('\n\n\n\n  Finished all of ' + folder)
        

    print("DONE!!!")

    print('\n\n\n\n', bad_files)

    with open('Database.json', 'w') as file:
        json.dump(database, file, indent=4)


def update_database():
    '''
    For this iteration, we are removing all instances of autosubtracted data sets
    as well as removing the need to specify "Subtracted" when calling data.
    '''

    with open("Database.json", 'r') as file:
        database = json.load(file)

    new_db = {}
    bad_method = 'Autosubtracted'

    for folder, files in database.items():
        new_db[folder] = {}
        for file, methods in files.items():
            if bad_method in methods:
                del methods[bad_method]

            remaining_method = next(iter(methods))
            new_db[folder][file] = methods[remaining_method]

    with open("New Database.json", 'w') as file:
        json.dump(new_db, file, indent=4)

if __name__ == '__main__':
    update_database()

