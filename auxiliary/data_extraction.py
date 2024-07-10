"""
    This script used the extract_ROI and the roi_processing functions, to prepare all the training and validation data.
    The data is labelled and saved in the Data folder. From the labels the ROI is extracted and processed. Then all the
    ROIs are tagged according to their category, which can be determined by the file from which their labels and image
    come from, and then they are split into training and validation data and saved into a list, which is finally
    exported as JSON file list.
"""

import os
import json
import tensorflow as tf
import numpy as np
from auxiliary.extract_ROI import extract_roi
from auxiliary.roi_processing import roi_processing
from random import shuffle


def extract_data():
    # path of original image
    parent_path = os.getcwd()
    # List of all the files present in the 'Data' folder
    files = os.listdir(os.path.join(parent_path, 'Data'))
    # Creates a list with the paths for the labels of each image in the 'Data' folder
    labels_path = [os.path.join(parent_path, 'Data', i) for i in files if '.json' in i]
    # Initialization of empty lists for each class of plates. Each class has a training and a validation list
    plate1_data_train = []
    plate1_data_validate = []
    plate2_data_train = []
    plate2_data_validate = []
    plate3_data_train = []
    plate3_data_validate = []

    # Iteration over all the label files in the Data folder
    for file_path in labels_path:
        # Get the image path corresponding to the current label file. Since both files have the same name, differing
        # only on the type of file, this can be done by replacing the .json by .jpg
        img_path = file_path.replace('.json', '.jpg')
        # File name is only the last part of the file path
        file_name = os.path.basename(os.path.normpath(file_path))
        # Similarly image name is only the last part of the image path
        img_name = os.path.basename(os.path.normpath(img_path))

        # Opening the JSON file so its data can be extracted
        file = open(file_path)
        # Saving the data from the JSON to a variable
        data = json.load(file)

        # Number of polygons, which are delimiting the ROIS, in the current JSON file
        elements = len(data['shapes'])
        # Number of ROIS that will be dedicated to training the models
        training_size = round(0.7 * elements)

        # Initializing an array that can be used to organize the data coming from the JSON. Technically not
        # necessary, but adding it made implementation simpler.
        data_array = np.empty([elements, 2], dtype=object)
        # Initialization of array that is used to save the relevant data for each ROI
        full_data = []

        # Iteration over all the polygons in the current JSON file
        for index in range(0, elements):
            # Getting the list of points defining the relevant polygon from the JSON file
            points_lst = data['shapes'][index]['points']
            # Conversion of the list of points from floats to integers
            data_array[index, 0] = [[int(a) for a in row] for row in points_lst]
            # Control flow statements to decide which tag is associated to the extracted polygon. This is simply
            # based on the file name from where the list of points is coming from
            if 'Plate1' in file_path:
                # First class of metal plates
                data_array[index, 1] = [0., 0., 1.]
            elif 'Plate2' in file_path:
                # Second class of metal plates
                data_array[index, 1] = [0., 1., 0.]
            elif 'Plate3' in file_path:
                # Third class of metal plates
                data_array[index, 1] = [1., 0., 0.]
        # Closing the current JSON file as it is not needed in further parts of the code
        file.close()

        # Once again iterating over the total number of polygons extracted
        for index in range(0, elements):
            # Statement to let user know which iteration is being run and for which image.
            print(f'Extracting roi #: {index} in labels for: {file_name} from: {img_name}')
            # Conversion of the list of points in row 'index' into a NumPy array. Column zero contains the list of
            # points and column one contains the tag corresponding to the list points in that row
            array = np.array(data_array[index, 0])
            # Extraction of the ROI defined by the array of points from the image corresponding to the labeled data.
            roi = extract_roi(img_path=img_path, roi=array)
            # Processing of the extracted ROI, and flattening of the resulting vector.
            fft = roi_processing(roi=roi).flatten()
            # Flattening of the ROI.
            roi = roi.flatten()

            # Creating an input data vector consisting of the concatenated flattened image and its FFT
            input_data = np.concatenate((roi, fft))
            # Appending the resulting vector of the ROI and its FFT with its associated tag to the full_data list.
            # During the appending the ROI and FFT data are also normalized using a TensorFlow function.
            full_data.append(tuple([list(np.asarray(tf.math.l2_normalize(input_data))), data_array[index, 1]]))

        # After all the data for a giving class of plates has been added to the full_data list. The list is shuffled.
        shuffle(full_data)
        # Saves 70% of the data for this particular class of plates, into the training_data list
        training_data = full_data[:training_size]
        # The remaining 30% of the data for this class is saved to the validation_data list
        validation_data = full_data[training_size:]

        # Depending on which class of plates was processed previously, the training and Validation data are saved to
        # their respective lists.
        if 'Plate1' in file_path:
            plate1_data_train = training_data
            plate1_data_validate = validation_data
        elif 'Plate2' in file_path:
            plate2_data_train = training_data
            plate2_data_validate = validation_data
        elif 'Plate3' in file_path:
            plate3_data_train = training_data
            plate3_data_validate = validation_data

    # The data from all the classes of plates are saved in a single training data list and a single validation data list
    input_training_data = [plate1_data_train, plate2_data_train, plate3_data_train]
    input_validation_data = [plate1_data_validate, plate2_data_validate, plate3_data_validate]

    # Saving the training data list to a JSON file
    with open(os.path.join(parent_path, 'input_training_data.json'), 'w') as f:
        json.dump(input_training_data, f, indent=4)
    f.close()

    # Saving of the validation data list to a JSON file
    with open(os.path.join(parent_path, 'input_validation_data.json'), 'w') as f:
        json.dump(input_validation_data, f, indent=4)
    f.close()


if __name__ == '__main__':
    extract_data()
