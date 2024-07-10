"""
    NOTE: To run this script, Neural_Network_model.py has to be run first

    This script plots the progression of loss and accuracy as well as the progression of one weight and its gradient
    for each layer of all samples. An image of each input sample is also plotted at the end. The input has been reshaped
    so the picture can be seen

    WARNING: Will generate a lot of plots
"""

import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_training_progression():
    # Loading of test data, from here up to the conversion of x_test to an array is the same process as described in
    # test_load_model.py. In this case one sample of each is being loaded simply to generate a picture at the end
    file = open('input_validation_data.json')
    validation_data = json.load(file)
    file.close()

    x_test_lst = []
    y_test_lst = []

    one_sample_each_cat = []

    for i in range(len(validation_data)):
        x_test_lst.append(validation_data[i][0][0])
        y_test_lst.append(validation_data[i][0][1])

    x_test = np.array(x_test_lst).astype('float32')

    # List of the files containing the loss and accuracy information for each model.
    list_records = ['loss_accuracy.csv',
                    'loss_accuracy_2.csv',
                    'loss_accuracy_3.csv',
                    'loss_accuracy_4.csv',
                    'loss_accuracy_5.csv']

    # Iteration over the files containing loss and accuracy information
    for k, i in enumerate(list_records):
        # Reading the data from the csv
        data = pd.read_csv(i)
        # Creation of figure for plotting the data in the current file
        plt.figure("Model" + str(k + 1))
        # Name of the figure. Based on the model number which is tracked by variable k
        plt.title(f'Progression of Loss and Accuracy in Model {k + 1}')
        # Naming the x label
        plt.xlabel("Epoch")
        # Naming the y label
        plt.ylabel("Loss/Accuracy")
        # Plotting the training and validation losses and accuracies with respect to the epoch.
        plt.plot(data["epoch"], data["Accuracy"], label=f'Accuracy Model{k + 1}')
        plt.plot(data["epoch"], data["loss"], label=f'Loss Model{k + 1}')
        plt.plot(data["epoch"], data["val_Accuracy"], label=f'Val. Accuracy Model{k + 1}')
        plt.plot(data["epoch"], data["val_loss"], label=f'Val. Loss Model{k + 1}')
        # Creates a legend in the 'best' available location of the plot
        plt.legend(loc="best")

    # Iteration over all the files containing the information on a weight and its gradient for each layer and for
    # each model
    for file in os.listdir('weights_per_layer'):
        # Creating a name to be displayed in the figure
        file_name = file.replace('.csv', "")
        # Reading data of the current file
        data = pd.read_csv(f'weights_per_layer\\{file}')
        # Creation of a figure with the name file_name created above
        plt.figure(file_name)
        # Adding title to the figure that will be saved
        plt.title(f'Progression of One weight in: {file_name}')
        # Changing the name of the x label
        plt.xlabel('Epoch')
        # Changing the name of the y label
        plt.ylabel('Weight/Gradient Value')
        # Plotting the weight and the gradient in the current file. This corresponds to a single weight and its
        # gradient in a single layer, for one of the five models. This means that one figure is created for every
        # single layer in every single model.
        plt.plot(data['Epoch'], data['Weight'], label=f'Weight')
        plt.plot(data['Epoch'], data['Gradient_W'], label=f'Weight Gradient')
        # Creates a legend in the 'best' available location of the plot
        plt.legend(loc='best')

    # Iteration over the sample image list
    for i in range(len(x_test)):
        # Creating a figure for the current plate image
        plt.figure(f'Metal Plate {y_test_lst[i]}')
        # Creating a title for the plate picture
        plt.title(f'Metal Plate {y_test_lst[i]}')
        # Plotting of the image data after it has been reshaped to an image format. The shape is 200 by 100 because
        # the FFT is concatenated with the original image, so one of the dimensions is doubled in size.
        plt.imshow(x_test[i].reshape(200, 100))

    # Show all the plots created previously
    plt.show()


if __name__ == "__main__":
    plot_training_progression()
