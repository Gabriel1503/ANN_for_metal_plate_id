"""
    This file creates and trains all the Neural Network models designed for the assignment. It also saves the data
    needed for graphing in separate CSV files.
"""

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle
from time import time

from auxiliary.data_extraction import extract_data
from plot_training_progression import plot_training_progression
from auxiliary.test_load_model import test_load_model

# extract the data from labelled images
extract_data()

# variable to store the location of the current script
parent_path = os.getcwd()

# Opens the JSON file where the prepared training data was saved, and saves it to a local python variable
file = open('input_training_data.json')
training_data = json.load(file)
file.close()

# Opens the JSON file where the prepared validation data was saved, and saves it to a local python variable
file = open('input_validation_data.json')
validation_data = json.load(file)
file.close()

# Shuffling of both the training and validation datas
shuffle(validation_data)
shuffle(training_data)

# Initialization of lists that will contain the training and validation data of the images, and lists that will contain
# the training and validation ground truths (image tags).
x_train_lst = []
y_train_lst = []
x_test_lst = []
y_test_lst = []

# Iteration over the whole training data list
for i in range(len(training_data)):
    # Iteration over the lists containing the data for each plate class
    for j in range(len(training_data[i])):
        # Saving of the image data and its ground truth to the appropriate lists
        x_train_lst.append(training_data[i][j][0])
        y_train_lst.append(training_data[i][j][1])

# Iteration over the whole validation data list
for i in range(len(validation_data)):
    # Iteration over the lists containing the data for each plate class
    for j in range(len(validation_data[i])):
        # Saving of the image data and its ground truth to the appropriate lists
        x_test_lst.append(validation_data[i][j][0])
        y_test_lst.append(validation_data[i][j][1])

# Conversion of all the lists into NumPy arrays
x_train = np.array(x_train_lst).astype('float32')
y_train = np.array(y_train_lst).astype('float32')
x_test = np.array(x_test_lst).astype('float32')
y_test = np.array(y_test_lst).astype('float32')

# Calculation of input size for the input layer of the neural networks
input_size = x_train[0].size

# Save path of current working directory to a variable.
path = os.getcwd()

r"""#################################### START MODEL1 ###############################################################"""
# Making input layer
# -------------------------------------------------------------------
input_layer1 = tf.keras.layers.Input(shape=(input_size,))
# Hidden layers
# -------------------------------------------------------------------
x = tf.keras.layers.Dense(units=50, activation="relu")(input_layer1)
x = tf.keras.layers.Dense(units=30, activation="relu")(x)
# layer for adding complexity
x = tf.keras.layers.Dense(units=25, activation="sigmoid")(x)
x = tf.keras.layers.Dense(units=12, activation="relu")(x)
# layer for adding complexity
x = tf.keras.layers.Dense(units=5, activation="sigmoid")(x)
# -------------------------------------------------------------------
# Output Layer
output1 = tf.keras.layers.Dense(units=3, activation="softmax")(x)
# Generates the NN model
model1 = tf.keras.Model(inputs=input_layer1, outputs=output1)
model1._name = "model_1"

r""" ************************ FIRST TRAINING *************************** """
# Path and name declaration of a file that will contain the loss values of the first model as the training progresses
file_loss_values = f'{path}\\loss_accuracy.csv'
# Creation of a checkpoints folder for model1. The model is saves as a '.keras' file
checkpoint_path = "{}\\checkpoints".format(path) + "\\cp-{epoch:04d}.keras"
# Creation of a CSV logger to save the values if the loss and accuracy
history_logger = tf.keras.callbacks.CSVLogger(file_loss_values, separator=",", append=False)
# Creation of the callback that will be saving a model at the end of each epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch')
# Compilation of the model
model1.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=["Accuracy"])
r"""#################################### END MODEL1 #################################################################"""

# ----------------------------------------------------------------------------------------------------------------------

r"""#################################### START MODEL2 ###############################################################"""
# Making input layer
# -------------------------------------------------------------------
input_layer2 = tf.keras.layers.Input(shape=(input_size,))
# Hidden layers
# -------------------------------------------------------------------
x = tf.keras.layers.Dense(units=50, activation="relu")(input_layer2)
# layer for adding complexity
x = tf.keras.layers.Dense(units=12, activation="relu")(x)
# layer for adding complexity
x = tf.keras.layers.Dense(units=5, activation="relu")(x)
# -------------------------------------------------------------------
# Output Layer
output2 = tf.keras.layers.Dense(units=3, activation="softmax")(x)
# Generates the NN model
model2 = tf.keras.Model(inputs=input_layer2, outputs=output2)
model2._name = "model_2"

r""" ************************ FIRST TRAINING *************************** """
# Path and name declaration of a file that will contain the loss values of model2 as the training progresses
file_loss_values_2 = f'{path}\\loss_accuracy_2.csv'
# Creation of a checkpoints folder for model2. The model is saves as a '.keras' file
checkpoint_path_2 = "{}\\checkpoints_2".format(path) + "\\cp-{epoch:04d}.keras"
# Creation of a CSV logger to save the values if the loss and accuracy
history_logger_2 = tf.keras.callbacks.CSVLogger(file_loss_values_2, separator=",", append=False)
# Creation of the callback that will be saving a model at the end of each epoch
cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2, save_freq='epoch')
# Compilation of the model
model2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=["Accuracy"])
r"""#################################### END MODEL2 #################################################################"""

# ----------------------------------------------------------------------------------------------------------------------

r"""#################################### START MODEL3 ###############################################################"""
# Making input layer
# -------------------------------------------------------------------
input_layer3 = tf.keras.layers.Input(shape=(input_size,))
# Hidden layers
# -------------------------------------------------------------------
x = tf.keras.layers.Dense(units=30, activation="sigmoid")(input_layer3)
# layer for adding complexity
x = tf.keras.layers.Dense(units=25, activation="sigmoid")(x)
x = tf.keras.layers.Dense(units=12, activation="sigmoid")(x)
# layer for adding complexity
x = tf.keras.layers.Dense(units=5, activation="sigmoid")(x)
# -------------------------------------------------------------------
# Output Layer
output3 = tf.keras.layers.Dense(units=3, activation="softmax")(x)
# Generates the NN model
model3 = tf.keras.Model(inputs=input_layer3, outputs=output3)
model3._name = "model_3"

r""" ************************ FIRST TRAINING *************************** """
# Path and name declaration of a file that will contain the loss values of model3 as the training progresses
file_loss_values_3 = f'{path}\\loss_accuracy_3.csv'
# Creation of a checkpoints folder for model3. The model is saves as a '.keras' file
checkpoint_path_3 = "{}\\checkpoints_3".format(path) + "\\cp-{epoch:04d}.keras"
# Creation of a CSV logger to save the values if the loss and accuracy
history_logger_3 = tf.keras.callbacks.CSVLogger(file_loss_values_3, separator=",", append=False)
# Creation of the callback that will be saving a model at the end of each epoch
cp_callback_3 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_3, save_freq='epoch')
# Compilation of the model
model3.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=["Accuracy"])
r"""#################################### END MODEL3 #################################################################"""

# ----------------------------------------------------------------------------------------------------------------------

r"""#################################### START MODEL4 ###############################################################"""
# Making input layer
# -------------------------------------------------------------------
input_layer4 = tf.keras.layers.Input(shape=(input_size,))
# Hidden layers
# -------------------------------------------------------------------
x = tf.keras.layers.Dense(units=50, activation="relu")(input_layer4)
# layer for adding complexity
x = tf.keras.layers.Dense(units=25, activation="sigmoid")(x)
x = tf.keras.layers.Dense(units=12, activation="relu")(x)
# -------------------------------------------------------------------
# Output Layer
output4 = tf.keras.layers.Dense(units=3, activation="softmax")(x)
# Generates the NN model
model4 = tf.keras.Model(inputs=input_layer4, outputs=output4)
model4._name = "model_4"

r""" ************************ FIRST TRAINING *************************** """
# Path and name declaration of a file that will contain the loss values of model4 as the training progresses
file_loss_values_4 = f'{path}\\loss_accuracy_4.csv'
# Creation of a checkpoints folder for model4. The model is saves as a '.keras' file
checkpoint_path_4 = "{}\\checkpoints_4".format(path) + "\\cp-{epoch:04d}.keras"
# Creation of a CSV logger to save the values if the loss and accuracy
history_logger_4 = tf.keras.callbacks.CSVLogger(file_loss_values_4, separator=",", append=False)
# Creation of the callback that will be saving a model at the end of each epoch
cp_callback_4 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_4, save_freq=10)
# Compilation of the model
model4.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=["Accuracy"])
r"""#################################### END MODEL4 #################################################################"""

# ----------------------------------------------------------------------------------------------------------------------

r"""#################################### START MODEL5 ###############################################################"""
# Making input layer
# -------------------------------------------------------------------
input_layer5 = tf.keras.layers.Input(shape=(input_size,))
# Hidden layers
# -------------------------------------------------------------------
x = tf.keras.layers.Dense(units=50, activation="relu")(input_layer5)
# layer for adding complexity
x = tf.keras.layers.Dense(units=5, activation="sigmoid")(x)
# -------------------------------------------------------------------
# Output Layer
output5 = tf.keras.layers.Dense(units=3, activation="softmax")(x)
# Generates the NN model
model5 = tf.keras.Model(inputs=input_layer5, outputs=output5)
model5._name = "model_5"

r""" ************************ FIRST TRAINING *************************** """
# Path and name declaration of a file that will contain the loss values of model5
file_loss_values_5 = f'{path}\\loss_accuracy_5.csv'
# Creation of a checkpoints folder for model5. The model is saves as a '.keras' file
checkpoint_path_5 = "{}\\checkpoints_5".format(path) + "\\cp-{epoch:04d}.keras"
# Creation of a CSV logger to save the values if the loss and accuracy
history_logger_5 = tf.keras.callbacks.CSVLogger(file_loss_values_5, separator=",", append=False)
# Creation of the callback that will be saving a model at the end of each epoch
cp_callback_5 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_5, save_freq='epoch')
# Compilation of the model
model5.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=["Accuracy"])
r"""#################################### END MODEL5 #################################################################"""

# ----------------------------------------------------------------------------------------------------------------------

# Parameter with the number of epochs that each model will pass trhough for training
epochs = 50

print("***************************************************************************************************************")
print("*******************************************TRAINING MODEL 1****************************************************")
# Time at the start of the training
start = time()
# Training of model1
model1.fit(x=x_train,
           y=y_train,
           epochs=epochs,
           validation_data=(x_test,
                            y_test),
           callbacks=[history_logger,
                      cp_callback])

print("\n"
      "***************************************************************************************************************")
# By subtracting the start time from the time the training ends we have the time that it took to train the model.
print(
    f"*****************************************TIME TO TRAIN: {time() - start}****************************************")

print("*******************************************TRAINING MODEL 2****************************************************")
# Time at the start of the training
start = time()
# Training of model2
model2.fit(x=x_train,
           y=y_train,
           epochs=epochs,
           validation_data=(x_test,
                            y_test),
           callbacks=[history_logger_2,
                      cp_callback_2])

print("\n"
      "***************************************************************************************************************")
# By subtracting the start time from the time the training ends we have the time that it took to train the model.
print(
    f"*****************************************TIME TO TRAIN: {time() - start}****************************************")

print("*******************************************TRAINING MODEL 3****************************************************")
# Time at the start of the training
start = time()
# Training model3
model3.fit(x=x_train,
           y=y_train,
           epochs=epochs,
           validation_data=(x_test,
                            y_test),
           callbacks=[history_logger_3,
                      cp_callback_3])

print("\n"
      "***************************************************************************************************************")
# By subtracting the start time from the time the training ends we have the time that it took to train the model.
print(
    f"*****************************************TIME TO TRAIN: {time() - start}****************************************")

print("*******************************************TRAINING MODEL 4****************************************************")
# Time at the start of the training
start = time()
# Training model4
model4.fit(x=x_train,
           y=y_train,
           epochs=epochs,
           validation_data=(x_test,
                            y_test),
           callbacks=[history_logger_4,
                      cp_callback_4])
print("\n"
      "***************************************************************************************************************")
# By subtracting the start time from the time the training ends we have the time that it took to train the model.
print(
    f"*****************************************TIME TO TRAIN: {time() - start}****************************************")

print("*******************************************TRAINING MODEL 5****************************************************")
# Time at the start of the training
start = time()
# Training model 5
model5.fit(x=x_train,
           y=y_train,
           epochs=epochs,
           validation_data=(x_test,
                            y_test),
           callbacks=[history_logger_5,
                      cp_callback_5])
print(
    f"*****************************************TIME TO TRAIN: {time() - start}****************************************")
# By subtracting the start time from the time the training ends we have the time that it took to train the model.
print("***************************************************************************************************************")

# Creation of a list containing the file path for the checkpoints of each of the models
lst_ckp_path = [f'{path}\\checkpoints',
                f'{path}\\checkpoints_2',
                f'{path}\\checkpoints_3',
                f'{path}\\checkpoints_4',
                f'{path}\\checkpoints_5']

# Initialization of lists for saving one weight, and its gradient, per layer for each of the models.
weights_1 = []
weights_2 = []
weights_3 = []
weights_4 = []
weights_5 = []

# Initializing parameters which will save the number of layers that each model has (not counting input layer).
num_layer_model1, num_layer_model2, num_layer_model3, num_layer_model4, num_layer_model5 = 0, 0, 0, 0, 0

# Iteration over the checkpoints folders.
for path in lst_ckp_path:
    # Variable used for iterating over the models inside each checkpoint
    k = 1
    # Print statement so we know which model we are processing. It is given from the checkpoint path
    print(path)
    for file in os.listdir(path):
        # Print statement to inform which version of the model inside the checkpoint path is being processed
        print(file)
        # Loading of the model as a temporary model
        temp_model = tf.keras.models.load_model(f'{path}\\cp-{k:04d}.keras')
        # Creation of a gradient tape which will be later used for calculation of gradients
        with tf.GradientTape() as tape:
            # Output of the model when the training data is passed as input
            output = temp_model(x_train)
            # Definition of the loss function used to evaluate the model
            loss = tf.keras.losses.categorical_crossentropy(y_true=y_train[:, 0], y_pred=output[:, 0])
        # Calculation of the gradient of the loss with respect to weights and biases of the model
        grads = list(tape.gradient(loss, temp_model.trainable_weights))
        # Auxiliary variable whic will be used to get the gradient of only the weights and not the biases.
        j = 0
        # Iteration over the layers of the loaded model
        for i in range(1, len(temp_model.layers)):
            # Creation of a list containing only the weights of the model for the ith layer
            weights = list(temp_model.layers[i].get_weights()[0])
            # Decision tree to decide which list of weights should receive the weights and gradients extracted. This is
            # decided based on which folder the model is coming from
            if 'checkpoints_5' in path:
                # Appends a list containing the layer number, the epoch, one particular weight and its gradient
                weights_5.append([i, k, weights[3][1], float(grads[j][3][1])])
                # Saves the number of layers of the model
                num_layer_model5 = len(temp_model.layers)
            elif 'checkpoints_4' in path:
                # Appends a list containing the layer number, the epoch, one particular weight and its gradient
                weights_4.append([i, k, weights[3][1], float(grads[j][3][1])])
                # Saves the number of layers of the model
                num_layer_model4 = len(temp_model.layers)
            elif 'checkpoints_3' in path:
                # Appends a list containing the layer number, the epoch, one particular weight and its gradient
                weights_3.append([i, k, weights[3][1], float(grads[j][3][1])])
                # Saves the number of layers of the model
                num_layer_model3 = len(temp_model.layers)
            elif 'checkpoints_2' in path:
                # Appends a list containing the layer number, the epoch, one particular weight and its gradient
                weights_2.append([i, k, weights[3][1], float(grads[j][3][1])])
                # Saves the number of layers of the model
                num_layer_model2 = len(temp_model.layers)
            elif 'checkpoints' in path:
                # Appends a list containing the layer number, the epoch, one particular weight and its gradient
                weights_1.append([i, k, weights[3][1], float(grads[j][3][1])])
                # Saves the number of layers of the model
                num_layer_model1 = len(temp_model.layers)
            # Increments j by 2 as the information about weights are in the even indices of the grads list.
            j += 2
        # increments k by one to extract the information from next model (epoch)
        k += 1

"""
    The following nested loop for the creation of a data frame and exporting it to csv is repeated five times. One for
    each model. They are not made all in a single loop because the outer loop as a different range for each model. The
    comments will be added only for the first loop as the logic is repeated for all the others, except by the range.
"""
# This loop iterates over the number of layers of the given model
for layer in range(1, num_layer_model1):
    # Creation of temporary empty list
    lst = []
    # Iteration over all elements in the weights list
    for element in weights_1:
        # Checking that the current element belongs to a particular layer
        if element[0] == layer:
            # In the case the element belongs to the current layer number, we append it to the empty list
            lst.append(element)
    # After all elements of a single layer have been appended to the list a dataframe is created
    df = pd.DataFrame(lst)
    # Renaming of the columns of the data frame
    df.columns = ['Layer', 'Epoch', 'Weight', 'Gradient_W']
    # Saving the data frame as a csv file containing the progression of one weight and its gradient. One csv file is
    # created per layer per model.
    os.makedirs(os.path.join(parent_path, f'weights_per_layer'), exist_ok=True)
    df.to_csv(f'weights_per_layer\\model1_layer_{layer}.csv', index=False)

# Same as previous loop with different ranges
for layer in range(1, num_layer_model2):
    lst = []
    for element in weights_2:
        if element[0] == layer:
            lst.append(element)
    df = pd.DataFrame(lst)
    df.columns = ['Layer', 'Epoch', 'Weight', 'Gradient_W']

    df.to_csv(f'weights_per_layer\\model2_layer_{layer}.csv', index=False)

# Same as previous loop with different ranges
for layer in range(1, num_layer_model3):
    lst = []
    for element in weights_3:
        if element[0] == layer:
            lst.append(element)
    df = pd.DataFrame(lst)
    df.columns = ['Layer', 'Epoch', 'Weight', 'Gradient_W']

    df.to_csv(f'weights_per_layer\\model3_layer_{layer}.csv', index=False)

# Same as previous loop with different ranges
for layer in range(1, num_layer_model4):
    lst = []
    for element in weights_4:
        if element[0] == layer:
            lst.append(element)
    df = pd.DataFrame(lst)
    df.columns = ['Layer', 'Epoch', 'Weight', 'Gradient_W']

    df.to_csv(f'weights_per_layer\\model4_layer_{layer}.csv', index=False)

# Same as previous loop with different ranges
for layer in range(1, num_layer_model5):
    lst = []
    for element in weights_5:
        if element[0] == layer:
            lst.append(element)
    df = pd.DataFrame(lst)
    df.columns = ['Layer', 'Epoch', 'Weight', 'Gradient_W']

    df.to_csv(f'weights_per_layer\\model5_layer_{layer}.csv', index=False)

# Testing final version of each model
test_load_model()
