"""
    This script loads the latest version of each model and passes one sample of each image class. The output can then
    be compared with the ground truths, and the performance of the models can be compared.
"""

import json
import numpy as np
import tensorflow as tf


def test_load_model():
    # ------------------------------------------------------------------------------------------------------------------
    # Loading of the validation data. The process is the same as the one performed in the Neura_Network_model.py data
    # loading step. Except here only validation data is being loaded.
    file = open('input_validation_data.json')
    validation_data = json.load(file)
    file.close()

    x_test_lst = []
    y_test_lst = []

    # Iteration over the list containing the lists for each sample class
    for i in range(len(validation_data)):
        # From each sample list, one sample is loaded and its ground truth
        x_test_lst.append(validation_data[i][0][0])
        y_test_lst.append(validation_data[i][0][1])

    # Conversion of the image data into a NumPy array
    x_test = np.array(x_test_lst).astype('float32')

    # ------------------------------------------------------------------------------------------------------------------
    # Loading the latest models saved from the checkpoints folders.
    model1 = tf.keras.models.load_model('checkpoints/cp-0050.keras')
    model2 = tf.keras.models.load_model('checkpoints_2/cp-0050.keras')
    model3 = tf.keras.models.load_model('checkpoints_3/cp-0050.keras')
    model4 = tf.keras.models.load_model('checkpoints_4/cp-0050.keras')
    model5 = tf.keras.models.load_model('checkpoints_5/cp-0050.keras')
    # ------------------------------------------------------------------------------------------------------------------
    # Iteration over each sample in the x_test array
    for i in range(len(x_test)):
        # Prediction of output given by each of the models.
        output1 = model1.predict(np.expand_dims(x_test[i], axis=0))
        # Printing of the prediction and of the correct label.
        print(f"Model1 Output = {output1}, True Value {y_test_lst[i]}")
        output2 = model2.predict(np.expand_dims(x_test[i], axis=0))
        print(f"Model2 Output = {output2}, True Value {y_test_lst[i]}")
        output3 = model3.predict(np.expand_dims(x_test[i], axis=0))
        print(f"Model3 Output = {output3}, True Value {y_test_lst[i]}")
        output4 = model4.predict(np.expand_dims(x_test[i], axis=0))
        print(f"Model4 Output = {output4}, True Value {y_test_lst[i]}")
        output5 = model5.predict(np.expand_dims(x_test[i], axis=0))
        print(f"Model5 Output = {output5}, True Value {y_test_lst[i]}")
    # ------------------------------------------------------------------------------------------------------------------
