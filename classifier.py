import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.python.keras as keras
import keras

TRAIN_DATASET_PATH = "train_dataset.json"
TEST_DATASET_PATH = "test_dataset.json"

def load_data(datast_path):
    with open(datast_path,"r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])
    return X,Y

def prepare_dataset(validation_size):
    X,Y = load_data(TRAIN_DATASET_PATH)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train,X_validation,Y_train,Y_validation

def build_model(input_shape):

    #create model
    model = keras.Sequential()

    #1st con layer(num_of_kernels,size_of_kernel,activation_function)
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    #2nd con layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #3rd con layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #flatten the output to a 1D array
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(24,activation='softmax'))

    return model

if __name__ == "__main__":
    X_train,X_validation,Y_Train,Y_validation = prepare_dataset(0.2)
    X_test,Y_test = load_data(TEST_DATASET_PATH)

    #Building the cnn
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model = build_model(input_shape)

    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    #train the model
    model.fit(X_train,Y_Train,validation_data=(X_validation,Y_validation),batch_size=32,epochs=30)

    #evaluate the CNN on the test set
    test_error,test_accuracy = model.evaluate(X_test,Y_test,verbose=1)
    print("Accuracy on test set : {}".format(test_accuracy))