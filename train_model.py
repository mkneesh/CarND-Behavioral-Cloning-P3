import csv
import os
import pickle
import tensorflow as tf
import numpy as np
from scipy import misc

from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
from keras.models import Model, Sequential

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

def read_dataset(data_root):
    read_image = lambda fname : misc.imread(os.path.join(data_root, fname.strip()))
    with open(os.path.join(data_root, "driving_log.csv")) as f:
        # 'center','left','right','steering','throttle','brake','speed'
        for i, row in enumerate(csv.DictReader(f)):
            center = read_image(row['center'])
            left = read_image(row['left'])
            right = read_image(row['right'])
            if i == 1:
                print (center.shape)
                print (left.shape)
                print (right.shape)
                print(i)
            if i % 100 == 0:
                print("Processed:", i)
            # TODO: left and right as well.
            # yield (left, center, right) (row['steering'], row['throttle'], row['brake'], row['speed'])
            yield center, row['steering']
            yield left, row['steering']
            yield right, row['steering']

def load_data(data_root):
    images = []
    measurements = []
    for center_img, measurement in read_dataset(data_root):
        images.append(center_img)
        measurements.append(measurement)
    return np.array(images), np.array(measurements)

def simple_ffn(data_root="data/data"):
    X_train, y_train = load_data(data_root="data/data")
    print (X_train.shape)
    print (y_train.shape)

    print('Building model')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    print('Saving')
    model.save('model.h5')

def lenet(data_root="data/data"):
    X_train, y_train = load_data(data_root="data/data")
    print('Building model')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    print('Saving')
    model.save('lenet_model.h5')

lenet()    
