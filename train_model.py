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
flags.DEFINE_string('data_root', 'data/data', "Location of the unzipped data file.")
flags.DEFINE_integer('epochs', 7, "The number of epochs.")
flags.DEFINE_bool('augmentation_flip_images', True, "If True, generate additional training data by flipping images.")
flags.DEFINE_integer('max_images', -1, "If positive, the maximum number of images to use. Used for quick iteration.")

def flip_images(img, steering):
    return np.fliplr(img), -float(steering)    

def read_dataset(data_root=FLAGS.data_root):
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
            if FLAGS.max_images > 0 and i >= FLAGS.max_images:
                return
            # TODO: left and right as well.
            # yield (left, center, right) (row['steering'], row['throttle'], row['brake'], row['speed'])
            yield center, row['steering']
            yield left, row['steering']
            yield right, row['steering']
            if FLAGS.augmentation_flip_images:
                yield flip_images(center, row['steering'])
                yield flip_images(left, row['steering'])
                yield flip_images(right, row['steering'])

def load_data():
    images = []
    measurements = []
    for center_img, measurement in read_dataset():
        images.append(center_img)
        measurements.append(measurement)
    return np.array(images), np.array(measurements)

def simple_ffn():
    X_train, y_train = load_data()
    print (X_train.shape)
    print (y_train.shape)

    print('Building model')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)
    print('Saving')
    model.save('model.h5')

def lenet():
    X_train, y_train = load_data()
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
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)
    model_name = 'lenet_aug_model.h5' if FLAGS.augmentation_flip_images else 'lenet_model.h5'
    print('Saving to:', model_name)
    model.save(model_name)

lenet()    
