# model_training.py

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))

def save_model(model, model_path):
    model.save(model_path)

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    return X_train, X_test, y_train, y_test

def main():
    data_dir = 'data/'
    model_path = 'models/model.h5'

    X_train, X_test, y_train, y_test = load_data(data_dir)

    model = create_model(X_train.shape[1:])
    train_model(model, X_train, y_train, X_test, y_test)
    save_model(model, model_path)

if __name__ == '__main__':
    main()