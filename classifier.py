import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
import numpy as np
import random

"""Prepares and executes classification of power quality disturbance signals by
using a one-dimensional convolutional neural network model."""
class classifier:
    def __init__(self, sf=2048, tp=1024, nc=2, cpc=800):
        self.sf = sf # Sampling frequency
        self.tp = tp # Signal time points
        self.nc = nc # Number of categories
        self.cpc = cpc # Number of cases per category
        fn = 100 # Number of features.
        ks = 34 # Kernel size for each feature.
        self.create_model(tp, nc, fn, ks)
        self.train(tp, nc, cpc, sf)

    def train(self, tp, nc, cpc, sf):
        """Creates the one-dimensional neural network structure and trains it
        by creating synthetic signals with isolated and mixed power quality
        disturbances."""
        self.create_train_dataset(tp, nc, cpc, sf)
        self.model.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.train_dataset_x,
            self.train_dataset_y,
            batch_size=60,
            epochs=20,
            validation_split=0.2)


    def create_train_dataset(self, tp, nc, cpc, sf):
        """Creates the signals dataset for training the model.

        This dataset contains isolated and mixed power quality disturbances. It
        contains ``cpc`` cases for each category."""
        # Create resulting matrix
        self.train_dataset_x = np.zeros((cpc*nc, tp))
        self.train_dataset_y = np.zeros((cpc*nc, 1))
        time = np.array([k/sf for k in range(tp)])
        # Fill for category 0 (Sane signal)
        for k in range(cpc):
            self.train_dataset_x[k, :] = random.uniform(0.8, 1.0)* \
                np.sin(2.0*np.pi*60.0*time)
            self.train_dataset_y[k] = 0
        # Fill for category 1 (Sag signal)
        for k in range(cpc):
            self.train_dataset_x[cpc+k, :] = np.sin(2.0*np.pi*60.0*time)
            st = random.randrange(0, 600)
            self.train_dataset_x[cpc+k, st:st+300] = \
                random.uniform(0.2, 0.8)*self.train_dataset_x[cpc+k, st:st+300]
            self.train_dataset_y[cpc+k] = 1
        self.train_dataset_x = np.expand_dims(self.train_dataset_x, axis=2)

    def create_model(self, tp, nc, fn, ks):
        """Creates the convolutional neural network model.

        This model is the same as described by S. Wang (2019)."""
        self.model = Sequential()
        self.model.add(Reshape((tp, 1), input_shape=(tp,1)))
        self.model.add(Conv1D(fn, ks, activation='relu', input_shape=(tp,)))
        self.model.add(Conv1D(fn, ks, activation='relu'))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(160, ks, activation='relu', input_shape=(tp,)))
        self.model.add(Conv1D(160, ks, activation='relu'))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nc, activation='softmax'))

    def eval(self, signal):
        pass
