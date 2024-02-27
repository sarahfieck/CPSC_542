# This portion includes the Convolutional Neural Network Model
# It includes all architecture, adjustments to layers, and metrics

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import keras
from PIL import Image
import glob
import os

import tensorflow as tf
import keras
import tensorflow.keras as kb
from tensorflow.keras import backend
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from pickle import dump
from tensorflow.keras.utils import to_categorical

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import pickle

# Model Implementation
def cnn_model(train_ds,test_ds):
    
    # Regular VGG16 Model: https://keras.io/api/applications/vgg/
    model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        classes = 8,
        classifier_activation="relu"
    )

    # Freezes layers for transfer learning
    for layer in model.layers:
        layer.trainable = False

    # Regularization
    x = Dropout(0.4)(model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)

    # Flattening & Prediction Layer
    x = Flatten()(x)
    predictions = Dense(8, activation='softmax')(x)  
    model = Model(inputs=model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
    model.summary()

    # Fitting the Model
    history = model.fit(
        train_ds,
        epochs = 40,
        validation_data = test_ds
        )
    
    # Save the model as an .h5 to use later
    model.save('assignment1_model4.h5')
    
    # Pickle the history to file
    with open('file_name4.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Save the model architecture as an image 
    tf.keras.utils.plot_model(model, to_file='model4.png', show_shapes=True, show_layer_names=True)

