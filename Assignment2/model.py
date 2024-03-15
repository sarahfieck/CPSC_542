# Model: Home to my U-Net image segmentation model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import *
import cv2
import os
import pickle5 as pickle
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy

def unet_model(train_gen, test_gen):

    # model_checkpoint_path = os.path.join(save_dir, 'best_model.h5')

    inp = Input(shape=(128, 128, 3))

    # Encoder: We want to downsample and get that hidden representation
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.3)(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    norm5 = BatchNormalization()(conv5)
    norm5 = Dropout(0.4)(norm5)
    
    # Bottleneck
    bottle = norm5

    # Decoder - Time to upsample, recreate our input and get a segmentation
    upsample6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(bottle)
    merge6 = concatenate([conv4, upsample6], axis = 3) # https://paperswithcode.com/method/concatenated-skip-connection
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    layer6 = Dropout(0.4)(conv6)

    upsample7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, upsample7], axis = 3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    layer7 = BatchNormalization()(conv7)

    upsample8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, upsample8], axis = 3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    layer8 = Dropout(0.3)(conv8)

    upsample9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, upsample9], axis = 3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    layer9 = BatchNormalization()(conv9)
    
    out = Conv2D(1, 1, activation = 'sigmoid')(layer9)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(), loss = BinaryCrossentropy(), metrics = ['accuracy', keras.metrics.IoU(num_classes = 2, target_class_ids=[1])])

    history = model.fit(train_gen, validation_data = test_gen, epochs = 100)
    model.save('unet_model.h5')

    # Pickle the behavior to file
    with open('unet_behavior.pkl', 'wb') as f:
        pickle.dump(history, f)

    return


