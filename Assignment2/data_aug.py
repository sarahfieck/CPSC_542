# Data Aug is used to augment the data for more variety in our training & testing selection

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import glob
import os
import shutil
from PIL import Image
import tensorflow as tf
import keras
import tensorflow.keras as kb
import random
import tensorflow as tf
from tensorflow.keras.models import *
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image Data Generators for Augmentation

newest_test_mask = "./nu_dir/test/mask_test"
newest_test_image = "./nu_dir/test/image_test"
newest_train_image = "./nu_dir/train/image_train"
newest_train_mask = "./nu_dir/train/mask_train"

# SegGenerator Class augments & feeds data into the model.
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class SegGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batchsz, imagesz, augment):
        self.imgp = [os.path.join(images, img) for img in os.listdir(images)]
        self.maskp = [os.path.join(masks, mask) for mask in os.listdir(masks)]
        # self.batchsz = batchsz # Batch Size
        self.imagesz = imagesz # Image Size
        self.augment = False
        # self.class_weights = class_weights  # Possible class weight?? Does not work rn

        # Augmentation details, allowing for variety in our dataset
        if self.augment:
            self.imagegen = ImageDataGenerator(
                rescale = 1./255,
                samplewise_center = True,
                samplewise_std_normalization=True,
                rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            self.maskgen = ImageDataGenerator(
                rescale=1./255,
                samplewise_center=True, 
                samplewise_std_normalization=True,
                rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )
    
    def __len__(self):
        return int(np.floor(len(self.imgp) / self.batchsz))

    def __getitem__(self, idx):
            # Augments the masks based on the batch
            def augment_masks(batch_mk, maskgen):
                batch_mk2 = np.expand_dims(batch_mk, axis=-1)
                augment_mk = np.array([maskgen.random_transform(mask) for mask in batch_mk2])
                augment_mk_squeezed = np.squeeze(augment_mk, axis=-1)
                return augment_mk_squeezed

            batch_path = self.imgp[idx * self.batchsz:(idx + 1) * self.batchsz]
            batch_mk2 = self.maskp[idx * self.batchsz:(idx + 1) * self.batchsz]
            batch_img = np.array([cv2.resize(cv2.imread(file_path), self.imagesz) for file_path in batch_path])
            batch_mk = np.array([cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), self.imagesz) for file_path in batch_mk2])
            # sample_weights = np.take(np.array(self.class_weights), np.round(y[:, :, 1]).astype('int'))  # Potentially adds weights? Does not work rn
            # print("Augmentation debug")

            # Augmentation: Image & Masks
            if self.augment:
                batch_img = np.array([self.imagegen.random_transform(img) for img in batch_img])
                batch_mk = augment_masks(batch_mk, self.maskgen)
            
            batch_mk = np.expand_dims(batch_mk, axis=-1)

            # Normalization so the metrics are not scary
            batch_img = batch_img.astype(np.float32) / 255.0
            batch_mk = batch_mk.astype(np.float32) / 255.0

            # print("Augmentation complete")
            return batch_img, batch_mk
        # Would include the weights as a return value if it worked :(
