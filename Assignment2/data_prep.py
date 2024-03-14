# Data Prep is a class used to perform train/test splits, size adjustments, and file type adjustments.

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
# import keras_tuner as kt
import keras
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
import pickle5 as pickle
# split_data: Used to divide the data into training & testing directories, with subdirectories for masks and image portions.

def split_data(images_dir, masks_dir, train_dir, test_dir):
    
    # 80/20 Split
    ratio=0.8 
    seed=42
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'image_train'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'mask_train'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'image_test'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'mask_test'), exist_ok=True)
    
    # Split the data with an indicies
    img_files = os.listdir(images_dir)
    random.seed(seed)
    random.shuffle(img_files)
    split = int(len(img_files) * ratio)

    train = img_files[:split]
    test = img_files[split:]
    
    # Train Directory
    for file in train:
        shutil.copy(os.path.join(images_dir, file), os.path.join(train_dir, 'image_train', file))
        shutil.copy(os.path.join(masks_dir, file), os.path.join(train_dir, 'mask_train', file))
    
    # Test Directory
    for file in test:
        shutil.copy(os.path.join(images_dir, file), os.path.join(test_dir, 'image_test', file))
        shutil.copy(os.path.join(masks_dir, file), os.path.join(test_dir, 'mask_test', file))


# Adjusting Dimensions to 128 x 128 as png

def adjust_size(input_dir, output_dir):
    count = 0 # Count so the image and mask can match
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        image = Image.open(file_path)
        new_image = image.resize((128, 128))
        new_image.save(f'{output_dir}/{count}.png')
        count+= 1
    return


# Setting all the images as jpgs

def to_jpg(input_dir, output_dir):
    count = 0 # Adds a count so the images and masks can match
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        image = Image.open(file_path)
        image = image.convert('RGB') # Ensures color
        image.save(f'{output_dir}/{count}.jpg')
        count+= 1
    return






