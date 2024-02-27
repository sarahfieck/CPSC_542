# This class prioritizes any data processing and augmentation throughout the project.
# It also includes most imports needed later in the assignment.

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
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import shutil


# Loads Data Directory & Augmentation
def data_set():

    # Variables of all the directories
    n_train_dir = './new_flowers/train'
    n_train_dai = './new_flowers/train/Daisy'
    n_train_orc = './new_flowers/train/Orchid'
    n_train_ros = './new_flowers/train/Rose'
    n_train_sun = './new_flowers/train/Sunflower'
    n_train_tul = './new_flowers/train/Tulip'
    n_train_lil = './new_flowers/train/Lilly'
    n_train_lot = './new_flowers/train/Lotus'
    n_train_dan = './new_flowers/train/Dandelion'

    n_test_dir = './new_flowers/test'
    n_test_dai = './new_flowers/test/Daisy'
    n_test_orc = './new_flowers/test/Orchid'
    n_test_ros = './new_flowers/test/Rose'
    n_test_sun = './new_flowers/test/Sunflower'
    n_test_tul = './new_flowers/test/Tulip'
    n_test_lil = './new_flowers/test/Lilly'
    n_test_lot = './new_flowers/test/Lotus'
    n_test_dan = './new_flowers/test/Dandelion'
    
    # Image Data Generators for Augmetation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Data Augmentation
    train_ds = train_datagen.flow_from_directory(n_train_dir,
                                                target_size = (224, 224),
                                                batch_size= 32,
                                                class_mode = 'categorical')

    test_ds = test_datagen.flow_from_directory(n_test_dir,
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    return train_ds, test_ds





# Adjusts image size in a directory so they are all consistent
def fix_photo_size(in_dr, out_dr):
    count = 0
    os.makedirs(out_dr, exist_ok=True)
    for file in os.listdir(in_dr):
      path = os.path.join(in_dr, file)
      image = Image.open(path)
      new_image = image.resize((224, 224))
      new_image.save(f'{out_dr}/test{count}.jpg')
      count += 1

# Displays the dimensions of the image provided as the path
def get_dimensions(image_path):
   img = keras.preprocessing.image.load_img(image_path)
   img_array = keras.preprocessing.image.img_to_array(img)

   height, width, channels = img_array.shape
   print("Height:", height)
   print("Width:", width)
   print("Channels:", channels)