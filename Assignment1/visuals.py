# This file handles the visuals: GradCAM, metrics plot, and possibly the confusion matrix (if i can get it to work lol)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import keras
from PIL import Image
from IPython.display import Image, display
import glob
import os

import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras import backend
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.text import Tokenizer
# from pickle import dump
from tensorflow.keras.utils import to_categorical
# from ggplot import ggplot <- Problem child.


os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import shutil

from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

# GradCAM Heatmap Visual Generator: Heavily inspired by https://keras.io/examples/vision/grad_cam/
def grad_cam(img_file):
    model = tf.keras.models.load_model('./assignment1_model4.h5')
    model.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    builder = keras.applications.xception.Xception
    size = (224, 224)
    last_layer = "block5_conv3"
    img = img_file

    # Resizes the image array, turning it into a batch
    def res_array(path, sz):
        img = keras.utils.load_img(path, target_size = sz)
        arr = keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return arr

    # Maps input image to activations
    def heatmap(array, md, last_layer, pred_index = None):
        grad = keras.models.Model(
            md.inputs, [md.get_layer(last_layer).output, md.output]
        )

        # Compute gradient of the top predicted class
        with tf.GradientTape() as tape:
            last_layer, preds = grad(array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            channel = preds[:, pred_index]

        # Vector of mean intensity of the gradient
        grad = tape.gradient(channel, last_layer)
        p_grads = tf.reduce_mean(grad, axis=(0, 1, 2))

        # Multiply each channel in the feature map array & normalize
        last_layer_z = last_layer[0]
        heat = last_layer_z @ p_grads[..., tf.newaxis]
        heat = tf.squeeze(heat)
        heatmap = tf.maximum(heat, 0) / tf.math.reduce_max(heat)
        return heatmap.numpy()

    # Remove last layer's softmax & generate heat
    model.layers[-1].activation = None
    arr = res_array(img_file, size)
    heatmap = heatmap(arr, model, last_layer)
    plt.imshow(heatmap, cmap='viridis')
    plt.show()

    # Save & Export
    path = "test.jpg"
    def save_finalize(img, heat, cam_path = path, alpha = 0.4):
        # Load the original image
        img = keras.utils.load_img(img)
        img = keras.utils.img_to_array(img)

        # Colorizing the heatmap
        heat = np.uint8(255 * heat)
        color = mpl.colormaps["jet"]
        color = color(np.arange(256))[:, :3]
        color_heatmap = color[heat]
        color_heatmap = keras.utils.array_to_img(color_heatmap)
        color_heatmap = color_heatmap.resize((img.shape[1], img.shape[0]))
        color_heatmap = keras.utils.img_to_array(color_heatmap)

        # Combine heatmap & source image
        disp = color_heatmap * alpha + img
        disp = keras.utils.array_to_img(disp)

        # Save the superimposed image
        disp.save(cam_path)

    save_finalize(img, heatmap)

# End of grad_cam & heavy article sourcing

# Metrics Plots: Loss function & Accuracy
def metrics():

    import pickle
    history = './file_name4.pkl'
    # Reads the behavior of the training & testing process
    with open(history, 'rb') as f:
        hist = pickle.load(f)

    # Extracts the metrics: Loss Function
    loss = hist.history["loss"]
    test_loss = hist.history["test_loss"]
    epochs = range(len(loss))

    # Creates & Saves the plot
    plt.figure()
    plt.plot(epochs, loss, "b", label = "Training loss")
    plt.plot(epochs, test_loss, "r", label= "Testing loss")
    plt.title("Loss Function History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('loss4.png')

    # Accuracy Time
    acc = hist.history["accuracy"]
    test_acc = hist.history["val_accuracy"]
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, "b", label = "Training Accuracy")
    plt.plot(epochs, test_acc, "r", label = "Validation Accuracy")
    plt.title("Accuracy History")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("acc4.png")

# DOES NOT WORK: I wanted to make a confusion matrix? https://stackoverflow.com/questions/64910516/plot-confusion-matrix-from-cnn-model
# def conf(model, x, y):
#     y_pred = model.predict(x)
#     y_true = y
#     res = tf.math.confusion_matrix(y_true,y_pred)
    
# Class Distribution Visual
def class_dist(train, test):
    # Puts class & count in a dictionary to be put in a dataset later
    def count_class(ds):
        classify = os.listdir(ds)
        count = {}
        for class_name in classify:
            dir = os.path.join(ds, class_name)
            if os.path.isdir(dir):
                images = os.listdir(dir)
                count[class_name] = len(images)
        return count

    # Puts it in a dataset
    train_dist = count_class(train)
    train_metric = pd.DataFrame(train_dist.items(), columns=['Flower','Count'])
    train_metric['Dataset'] = 'Train'
    # print(train_metric)

    test_dist = count_class(test)
    test_metric = pd.DataFrame(test_dist.items(), columns=['Flower','Count'])
    test_metric['Dataset'] = 'Test'
    # print(test_metric)

    vis_df = pd.concat([train_metric, test_metric])

    # Visual of Class Distribution
    # For some reason, this portion of my code won't run locally. ggplot was not working?
    # I was able to do it on Colab and obtain a visual

    # (ggplot(vis_df, aes(x = 'Flower', y = 'Count', fill = 'Dataset')) 
    #     + geom_bar(stat = 'identity')
    #     + labs(title = "Distribution of Flower Images in Train & Test Directories")
    #     + theme_minimal())