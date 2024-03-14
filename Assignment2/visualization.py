# Visualization: Used to visualize the model's behavior. Does this with metric plots, true vs predicted values, and activation visuals 

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import keras
from PIL import Image
from IPython.display import Image, display
import glob
import os
import matplotlib.pyplot as plt
import cv2
import pickle5 as pickle
from tensorflow.keras.models import Model
from keras.models import load_model
from sklearn.metrics import accuracy_score, jaccard_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from matplotlib import pyplot
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow.keras as kb



# Loss Metric Plot
def loss(model):
    # Reads the behavior of the training & testing process
    with open(model, 'rb') as f:
        hist = pickle.load(f)

    # Extracts the metrics: Loss Function
    loss = hist.history["loss"]
    test_loss = hist.history["val_loss"]
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
    plt.savefig('loss_plot.png') 

# Accuracy Metrics
def acc(model):
    # Reads the behavior of the training & testing process
    with open(model, 'rb') as f:
        hist = pickle.load(f)

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
    plt.savefig("accuracy_plot.png")
    

# IOU Metrics
def iou(model):
        # Reads the behavior of the training & testing process
    with open(model, 'rb') as f:
        hist = pickle.load(f)

    # IOU Time
    acc = hist.history["io_u"]
    test_acc = hist.history["val_io_u"]
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, "b", label = "Training Intersection over Union")
    plt.plot(epochs, test_acc, "r", label = "Validation Intersection over Union")
    plt.title("IOU History")
    plt.xlabel("Epochs")
    plt.ylabel("IOU")
    plt.legend()
    plt.show()
    plt.savefig("iou_plot.png")

def true_vs_pred(testp, truemsk, model):
    test = cv2.imread(testp)
    true_mask = cv2.imread(truemsk, cv2.IMREAD_GRAYSCALE)
    
    # UNet Model
    model = tf.keras.models.load_model(model)
    
    # Generate Predictions
    res = cv2.resize(test, (128, 128))
    inp = np.expand_dims(res, axis=0)
    inp = inp / 255.0  # Normalize
    predmsk = model.predict(inp)
    predmsk = np.argmax(predmsk, axis=-1)
    predmsk_binary = tf.cast(predmsk > 0.5, dtype=tf.float32) # Since we are making a binary model

    # Plotting the three visuals together to compare
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(test)
    axs[0].set_title('Test Image')
    axs[1].imshow(true_mask, cmap='gray')
    axs[1].set_title('Actual Mask')
    axs[2].imshow(predmsk_binary[0], cmap='gray')
    axs[2].set_title('Predict Mask')

    plt.savefig('seg.png')

# Convolutional Layer Activation Visuals
def conv_layers(model, img):
    # Loading in vars
    model = tf.keras.models.load_model(model)
    img = load_img(img, target_size=(128, 128))

    # Grab layer stuff
    layer_out = []
    layer_name = []

    # Collecting layer names
    for layer in model.layers:
        if isinstance(layer, (kb.layers.Conv2D, kb.layers.MaxPooling2D)):
            layer_out.append(layer.output)
            layer_name.append(layer.name)

    layer_activations = kb.Model(inputs = model.input, outputs = layer_out)
    layer_activations.summary()

    img = kb.utils.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    activations = layer_activations.predict(img)

    # Looking at the Encoder Activation -> Individually

    # # The last layer
    # layer = 9
    # channel = 3
    # layer_act = activations[layer]
    # plt.imshow(layer_act[0,:,:,channel], cmap = "viridis")
    # plt.savefig("layer_9.png")

    # # The first layer
    # layer = 1
    # channel = 3
    # layer_act = activations[layer]
    # plt.imshow(layer_act[0,:,:,channel], cmap = "viridis")
    # plt.savefig("layer_1.png")

    # # Halfway
    # layer = 4
    # channel = 3
    # layer_act = activations[layer]
    # plt.imshow(layer_act[0,:,:,channel], cmap = "viridis")
    # plt.savefig("layer_4.png")


    # # Looking at the decoder?

    # # First layer
    # layer = 20
    # channel = 3
    # layer_act = activations[layer]
    # plt.imshow(layer_act[0,:,:,channel], cmap = "viridis")
    # plt.savefig("layer_20.png")

    # # Last layer
    # layer = 24
    # channel = 3
    # layer_act = activations[layer]
    # plt.imshow(layer_act[0,:,:,channel], cmap = "viridis")
    # plt.savefig("layer_24.png")

    # I want to see all of them, there are 18 conv layers in my model. Let's see how they activate an image
    conv_count = 18
    nrow = 6
    ncol = 3
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 15))

    for layer in range(1, conv_count + 1):  
        row = (layer - 1) // ncol
        col = (layer - 1) % ncol

        # Plotting the activation
        layer_act = activations[layer]
        axes[row, col].imshow(layer_act[0, :, :, 0], cmap="viridis")
        axes[row, col].set_title(f'Layer {layer}')

    # Layout & save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top margin
    plt.suptitle("Activation of Each Convolutional Layer in U-Net",fontsize=16)
    plt.savefig("activate.png")
    


