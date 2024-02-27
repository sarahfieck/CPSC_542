# Where I run my model

from data_info import data_set
from model import cnn_model
from visuals import grad_cam, metrics, class_dist
import pickle

train, test = data_set()
model = cnn_model(train,test) # If you wanted to rerun the model

# Alternatively, you can load it in.
mod = './assignment1_model4.h5'

# Could literally be any flower file I just chose this one
img = './new_flowers/train/Daisy/test95.jpg'
behav = './file_name4.pkl'


# For some reason, the visual portion of my code won't run locally. I was able to do it on Colab and obtain a visual
class_dist('./new_flowers/train','./new_flowers/test')

grad_cam(img)
metrics()
