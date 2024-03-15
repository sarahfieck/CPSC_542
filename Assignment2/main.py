# Where the magic happens

from data_prep import *
from data_aug import *
from model import *
from visualization import *
import warnings
warnings.filterwarnings('ignore')


# Splitting Dataset from Data Prep

images = "./supervisely_person_clean_2667_img/images"
masks = "./supervisely_person_clean_2667_img/masks"
train = "./train"
test = "./test"
# split_data(images, masks, train, test)

new = "./n_dir"
new_test_mask = "./n_dir/test/mask_test"
new_test_image = "./n_dir/test/image_test"
new_train_image = "./n_dir/train/image_test"
new_train_mask = "./n_dir/train/mask_test"

old_test_mask = "./old_dir/test/mask_test"
old_test_image = "./old_dir/test/image_test"
old_train_image = "./old_dir/train/image_train"
old_train_mask= "./old_dir/train/mask_train"

newest_test_mask = "./nu_dir/test/mask_test"
newest_test_image = "./nu_dir/test/image_test"
newest_train_image = "./nu_dir/train/image_train"
newest_train_mask = "./nu_dir/train/mask_train"

# Change file type -> Maybe do not run this
# to_jpg(old_test_mask, new_test_mask)
# to_jpg(old_train_mask, new_train_mask)
# to_jpg(old_test_image, new_test_image)
# to_jpg(old_train_image, new_train_image)

# Adjust size -> Maybe do not run this
# adjust_size(new_test_mask, newest_test_mask)
# adjust_size(new_train_mask, newest_train_mask)
# adjust_size(new_test_image, newest_test_image)
# adjust_size(new_train_image, newest_train_image)

# Data Augmentation
image = (128, 128)
batch = 32
# weights = {0: 1, 1: 3} <- The weights if I could use them......

train_gen = SegGenerator(newest_train_image, newest_train_mask, batch, image, augment = True)
test_gen = SegGenerator(newest_test_image, newest_test_mask, batch, image, augment = False)

# Establish U-Net Model
unet_model(train_gen, test_gen)

# Metrics & Evaluation Visuals

test_image_path = './nu_dir/test/image_test/0.png'
true_mask_path = './nu_dir/test/mask_test/0.png' 
model_path = './unet_model.h5'

# True vs Predicted Mask
true_vs_pred(test_image_path, true_mask_path, model_path)

# Metrics Plots
loss("./unet_behavior.pkl")
acc("./unet_behavior.pkl")
iou("./unet_behavior.pkl")

# Activation Visuals
conv_layers(model_path,test_image_path)
