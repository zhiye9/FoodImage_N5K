from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
import argparse
#import cv2
from PIL import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def initialize_model(model_name, model_pre, feature_extract = False, use_pretrained=True):
    model_ft = model_pre
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    #num_ftrs = model_ft.AuxLogits.fc.in_features
    #model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 1)
    # Handle the primary net
    if model_name.startswith('swin'):
        num_ftrs = model_ft.head.in_features
        input_layer = []
        input_layer.append(nn.Linear(num_ftrs, 4096))
        input_layer.append(nn.Linear(4096, 4096))
        input_layer.append(nn.Linear(4096, 1))
        ffc = nn.Sequential(*input_layer)
        model_ft.head = ffc
    else:
        num_ftrs = model_ft.fc.in_features
        input_layer = [] 
        input_layer.append(nn.Linear(num_ftrs, 4096))
        input_layer.append(nn.Linear(4096, 4096))
        input_layer.append(nn.Linear(4096, 1))
        ffc = nn.Sequential(*input_layer)
        model_ft.fc = ffc
    input_size = 224
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

PATH = "/home/zhi/nas/N5K/results/results/checkpoint_150_lr001_swin_b.pth"
model_pretrained = models.__dict__['swin_b'](pretrained=True)
model_ft, input_size = initialize_model(model_name = 'swin_b', model_pre = model_pretrained)
model = model_ft
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)

image_path = "/home/zhi/nas/food_img/Nutrition5k/imagery/side_angles/"
train_path = "/home/zhi/nas/food_img/Nutrition_pred/incept_v3/df_train_all_id.txt"
test_path = "/home/zhi/nas/food_img/Nutrition_pred/incept_v3/df_test_all_id.txt"

#data_loader_all = load_data(image_path, train_path, test_path, batch_size = 1)

img_dir = 'dish_1559240441/frames_sampled60/camera_B_frame_001.jpeg'

if (os.path.isfile(image_path + img_dir)):
    imgs = image_path + img_dir

input_size = 224

def img_loader(path):
    return PIL.Image.open(path).convert('RGB')

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


img = img_loader(imgs)
img_tf = data_transforms(img)
img_tf = img_tf.unsqueeze(0).to(device)

model(img_tf)

img