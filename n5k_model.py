from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("CUDA Version:", torch.version.cuda)
import PIL
import argparse
#import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_option():
    parser = argparse.ArgumentParser('N5K Network')

    parser.add_argument('--batchsize', default=32, type=int, help="batch size for single GPU")
    parser.add_argument('--image_path', type=str, default="/home/projects/cu_10108/data/Generated/ye_food_img/N5K/", help='path to N5K dataset')
    parser.add_argument("--train_path", type=str, default="/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/df_train_all_id.txt", help='path to training list')
    parser.add_argument("--test_path", type=str, default="/home/projects/cu_10108/data/Generated/ye_food_img/incept_v3/df_test_all_id.txt", help='path to testing list')
    parser.add_argument('--output_model', type=str, help='Name of output model')
    parser.add_argument('--output_val', type=str, help='Name of output val')
    parser.add_argument('--output_train', type=str, help='Name of output train')
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--epoch", default=5, type=int, help="The number of epochs.")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet152', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet152)')
    parser.add_argument('--is_inception', dest='feature', default=False, action='store_true')
    args, unparsed = parser.parse_known_args()
    return args


def img_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=img_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(',')
            if (os.path.isfile(image_path + words[0])):
                imgs.append((words[0], float(words[1].strip())))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = img_loader
        self.image_path = image_path
    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # label = list(map(int, label))
        # print label
        # print type(label)
        #img = self.loader('/home/vipl/llh/food101_finetuning/food101_vgg/origal_data/images/'+img_name.replace("\\","/"))
        img = self.loader(self.image_path + img_name)

        # print img
        if self.transform is not None:
            img = self.transform(img)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor


def load_data(image_path, train_dir, test_dir, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_dataset = MyDataset(txt_dir=train_dir, image_path=image_path, transform=data_transforms['train'])
    test_dataset = MyDataset(txt_dir=test_dir, image_path=image_path, transform=data_transforms['val'])
#    print(len(train_dataset))
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size,  shuffle=False, num_workers=0)
    data_loader = {'train': train_loader, 'val': test_loader}
    return data_loader


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

    val_acc_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print('train')
            else:
                model.eval()   # Set model to evaluate mode
                print('eval')
            running_loss = 0.0
            #runing_loss_per = 0.0
            #running_corrects = 0

            # Iterate over data.
            i = 0
            since = time.time()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs.reshape(-1), labels)
			#loss = criterion(outputs, labels)

                   # _, preds = torch.max(outputs, 1)
                   # print(outputs.shape)
                   # print(labels.shape)
                   # loss_per = mean_absolute_percentage_error(outputs.reshape(-1), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #loaders['train'].dataset) statistics
                #print(loss.item())
                #print(inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                #runing_loss_per += loss_per.item()
                #running_corrects += torch.sum(preds == labels.data)
                i += 1
                print("process{0}".format((i+1)*100/len(dataloaders[phase])), end="\r")
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = runing_loss_per / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} (MAPE) '.format(phase, epoch_loss))
            #print('{} Loss: {:.4f} MAE per: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            #print(time_elapsed)
            print('complete in {:.0f}m {:.0f}s' .format(time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'val' and epoch_loss > best_acc:
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)
    
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Top level data directory. Here we assume the format of the directory conforms
args = parse_option()
# Batch size for training (change depending on how much memory you have)
batch_size = args.batchsize

is_inception = args.is_inception

# Number of epochs to train for
num_epochs = args.epoch

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
 
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

print("=> using pre-trained model '{}'".format(args.arch))
model_pretrained = models.__dict__[args.arch](weights='IMAGENET1K_V1')
model_ft, input_size = initialize_model(model_name = args.arch, model_pre = model_pretrained)

# Print the model we just instantiated
#print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
data_loader_all = load_data(args.image_path, args.train_path, args.test_path, batch_size = args.batchsize)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Send the model to GPU
model_ft = model_ft.to(device)
#model_ft = nn.DataParallel(model_ft)
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
feature_extract = False
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)

# Setup the loss fxn
criterion = nn.L1Loss()
#criterion = MeanAbsolutePercentageError().to(device)
# Train and evaluate
model_ft, hist_val, hist_train = train_model(model_ft, data_loader_all, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)

torch.save(model_ft.state_dict(), args.output_model)

with open(args.output_val, "wb") as fp:
	pickle.dump(hist_val, fp)
with open(args.output_train, "wb") as fp:
        pickle.dump(hist_train, fp)
