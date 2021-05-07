# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rwr1WstnaagSF4xoRil84JIRmsQAhqXv
"""

## Complete your work below
## Mount Google Drive Data (If using Google Colaboratory)
try:
    from google.colab import drive
    drive.mount('/content/gdrive',force_remount=True)
except:
    print("Mounting Failed.")

# -*- coding: utf-8 -*-

# Name: test.py
# Training loop for CORnet on CIFAR-10

import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from gdrive.MyDrive.DL_Final_Project.cornet import CORnet

np.random.seed(0)
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
device = torch.device(device)

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-model','--model_name', default='CORnet_Z', type=str,
                    help='the name of the model to train')
parser.add_argument('-path','--filepath', default=None, type=int,
                    help='the path to the .pth file')
parser.add_argument('-batch_size','--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('-feedback_connections', '--feedback_connections', default={}, type=str,
                      help='whether the model has feedback connections')
parser.add_argument('-pretrained', '--pretrained', default=True, type=bool,
                        help='whether the training should be started from ImageNet pretraining or random initialization')
args = parser.parse_args()

args.batch_size = 4
args.model_name = "CORnet-S"
args.feedback_connections = {}
args.filepath = "CORnet-S_{}_15.pth"
args.pretrained = True

now = datetime.now()
date = f'{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}'
print('date: %s'%(date))

print('==> Preparing data..')

# image size = 32x32x3
# resize image to 256x256x3 to match ImageNet image size for pretrained CORnet-Z
# Modify mean and stdev to ImageNet mean and stdev
# CIFAR-10 : transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# hyperparameters based on the CORnet paper
batch_size = args.batch_size
path = args.filepath



testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# load model
print(f'model: {args.model_name}')
model = CORnet(architecture=args.model_name, pretrained=args.pretrained, feedback_connections= args.feedback_connections, n_classes=10)
model.load_state_dict(torch.load("/content/gdrive/MyDrive/DL_Final_Project/"+path))
model = model.to(device)
loss = nn.CrossEntropyLoss()

# Save the train and test loss at each epoch
# Save the train and test accuracy at each epoch
#model.eval()
test_loss_epoch = 0.
test_acc_epoch = 0.
test_total = 0

test_start = time.clock()

for k, (input_batch, label_batch) in enumerate(testloader, 0):
    #print('hi')
    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)

    output_batch = model(input_batch)
    test_loss_batch = loss(output_batch,label_batch)
    test_loss_epoch+=test_loss_batch.item()

    # calculate accuracy
    _, predicted = torch.max(output_batch.data, 1)
    test_total += label_batch.size(0)
    test_acc_epoch += (predicted.float() == label_batch.float()).sum()


print('Test loss is %f'%(test_loss_epoch/test_total))

print('Test acc is %f'%(test_acc_epoch/test_total))

print('Total time for testing: %0.2f s'%(time.clock()-test_start))

"""
CORnet-Z vanilla: 
Test loss is 0.112732
Test acc is 0.854900
Total time for testing: 40.69 s

CORnet-Z feedback:
Test loss is 0.120511
Test acc is 0.851600
Total time for testing: 135.89 s

CORnet-S feedback
Test loss is 0.069607
Test acc is 0.925700
Total time for testing: 612.72 
"""