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
from cornet import CORnet

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

args = parser.parse_args()

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


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# load model
print(f'model: {args.model_name}')
model = CORnet(pretrained=True, architecture=args.model_name, feedback_connections='all', n_classes=10)
model.load_state_dict(torch.load(path))
model = model.to(device)
loss = nn.CrossEntropyLoss()

# ============Sanity check=============

def imshow(img):
    img = img / 2 + 0.5     # unnormalize roughly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images[:4]))

# infer
outputs = model(images[:4])
# sort logits
logits_args = np.argsort(outputs.data)

for i in range(4):
    print('True label for img %i: %s'%(i,classes[labels[i]]))
    print('Predicted label for img %i: %s with probabilty %0.3f'%(i,classes[logits_args[i,-1]],outputs.data[logits_args[i,-1]]))

#========================================

# Save the train and test loss at each epoch
# Save the train and test accuracy at each epoch
model.eval()
test_loss_epoch = 0.
test_acc_epoch = 0.
test_total = 0

test_start = time.clock()

for k, (input_batch, label_batch) in enumerate(testloader, 0):
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

print('Total time for testing: %0.2f'%(time.clock()-test_start)

