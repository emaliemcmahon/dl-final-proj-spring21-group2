# -*- coding: utf-8 -*-

# Name: train.py
# Training loop for CORnet on CIFAR-10

import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint
from tqdm import tqdm
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
parser.add_argument('-model','--model_name', default='CORnet-Z', type=str,
                    help='the name of the model to train')
parser.add_argument('-feedback_connections','--feedback_connections', default={}, type=str,
                    help='whether the model has feedback connections')
parser.add_argument('-epochs','--n_epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('-batch_size','--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('-lr', '--learning_rate', default=.001, type=float,
                    help='initial learning rate')
parser.add_argument('-step','--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('-momentum','--momentum', default=.9, type=float, help='momentum')
parser.add_argument('-decay','--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('-gamma','--gamma', default=0.1, type=float,
                    help='scheduler multiplication factor')
parser.add_argument('-patience','--early_stop_patience', default=3, type=int,
                    help='no. of epochs patience for early stopping ')
args = parser.parse_args()

now = datetime.now()
date = f'{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}'
print('date: %s'%(date))
print(f'model: {args.model_name}')
print(f'feedback: {args.feedback_connections}')

print('==> Preparing data..')

# image size = 32x32x3
# resize image to 256x256x3 to match ImageNet image size for pretrained CORnet-Z
# Modify mean and stdev to ImageNet mean and stdev
# CIFAR-10 : transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# hyperparameters based on the CORnet paper
batch_size = args.batch_size
n_epochs = args.n_epochs
early_stop = args.early_stop_patience
learning_rate = args.learning_rate
gamma = args.gamma

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
model = CORnet(pretrained=True, architecture=args.model_name,
            feedback_connections=args.feedback_connections, n_classes=10).to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                   momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=gamma)

# Save the train and test loss at each epoch
# Save the train and test accuracy at each epoch
# Save the model at each epoch
# Implement early stopping

count = 0
total_loss_train, total_loss_test = [],[]
total_acc_train, total_acc_test = [],[]

train_start = time.clock()

for epoch in range(n_epochs):

    epoch_start = time.clock()
    model.train()
    train_loss_epoch = 0.
    train_acc_epoch = 0.
    train_correct = 0
    train_total = 0

    for i, (input_batch, label_batch) in tqdm(enumerate(trainloader,0), total=round(len(trainset)/batch_size)):
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        output_batch = model(input_batch)
        train_loss_batch = loss(output_batch,label_batch)

        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        train_loss_epoch += train_loss_batch.item()

        # calculate accuracy
        _, predicted = torch.max(output_batch.data, 1)
        train_total += label_batch.size(0)
        train_acc_epoch += (predicted.float() == label_batch.float()).sum()
        if i % 150 == 0:
            print('For epoch %i, batch %i train loss is %f'%(epoch, i, train_loss_batch.float()))

    total_loss_train.append(train_loss_epoch/train_total)
    total_acc_train.append(train_acc_epoch/train_total)

    model.eval()
    test_loss_epoch = 0.
    test_acc_epoch = 0.
    test_correct = 0
    test_total = 0

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

    total_loss_test.append(float(test_loss_epoch/test_total))
    total_acc_test.append(float(test_acc_epoch/test_total))

    print('For epoch %i train loss is %f'%(epoch,total_loss_train[-1]))
    print('For epoch %i test loss is %f'%(epoch,total_loss_test[-1]))

    print('For epoch %i train acc is %f'%(epoch,total_acc_train[-1]))
    print('For epoch %i test acc is %f'%(epoch,total_acc_test[-1]))

    # early stopping
    if total_loss_test[-1]>total_loss_test[-2]:
        if count < early_stop:
            count+=1
        else:
            count = 0
            print("Stopping early bec loss has not decreased for last %i epochs"%(early_stop))
            break
    else:
        torch.save(model.state_dict(),'checkpoints/cornetZ_%i_%s.pth'%(epoch,date))

    print('Time taken for this epoch: %0.2f'%(time.clock() - epoch_start))
    print('----------------')

print('Total time for training+testing: %0.2f'%(train_start - time.clock()))


# plotting the train and test loss and acc
plt.plot(total_loss_train,label='train loss')
plt.plot(total_loss_test,label='test loss')
plt.title(args.model_name + ' loss- training and testing')
plt.xlabel('no. of epochs')
plt.legend()
plt.savefig('plots/' + args.model_name + '_loss_%s.png'%(date))
plt.close()

plt.plot(total_acc_train,label='train acc')
plt.plot(total_acc_test,label='test acc')
plt.title(args.model_name + ' accuracy- training and testing')
plt.xlabel('no. of epochs')
plt.legend()
plt.savefig('plots/' + args.model_name + '_acc_%s.png'%(date))
plt.close()
