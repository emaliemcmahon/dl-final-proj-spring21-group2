# -*- coding: utf-8 -*-

# Name: train.py
# Training loop for CORnet on CIFAR-10

import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

gpu = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-date', '--date', default=None,
                    help='Enter the date for naming checkpoints and plots')
parser.add_argument('-model','--model_name', default='CORnet_Z', type=str,
                    help='the name of the model to train')
parser.add_argument('-epochs','--n_epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('-batch_size','--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('-step','--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('-momentum','--momentum', default=.9, type=float, help='momentum')
parser.add_argument('-decay','--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('-patience','--early_stop_patience', default=3, type=int,
                    help='no. of epochs patience for early stopping ')
args = parser.parse_args()

print('==> Preparing data..')

# image size = 32x32x32
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

#hyperparameters based on the CORnet paper
batch_size = args.batch_size
n_epochs = args.n_epochs
early_stop = args.early_stop_patience
learning_rate = args.learning_rate

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


# Check with Raj about the model inputs and name
print(args.model_name)
if args.model_name == 'CORnet_Z':
    from cornet import CORnet_Z
    model = CORnet_Z(pretrained=True, feedback_connections='all')

if gpu:
  model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                   momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
'''
# weight initialization -> Already done in cornet.py
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
'''
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

    for i, (input_batch, label_batch) in enumerate(trainloader,0):
        print(input_batch.size)
        if gpu:
            input_batch = input_batch.cuda()
            label_batch = label_batch.cuda()

        output_batch = model(input_batch) #this may change based on Raj's implementation
        train_loss_batch = loss(output_batch,label_batch)

        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        train_loss_epoch+=train_loss_batch.item()
        #calculate accuracy
        _, predicted = torch.max(output_batch.data, 1)
        train_total += label_batch.size(0)
        train_acc_epoch += (predicted.float() == label_batch.float()).sum()

    total_loss_train.append(train_loss_epoch/train_total)
    total_acc_train.append(train_acc_epoch/train_total)

    model.eval()
    test_loss_epoch = 0.
    test_acc_epoch = 0.
    test_correct = 0
    test_total = 0

    for k, (input_batch, label_batch) in enumerate(testloader,0):
        if gpu:
            input_batch = input_batch.cuda()
            label_batch = label_batch.cuda()

        output_batch = model(input_batch) #this may change based on Raj's implementation
        test_loss_batch = loss(output_batch,label_batch)
        test_loss_epoch+=test_loss_batch.item()
        #calculate accuracy
        _, predicted = torch.max(output_batch.data, 1)
        test_total += label_batch.size(0)
        test_acc_epoch += (predicted.float() == label_batch.float()).sum()

    total_loss_test.append(float(test_loss_epoch/test_total))
    total_acc_test.append(float(test_acc_epoch/test_total))

    print('For epoch %i train loss is %f'%(epoch,total_loss_train[-1]))
    print('For epoch %i test loss is %f'%(epoch,total_loss_test[-1]))

    print('For epoch %i train acc is %f'%(epoch,total_acc_train[-1]))
    print('For epoch %i test acc is %f'%(epoch,total_acc_test[-1]))


    # Early stopping
    if total_loss_test[-1]>total_loss_test[-2]:
        if count < early_stop:
            count+=1
        else:
            count = 0
            print("Stopping early bec loss has not decreased for last %i epochs"%(early_stop))
            break
    else:
        torch.save(model.state_dict(),'checkpoints/cornetZ_%i_%s.pth'%(epoch,args.date))

    print('Time taken for this epoch: %0.2f'%(time.clock()-epoch_start))
    print('----------------')

print('Total time for training+testing: %0.2f'%(train_start-time.clock()))

#Plotting the train and test loss and acc
plt.plot(total_loss_train,label='train loss')
plt.plot(total_loss_test,label='test loss')
plt.title('CORnet-Z loss- training and testing')
plt.xlabel('no. of epochs')
plt.legend()
plt.savefig('plots/cornetZ_loss_%s.png'%(args.date))
plt.close()

plt.plot(total_acc_train,label='train acc')
plt.plot(total_acc_test,label='test acc')
plt.title('CORnet-Z accuracy- training and testing')
plt.xlabel('no. of epochs')
plt.legend()
plt.savefig('plots/cornetZ_acc_%s.png'%(args.date))
plt.close()
