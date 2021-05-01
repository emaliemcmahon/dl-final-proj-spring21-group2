# -*- coding: utf-8 -*-

# Name: train.py
# Training loop for CORnet on CIFAR-10

import argparse
from tqdm import tqdm
from datetime import datetime
import os, glob, pickle, copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from cornet import CORnet
from plot_loss import plot_loss

def load_data(args):
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False)
    return trainloader, int(len(trainset)/args.batch_size), testloader


def load_model(device, args):
    # load model
    model = CORnet(architecture=args.model_name, pretrained=True, feedback_connections=args.feedback_connections, n_classes=10)
    if args.resume_training:
        ckpts = glob.glob(f'checkpoints/{args.model_name}_{args.feedback_connections}/*.pth')
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        print(f'picking up from {latest_ckpt}')
        model.load_state_dict(torch.load(latest_ckpt))
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return model, scaler, loss, optimizer, scheduler

def train(device, args, trainloader, n_batches, testloader, model, scaler, loss, optimizer, scheduler):
    # Save the train and test loss at each epoch
    # Save the train and test accuracy at each epoch
    # Save the model at each epoch
    # Implement early stopping
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.resume_training:
        train = np.load(f'plots/{args.model_name}_{args.feedback_connections}_train.npy', allow_pickle=True)
        test = np.load(f'plots/{args.model_name}_{args.feedback_connections}_test.npy', allow_pickle=True)

        total_loss_train, total_loss_test = list(train[0]), list(test[0])
        total_acc_train, total_acc_test = list(train[1]), list(test[1])
        start_epoch = len(total_loss_train) - 1
    else:
        total_loss_train, total_loss_test = [], []
        total_acc_train, total_acc_test = [], []
        start_epoch = 0

    patience_counter = 0
    for epoch in range(start_epoch, args.n_epochs):

        model.train()
        train_loss_epoch = 0.
        train_acc_epoch = 0.
        train_correct = 0
        train_total = 0

        for i, (input_batch, label_batch) in tqdm(enumerate(trainloader, 0), total=n_batches, position=0, leave=True):
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                output_batch = model(input_batch)
                train_loss_batch = loss(output_batch, label_batch)

            scaler.scale(train_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += train_loss_batch.item()

            # calculate accuracy
            _, predicted = torch.max(output_batch.data, 1)
            train_total += label_batch.size(0)
            train_acc_epoch += (predicted.float() == label_batch.float()).sum()
            if i == 0 or i % 250 == 0 or i == (n_batches-1):
                print('For epoch %i, batch %i train loss is %f' % (epoch, i, train_loss_batch.float()))

        total_loss_train.append(train_loss_epoch/train_total)
        total_acc_train.append(float(train_acc_epoch/train_total))

        model.eval()
        test_loss_epoch = 0.
        test_acc_epoch = 0.
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for k, (input_batch, label_batch) in enumerate(testloader, 0):
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                with torch.cuda.amp.autocast():
                    output_batch = model(input_batch)
                    test_loss_batch = loss(output_batch, label_batch)

                test_loss_epoch += test_loss_batch.item()

                # calculate accuracy
                _, predicted = torch.max(output_batch.data, 1)
                test_total += label_batch.size(0)
                test_acc_epoch += (predicted.float() == label_batch.float()).sum()

        total_loss_test.append(float(test_loss_epoch/test_total))
        total_acc_test.append(float(test_acc_epoch/test_total))

        print('For epoch %i train loss is %f' % (epoch, total_loss_train[-1]))
        print('For epoch %i test loss is %f' % (epoch, total_loss_test[-1]))

        print('For epoch %i train acc is %f' % (epoch, total_acc_train[-1]))
        print('For epoch %i test acc is %f' % (epoch, total_acc_test[-1]))


        np.save(f'plots/{args.model_name}_{args.feedback_connections}_train.npy', np.array([total_loss_train, total_acc_train]))
        np.save(f'plots/{args.model_name}_{args.feedback_connections}_test.npy', np.array([total_loss_test, total_acc_test]))
        plot_loss(args)
        now = datetime.now()
        date = f'{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}'
        torch.save(model.state_dict(), 'checkpoints/%s_%s/%s_%s_%s_%s.pth' % (args.model_name,
                   args.feedback_connections, args.model_name, args.feedback_connections, str(epoch).zfill(2), date))

        # early stopping
        if epoch > 1 and total_loss_test[-1] > total_loss_test[-2]:
            if patience_counter < args.early_stop_patience:
                patience_counter += 1
            else:
                patience_counter = 0
                print("Stopping early bec loss has not decreased for last %i epochs" % (args.early_stop_patience))
                break


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Training')
    parser.add_argument('-model', '--model_name', default='CORnet-Z', type=str,
                        help='the name of the model to train')
    parser.add_argument('-feedback_connections', '--feedback_connections', default={}, type=str,
                        help='whether the model has feedback connections')
    parser.add_argument('-epochs', '--n_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-batch_size', '--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('-lr', '--learning_rate', default=.001, type=float,
                        help='initial learning rate')
    parser.add_argument('-step', '--step_size', default=10, type=int,
                        help='after how many epochs learning rate should be decreased 10x')
    parser.add_argument('-momentum', '--momentum', default=.9, type=float, help='momentum')
    parser.add_argument('-decay', '--weight_decay', default=1e-4, type=float,
                        help='weight decay ')
    parser.add_argument('-gamma', '--gamma', default=1, type=float,
                        help='scheduler multiplication factor, default is no change in LR')
    parser.add_argument('-patience', '--early_stop_patience', default=3, type=int,
                        help='no. of epochs patience for early stopping ')
    parser.add_argument('-resume_training', '--resume_training', default=False, type=bool,
                        help='whether the training should be resumed')
    input_args = parser.parse_args()

    if input_args.resume_training:
        args = load_object(f'checkpoints/{args.model_name}_{args.feedback_connections}/hyperparameters.pkl')
        args.resume_training = True
    else:
        args = copy.deepcopy(input_args)

    now = datetime.now()
    date = f'{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}'
    print(args)
    return args

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    device = torch.device(device)

    args = parse_args()
    save_object(args, f'checkpoints/{args.model_name}_{args.feedback_connections}/hyperparameters.pkl')
    trainloader, n_batches, testloader = load_data(args)
    model, scaler, loss, optimizer, scheduler = load_model(device, args)
    train(device, args, trainloader, n_batches, testloader, model, scaler, loss, optimizer, scheduler)


if __name__ == "__main__":
    main()
