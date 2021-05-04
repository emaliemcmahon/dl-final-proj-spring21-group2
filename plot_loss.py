import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

def plot_loss(args):
    # plotting the train and test loss and acc
    train_loss = np.load(f'plots/{args.model_name}_{args.feedback_connections}_train_loss.npy', allow_pickle=True)
    test_loss = np.load(f'plots/{args.model_name}_{args.feedback_connections}_test_loss.npy', allow_pickle=True)
    accuracy = np.load(f'plots/{args.model_name}_{args.feedback_connections}_accuracy.npy', allow_pickle=True)

    plt.plot(train_loss,label='train loss')
    plt.title(args.model_name + ' training loss')
    plt.xlabel('no. of minibatches')
    plt.legend()
    plt.savefig(f'plots/{args.model_name}_{args.feedback_connections}_batchloss.png')
    plt.close()

    train_loss = train_loss.reshape((len(test_loss), int(len(train_loss)/len(test_loss)))).mean(axis=1)
    plt.plot(train_loss,label='train loss')
    plt.plot(test_loss,label='test loss')
    plt.title(args.model_name + ' loss- training and testing')
    plt.xlabel('no. of epochs')
    plt.legend()
    plt.savefig(f'plots/{args.model_name}_{args.feedback_connections}_loss.png')
    plt.close()

    plt.plot(accuracy[0],label='train acc')
    plt.plot(accuracy[1],label='test acc')
    plt.title(args.model_name + ' accuracy- training and testing')
    plt.xlabel('no. of epochs')
    plt.legend()
    plt.savefig(f'plots/{args.model_name}_{args.feedback_connections}_acc.png')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='plotting training results')
    parser.add_argument('-model', '--model_name', default='CORnet-Z', type=str,
                        help='the name of the model to train')
    parser.add_argument('-feedback_connections', '--feedback_connections', default={}, type=str,
                        help='whether the model has feedback connections')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    plot_loss(args)

if __name__ == "__main__":
    main()
