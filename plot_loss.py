import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_loss(args):
    # plotting the train and test loss and acc
    train = np.load(f'plots/{args.model_name}_train.npy')
    test = np.load(f'plots/{args.model_name}_test.npy')

    plt.plot(train[:,0],label='train loss')
    plt.plot(test[:,0],label='test loss')
    plt.title(args.model_name + ' loss- training and testing')
    plt.xlabel('no. of epochs')
    plt.legend()
    plt.savefig(f'plots/{args.model_name}_loss.png')
    plt.close()

    plt.plot(train[:,1],label='train acc')
    plt.plot(test[:,1],label='test acc')
    plt.title(args.model_name + ' accuracy- training and testing')
    plt.xlabel('no. of epochs')
    plt.legend()
    plt.savefig(f'plots/{args.model_name}_acc.png')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='plotting training results')
    parser.add_argument('-model', '--model_name', default='CORnet-Z', type=str,
                        help='the name of the model to train')
    return args

def main():
    args = parse_args()
    plot_loss(args)

if __name__ == "__main__":
    main()