import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def get_feedback_weights(model):

    feedback_weights = {}

    for receiver_name, giver_names in model.inverted_feedback_connections.items():
        feedback_weights[receiver_name] = {}
        
        input_name = model.sequence[model.sequence.index(receiver_name) - 1]
        feedback_weights[receiver_name][input_name] = model.feedback[receiver_name].weight[:, -model.sizes[receiver_name]['input'][1]:, :, :].squeeze().cpu().detach().numpy()
        
        start = 0
        end = 0
        for giver_name in giver_names:
            end += model.sizes[giver_name]['output'][1]
            feedback_weights[receiver_name][giver_name] = model.feedback[receiver_name].weight[:, start:end, :, :].squeeze().cpu().detach().numpy()
            start = end
        
    return feedback_weights


def plot_feedback_weights(feedback_weights):

    fig, axes = plt.subplots(ncols=len(feedback_weights), nrows=1, sharey=True)
    
    for i_receiver, receiver_name in enumerate(feedback_weights.keys()):
        data = np.vstack([feedback_weights[receiver_name][giver_name].mean(axis=1) for giver_name in feedback_weights[receiver_name].keys()]).transpose()
        line_collection = LineCollection(np.stack((np.tile(np.arange(1, len(feedback_weights[receiver_name]) + 1), (data.shape[0], 1)), data), axis=-1), linewidths=0.5, colors=(0, 0, 0, 0.2))
        axes[i_receiver].violinplot(data, showmeans=True, showextrema=True, showmedians=True)
        axes[i_receiver].add_collection(line_collection)
        axes[i_receiver].set_title(receiver_name)
        axes[i_receiver].set_xticklabels(feedback_weights[receiver_name].keys())
        axes[i_receiver].set_xticks([i_violin + 1 for i_violin in range(len(feedback_weights[receiver_name].keys()))])
        
    axes[0].set_ylabel('mean feedback weight')
    fig.suptitle('areas receiving feedback')

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':

    from cornet import CORnet
    
    model = CORnet(architecture='CORnet-Z', n_classes=10, feedback_connections='all', pretrained=True, n_passes=1)
    
    try:
        feedback_weights = get_feedback_weights(model)
        plot_feedback_weights(feedback_weights)
        print('function passed the test')
    except:
        print('function failed the test')
