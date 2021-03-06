import functools
import argparse
from pathlib import Path
from datetime import datetime
import torch
import importlib

from brainscore import score_model
from cornet import CORnet

from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

from candidate_models.model_commitments.cornets import CORnetCommitment
from candidate_models.base_models import cornet

# to use, make sure have brainscore downloaded: pip install git+https://github.com/brain-score/brain-score 
# to use result caching, make sure have downloaded: pip install git+https://github.com/mschrimpf/result_caching
# to use model_tools, download library: pip install "candidate_models @ git+https://github.com/brain-score/candidate_models"

# uncomment next line for result caching
def score_on_benchmark(model, model_name, benchmark, layers):
    # ImageNet mean and image size 224 x 224
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    # convert model to activations model --> extract activations from any layer
    activations_model = PytorchWrapper(identifier=model_name, model=model, preprocessing=preprocessing)

    # map layers onto cortical regions using standard commitments
    model = ModelCommitment(identifier=model_name, activations_model=activations_model,
                            layers=layers)

    # score activation model on given benchmark
    # in this case used public benchmark w/neural recordings in macaque IT
    #   from Majaj, Hong et al. 2015
    #   neural predictivity metric based on PLS regression
    score = score_model(model_identifier=model.identifier,
                        model=model,
                        benchmark_identifier=benchmark)
    print(score)
    """
    # simplified print score output
    center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
    print(f"score: {center.values:.3f}+-{error.values:.3f}")
    """
    return score


def parse_args():
    parser = argparse.ArgumentParser(description='BrainScore Evaluation')
    # read in model to be loaded and the .pth for parameters in checkpoint
    parser.add_argument('-m', '--model_name', default='CORnet-Z', type=str,
                        help='the name of the model to train')
    parser.add_argument('-c', '--checkpoint_path', type=str,
                         help='the file path the desired checkpoint is stored')
    parser.add_argument('-b', '--benchmark', default='dicarlo.MajajHong2015public.IT-pls', type=str,
                        help='the name of benchmark for brain score')
    parser.add_argument('-l', '--layers', default=None, type=str, nargs='+',
                        help='the specified layers for activation')    
    parser.add_argument('-f', '--feedback_connections', default={}, type=str,
                        help='whether feedback connections were implemented')                          
    args = parser.parse_args()


    now = datetime.now()
    date = f'{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}'
    print('date: %s'%(date))
    return args

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    device = torch.device(device)

    args = parse_args()

    # load model
    print(f'model: {args.model_name}')
    model = CORnet(pretrained=True, architecture=args.model_name, feedback_connections=args.feedback_connections, n_classes=10)
    path = args.checkpoint_path
    model.load_state_dict(torch.load(path))
    model = model.to(device)

    # run brain score
    print(f'benchmark: {args.benchmark}')
    print(f'layers: {args.layers}')
    score_on_benchmark(model, args.model_name, args.benchmark, args.layers)


if __name__ == "__main__":
    main()


