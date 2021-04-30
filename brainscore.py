import functools

from brainscore import score_model

from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

# to use, make sure have brainscore downloaded: pip install git+https://github.com/brain-score/brain-score 
# to use result caching, make sure have downloaded: pip install git+https://github.com/mschrimpf/result_caching
# to use model_tools, download library: pip install "candidate_models @ git+https://github.com/brain-score/candidate_models"

# uncomment next line for result caching
# @store()


def score_on_benchmark(model):
    # ImageNet mean and image size 224 x 224
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    # convert model to activations model --> extract activations from any layer
    activations_model = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing)

    # map layers onto cortical regions using standard commitments
    model = ModelCommitment(identifier='my-model',
                            activations_model=activations_model,
                            # TODO figure out which layers we want
                            layers=['conv1', 'relu1', 'relu2'])

    # score activation model on given benchmark
    # in this case used public benchmark w/neural recordings in macaque IT
    #   from Majaj, Hong et al. 2015
    #   neural predictivity metric based on PLS regression
    score = score_model(model_identifier=model.identifier, 
                        model=model,
                        benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
    print(score)
    """
    # simplified print score output
    center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
    print(f"score: {center.values:.3f}+-{error.values:.3f}")
    """
    return score
