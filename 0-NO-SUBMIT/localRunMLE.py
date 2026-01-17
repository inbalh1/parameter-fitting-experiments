import csv
import argparse
from pathlib import Path

from fit_parameters import ParameterFitterRunner
from parameter_fitters import ParameterFitter
from models import *
from parameters import *
from sample_and_measure import GraphSamplerAndMeasurer
from MLE import MLEFitter

# Local run of MLE

def sample_and_measure(mode="all"):
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/fitted_params/MLE/{model}/*.csv")]
    if mode == "compact":
        input_files = input_files[:1]
    base_input = '../output_data/fitted_params/erdos-renyi'
    base_output = f"../output_data/fitted_features/MLE-test/{model}"
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    samples = 10
    #Args = namedtuple("Args", ["model", "input_file", "output_file"])
    
    seed = 9381
    combine = False
    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        model_class = model_choices[model]

        if input_file is not None:
            with open(input_file, "r") as input_dicts_file:
                param_dict = list(csv.DictReader(input_dicts_file))[0]
        else:
            raise NotImplementedError

        runner = GraphSamplerAndMeasurer(
            param_dict, model_class, seed, samples, combine, output_file)
        runner.execute()

    
    
    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        model_class = model_choices[model + '-extended']


        fitter_class = MLEFitter
        custom_fitter_config = {}
        #if alpha is not None:
        #    custom_fitter_config["alpha"] = alpha
        #if threshold is not None:
        #    custom_fitter_config["threshold"] = threshold

        with open(input_file) as input_dicts_file:
            param_dict = list(csv.DictReader(input_dicts_file))
            param_dict = param_dict[0]
        
        runner = ParameterFitterRunner(
            param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        runner.execute()
    
def fit_parameters(mode):
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model}/*")]
    if mode == 'compact':
        input_files = input_files[:1]
    base_input = '../output_data/target_params/erdos-renyi'
    base_output = f"../output_data/fitted_params/MLE/{model}"
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    #Args = namedtuple("Args", ["model", "input_file", "output_file"])
    
    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        model_class = model_choices[model + '-extended']
        
        print("Working", input_file)


        fitter_class = MLEFitter
        custom_fitter_config = {}
        #if alpha is not None:
        #    custom_fitter_config["alpha"] = alpha
        #if threshold is not None:
        #    custom_fitter_config["threshold"] = threshold

        with open(input_file) as input_dicts_file:
            param_dict = list(csv.DictReader(input_dicts_file))
            param_dict = param_dict[0]
        
        runner = ParameterFitterRunner(
            param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        runner.execute()


if __name__ == "__main__":
#    sample_and_measure(mode="compact")
    fit_parameters(mode='all')
