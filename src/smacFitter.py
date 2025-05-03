import csv
import argparse
from pathlib import Path

from fit_parameters import ParameterFitterRunner
from parameter_fitters import ParameterFitter
from models import *
from parameters import *
import sys
sys.path.insert(0, '../smac')
import generalSmac

# This file isn't ready and doesn't work yet...

class SmacFitter(ParameterFitter):
    def run(self):
        # terget_parameters is a list[Parameter], target_features is a dict of the resulting features....
        print(self.target_parameters)
        n_trials = 100
        num_of_samples = 10
        averaged_params = generalSmac.multiObjective(n_trials=n_trials,
            target_parameters=self.target_parameters,
            model_class=model_class,
            num_of_samples=num_of_samples)
        print('params are: ', params)

        #for param in params:
        #    averaged_params.append(param.__class__(param.value))
        return averaged_params
        
    def statistics(self):
        averaging_iterations = 1
        total_iterations = 1
        flips = [-1] * len(self.target_parameters) # flip for each param
        return averaging_iterations, total_iterations, flips
        
    # def multiObjective(n_trials, target_features):
    #     scenario = Scenario(configspace, deterministic=True, n_trials=n_trials, objectives=["n", "d"])
    #     smac = BlackBoxFacade(scenario, target_function=target_function)
    #     incumbent = smac.optimize()
    #     return incumbent
        
        
    @staticmethod
    def name():
        return 'smac'
    
def local_run(mode='all'):
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model}/*")]
    
    if mode == 'compact':
        input_files = input_files[:1]
   
    base_input = '../output_data/target_params/erdos-renyi'
    base_output = f"../output_data/fitted_params/smac/{model}"
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    #Args = namedtuple("Args", ["model", "input_file", "output_file"])
    
    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        model_class = model_choices[model]
        
        print("Working", input_file)


        fitter_class = SmacFitter
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
    local_run('compact')
    print("Done")
    exit(0)
    # This is code duplication from fit_parameters (maybe should be outside, choosing method...)
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    parser.add_argument('--model', type=str.lower,
                        choices=model_choices.keys(), required=True)
    # parser.add_argument('--alpha', type=float, required=False)
    # parser.add_argument('--threshold', type=float, required=False)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model_class = model_choices[args.model]


    fitter_class = SmacFitter
    custom_fitter_config = {}
    #if alpha is not None:
    #    custom_fitter_config["alpha"] = alpha
    #if threshold is not None:
    #    custom_fitter_config["threshold"] = threshold

    with open(input_file) as input_dicts_file:
        param_dict = list(csv.DictReader(input_dicts_file))[0]
        print(param_dict)
    runner = ParameterFitterRunner(
        param_dict, model_class, fitter_class, output_file, custom_fitter_config)
    runner.execute()

