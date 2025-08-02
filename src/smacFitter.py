import csv
import argparse
from pathlib import Path

from fit_parameters import ParameterFitterRunner
from parameter_fitters import ParameterFitter
from models import *
from parameters import *
import multiprocessing
import logging
# import sys
# sys.path.insert(0, '../smac')
import generalSmac

# This file isn't ready and doesn't work yet...

class SmacFitter(ParameterFitter):
    def run(self):
        # target_parameters is a list[Parameter], target_features is a dict of the resulting features....
        print(self.target_parameters)
        n_trials = 100
        num_of_samples = 10
        smac_result = generalSmac.multiObjective(n_trials=n_trials,
            target_parameters=self.target_parameters,
            model_class=model_class,
            num_of_samples=num_of_samples)

        # Smac might return a few solutions. We take the first
        # TODO: might want to take a random one, or all of them
        return smac_result[0]
        
    def statistics(self)->None:
        return        
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
    logger = multiprocessing.get_logger()
    handler = logging.StreamHandler()
    # TODO: what is it all these logs I added?
    logger.setLevel(logging.DEBUG)
    # Optionally, set a formatter
    formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')
    handler.setFormatter(formatter)
    logger.info("In smac fitter")
    with open(input_file) as input_dicts_file:
        param_dict = list(csv.DictReader(input_dicts_file))[0]
        print(param_dict)
    runner = ParameterFitterRunner(
        param_dict, model_class, fitter_class, output_file, custom_fitter_config)
    runner.execute()

