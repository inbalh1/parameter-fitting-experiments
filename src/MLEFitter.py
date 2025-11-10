import csv
import argparse
from pathlib import Path

from fit_parameters import ParameterFitterRunner
from parameter_fitters import ParameterFitter
from models import *
from parameters import *

# Specific fitter for erdos-renyi. Not general MLE
class MLEFitter(ParameterFitter):
    def run(self):
        # TODO: still not sure what all of these params means...
        params = list(in_param.input_guess(target_param) for in_param, target_param in zip(
            self.model.input_parameters(), self.target_parameters))
        n, d, m = params
        d = d.__class__(self.calc_d(n.value, d.value, m.value))
        n = n.__class__(int(n.value))
        averaged_params = [n, d, m]
        #for param in params:
        #    averaged_params.append(param.__class__(param.value))
        return averaged_params
        
    def statistics(self):
        averaging_iterations = 1
        total_iterations = 1
        flips = [-1] * len(self.target_parameters) # flip for each param
        return averaging_iterations, total_iterations, flips
        
    def calc_d(self, n, d, m):
        # TODO: go over this calculation
        n = int(n)
        opt_p = 2 * m / (n * (n - 1))
        d = opt_p * (n - 1)
        return d
        
        
    @staticmethod
    def name():
        return 'MLE'
    
def local_run():
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model}/*")]
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
    #local_run()
    #print("Done")
    #exit(0)
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


    fitter_class = MLEFitter
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

