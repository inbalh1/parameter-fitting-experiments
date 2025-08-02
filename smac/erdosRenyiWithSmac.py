# Trying to understand how to use SMAC
# This file handles only Erdos Renyi, and isn't the most updated version of our SMAC usage
from ConfigSpace import Configuration, ConfigurationSpace
from smac import BlackBoxFacade
from smac import Scenario

import sys
from models import GraphModel, ErdosRenyi
import parameters
import csv
# target function -this "evaluation function", whose returned value we want to minimize.
# facade - there are several options, and they say its important


# TODO: can write the code in a more general way, to combine with the other experiments (the smac should be just the fitter I guess)
def append_params(accumulated_params: list, cur_params: list):
    if not accumulated_params:
        return cur_params[:]
    for i in range(len(accumulated_params)):
        accumulated_params[i].value += cur_params[i].value
    return accumulated_params

debug = False
def target_function_generator(target_params, model_class=None):
    # This target function suppose to work for any model class
    def target_function(config: Configuration, seed: int):
        n = parameters.NumberOfVertices(config["n"])
        d = parameters.AverageDegree(config["d"])
        model = model_class(n, d)
        if debug:
            print('Input: ', n, d)

        num_of_samples = 2
        accumulated_params = {'n': 0, 'd': 0}
        for i in range(num_of_samples):
            g = model.generate()
            accumulated_params['n'] += g.numberOfNodes()
            accumulated_params['d'] += 2 * g.numberOfEdges() / g.numberOfNodes()
            if debug:
                print("output params are: ")
                print(g.numberOfNodes())
                print(2 * g.numberOfEdges() / g.numberOfNodes())

        avg = {key: val / num_of_samples for (key, val) in accumulated_params.items()}
        if debug:
            print('Accumulated: ', accumulated_params)
            print('target: ', target_params)
            print(' avg output: ', avg)

        res_n = (avg['n'] - float(target_params['n'])) ** 2
        res_d = (avg['d'] - float(target_params['d'])) ** 2
        return [res_n, res_d]

    return target_function


# TODO: should check whether we have something like that
# Configuration space for Erdos renyi
configspace = ConfigurationSpace({
    "n": (1000, 10000),
    "d": (2, 10)
})

def uniObjective(n_trials):
    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def multiObjective(n_trials, target_features, model_class: GraphModel):
    target_function = target_function_generator(target_features, model_class)
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials, objectives=["n", "d"])
    smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    print('*** Incumbent***')
    print(incumbent)

    # Taking average over resulting configs
    # TODO: why are there several???

    final_res = []
    for config in incumbent:
        # TODO: shouldn't this be in the function of extracting params?
        config_params = [param(config['n']), param(config['d'])]
        final_res.append(config_params)
    return final_res


def local_run(mode="all"):
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model}/*")]
    
    if mode == "compact":
        input_files = input_files[:1]
    base_input = '../output_data/target_params/erdos-renyi'
    base_output = f"../output_data/fitted_params/smac/{model}"
    #Args = namedtuple("Args", ["model", "input_file", "output_file"])
    model_class = ErdosRenyi

    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')


        
        #fitter_class = MLEFitter
        custom_fitter_config = {}
        #if alpha is not None:
        #    custom_fitter_config["alpha"] = alpha
        #if threshold is not None:
        #    custom_fitter_config["threshold"] = threshold

        with open(input_file) as input_dicts_file:
            param_dict = list(csv.DictReader(input_dicts_file))
            param_dict = param_dict[0]

        # TODO: this should be the fitter class
        fitter = multiObjective(n_trials=10, target_features=param_dict, model_class=model_class)
        # TODO: what to do with the output??
        #runner = ParameterFitterRunner(
        #    param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        #runner.execute()


print("Starting local run")
local_run("compact")

# Questions:
# The configuration space
# Target function - what should we minimize - compare to a single input graph?
# Facade - make should to choose the right one...
# number of trials / mean over a batch of samples - how many??? Whats the connection?
