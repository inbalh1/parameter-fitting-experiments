# Trying to understand how to use SMAC
from ConfigSpace import Configuration, ConfigurationSpace
from smac import BlackBoxFacade
from smac import Scenario

import sys
from models import ErdosRenyi, GraphModel
from parameters import Parameter
import csv
# target function - I suppose this is this "evaluation function". They say that we want to minimize the value returned from this function
# facade - there are several options, and they say its important


# TODO: can write the code in a more general way, to combine with the other experiments (the smac should be just the fitter I guess)
def append_params(accumulated_params: list, cur_params: list):
    if not accumulated_params:
        return cur_params[:]
    for i in range(len(accumulated_params)):
        accumulated_params[i].value += cur_params[i].value
    return accumulated_params

def compare_param(param: Parameter, target_params: dict):
    return (param.value - float(target_params[param.name()])) ** 2

# TODO: give model_class as input
def target_function_generator(target_params: dict, num_of_samples: int=2, model_class: type[GraphModel]=None):
    # This target function suppose to work for any model class
    def target_function(config: Configuration, seed: int):
        input_params = []
        for input_param in model_class.input_parameters():
            value = config[input_param.name()]
            input_params.append(input_param(value))

        model = model_class(*input_params)

        accumulated_params = []
        for i in range(num_of_samples):
            g = model.generate()
            cur_output_params = model.measure_output_parameters(g)
            accumulated_params = append_params(accumulated_params, cur_output_params)
            # print("output params are: ")
            # print(cur_output_params)

        # print('Accumulated: ', accumulated_params)
        # print('target: ', target_params)
        avg_output_params = [param.__class__(param.value / num_of_samples) for param in accumulated_params]
        # print(' avg output: ', avg_output_params)

        final_res = [compare_param(param, target_params) for param in avg_output_params]
        # print("*** Final res: ")
        # print(final_res)
        # Return the targets
        return final_res
    return target_function


# TODO: should check whether we have something like that
# Configuration space for Erdos renyi
configspace = ConfigurationSpace({
    "n": (1000, 15000),
    "d": (2, 10)
})

def extractParamsFromConfig(config: Configuration, model_class: type[GraphModel],
                            cur_res:dict={}):
    for param in model_class.input_parameters():
        param_name = param.name()
        if param_name in cur_res:
            cur_res[param_name] += config[param_name]
        else:
            cur_res[param_name] = config[param_name]
    return cur_res

def uniObjective(n_trials):
    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def multiObjective(n_trials: int, target_features: dict, model_class: type[GraphModel], num_of_samples: int=2):
    target_function = target_function_generator(target_features, num_of_samples=num_of_samples, model_class=model_class)
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials, objectives=["n", "d"])
    smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    print('*** Incumbent***')
    print(incumbent)

    # Taking average over resulting configs
    # TODO: why are there several???
    avg_incumbent = {}
    for config in incumbent:
        avg_incumbent = extractParamsFromConfig(config, model_class, avg_incumbent)

    for param in model_class.input_parameters():
        avg_incumbent[param.name()] /= len(incumbent)

    final_res = [param(avg_incumbent[param.name()]) for param in model_class.input_parameters()]

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
    model_class = ErdosRenyi

    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        
        print("Working", input_file)


        
        #fitter_class = MLEFitter
        custom_fitter_config = {}
        #if alpha is not None:
        #    custom_fitter_config["alpha"] = alpha
        #if threshold is not None:
        #    custom_fitter_config["threshold"] = threshold

        with open(input_file) as input_dicts_file:
            param_dict = list(csv.DictReader(input_dicts_file))
            param_dict = param_dict[0]
        print("param dict is: ", param_dict)
        print("Input file: ", input_file)
        
        # TODO: this should be the fitter class
        fitter = multiObjective(10, param_dict, model_class)
        # TODO: what to do with the output?? Should somehow write it to a file
        #runner = ParameterFitterRunner(
        #    param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        #runner.execute()


local_run("compact")

# Questions:
# The configuration space
# Target function - what should we minimize - compare to a single input graph?
# Facade - make should to choose the right one...
# number of trials / mean over a batch of samples - how many??? Whats the connection?
