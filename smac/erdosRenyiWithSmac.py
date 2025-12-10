# Trying to understand how to use SMAC
# This file handles only Erdos Renyi, and isn't the most updated version of our SMAC usage
from ConfigSpace import Configuration, ConfigurationSpace, OrdinalHyperparameter
from smac import BlackBoxFacade
from smac import Scenario

import sys
import math
from models import GraphModel, ErdosRenyi
from parameters import Parameter, NumberOfVertices, AverageDegree
import csv
# target function -this "evaluation function", whose returned value we want to minimize.
# facade - there are several options, and they say its important

n_trials = 20
num_of_samples = 10

# TODO: can write the code in a more general way, to combine with the other experiments (the smac should be just the fitter I guess)
def append_params(accumulated_params: list, cur_params: list):
    if not accumulated_params:
        return cur_params[:]
    for i in range(len(accumulated_params)):
        accumulated_params[i].value += cur_params[i].value
    return accumulated_params

debug = False
def target_function_generator(target_params):
    # This target function suppose to work for any model class
    def target_function(config: Configuration, seed: int):
        n = NumberOfVertices(config["n"])
        d = AverageDegree(config["d"])
        model = ErdosRenyi(n, d)
        if debug:
            print('Input: ', n, d)

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


# Configuration space for Erdos renyi
#configspace = ConfigurationSpace()
#configspace.add(
#        OrdinalHyperparameter(
#        "n",
#        sequence=list(range(1000, 10000))  
#        ))        
#configspace.add(OrdinalHyperparameter(
#    "d",
#        sequence=list(range(1, 15))
#    ))
    

def generate_config_space(target_parameters: list[Parameter]) -> ConfigurationSpace:
    # Generate the configuration space, based on the input graph and model
    config = {}
    configspace = ConfigurationSpace()

    target_n = target_parameters['n']
    n = float(target_n)
    max_value = int(math.floor(n + math.sqrt(n))) + 1
    min_value = math.floor(n)
    configspace.add(
        OrdinalHyperparameter(
        "n",
        sequence=list(range(min_value, max_value))  
    ))

    target_d = target_parameters['d']
    configspace.add(OrdinalHyperparameter(
        "d",
        sequence=list(range(1, 15))
    ))    
    return configspace

def uniObjective(n_trials):
    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=n_trials
        )
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def extract_res_from_incumbent(incumbent, mode='all')->list[list[Parameter]]:
    if mode == 'first':
        config = incumbent[0]
        config_params = [NumberOfVertices(config['n']), AverageDegree(config['d'])]
        return [config_params]
    if mode == 'all':
        final_res = []
        for config in incumbent:
            # TODO: shouldn't this be in the function of extracting params?
            config_params = [NumberOfVertices(config['n']), AverageDegree(config['d'])]
            final_res.append(config_params)
        return final_res
        
    
def multiObjective(n_trials, target_features) -> list[list[Parameter]]:
    target_function = target_function_generator(target_features)
    configspace = generate_config_space(target_features)
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=n_trials,
        objectives=["n", "d"],
        output_directory=f"smac3_output/{target_features['n']}_{target_features['d']}"
        )
    smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    print('*** Incumbent***')
    print(incumbent)

    # Taking average over resulting configs
    # TODO: why are there several???

    return extract_res_from_incumbent(incumbent)

def writeResultsWrapper(fitted_parameters: list[list[Parameter]], output_file:str, *args, **kwargs):
    """
    Since smac might return multiple results, we take each one, and write is separately
    """
    for i, params in enumerate(fitted_parameters):
        if len(fitted_parameters) > 1:
            cur_output_file = f'{output_file.rsplit(".", 1)[0]}_{i}.{output_file.rsplit(".", 1)[1]}'
        else:
            cur_output_file = output_file
        writeResults(params, cur_output_file, *args, **kwargs)

# Function taken from ParameterFitterRunner
def writeResults(fitted_parameters: list[Parameter], output_file:str, model_class:type[GraphModel], target_features: list[Parameter], fitter_name: str):
    row_data = {}
    # row_data['Graph'] = param_dict['Graph']
    row_data['Fitter'] = fitter_name

    parameter_classes = [NumberOfVertices, AverageDegree]
    for parameter_class in parameter_classes:
        value = target_features[parameter_class.name()]
        parameter = parameter_class(value)
        row_data['target_' + parameter_class.name()] = parameter.value

    for fitted_param in fitted_parameters:
        row_data[fitted_param.name()] = fitted_param.value

    # averaging_iterations, total_iterations, flips = fitter.statistics()
    # smoothing_iterations = total_iterations - averaging_iterations
    # row_data['averaging_iterations'] = averaging_iterations
    # row_data['smoothing_iterations'] = smoothing_iterations
    # row_data['total_iterations'] = total_iterations
    # for flip_count, param in zip(flips, parameter_classes):
    #     row_data['flips_' + param.name()] = flip_count
    # for key, value in self.custom_fitter_config.items():
    #     row_data[key] = value

    with open(output_file, "w") as results_file:
        fieldnames = sorted(set(row_data.keys()))
        dict_writer = csv.DictWriter(results_file, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerow(row_data)


def local_run(mode="all"):
    import glob
    import os
    from collections import namedtuple

    model = 'erdos-renyi'
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model}/*")]
    
    if mode == "compact":
        input_files = input_files[:1]
    if mode == "medium":
        input_files = input_files[:10]
    base_input = '../output_data/target_params/erdos-renyi'
    base_output = f"../output_data/fitted_params/smac/{model}"
    #Args = namedtuple("Args", ["model", "input_file", "output_file"])
    model_class = ErdosRenyi

    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        print(f"input file: {input_file}")
        
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
        fitter = multiObjective(n_trials=n_trials, target_features=param_dict)
        fitted_parameters = fitter
        writeResultsWrapper(fitted_parameters, output_file, model_class, target_features=param_dict, fitter_name="smac")
        #runner = ParameterFitterRunner(
        #    param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        #runner.execute()


print("Starting local run")
local_run()

# Questions:
# The configuration space
# Target function - what should we minimize - compare to a single input graph?
# Facade - make should to choose the right one...
# number of trials / mean over a batch of samples - how many??? Whats the connection?
