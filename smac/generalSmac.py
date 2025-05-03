# Trying to understand how to use SMAC
from ConfigSpace import Configuration, ConfigurationSpace
from smac import BlackBoxFacade
from smac import Scenario

import sys
from models import ErdosRenyi, GraphModel
from parameters import Parameter
import csv
import math
# target function - I suppose this is this "evaluation function". They say that we want to minimize the value returned from this function
# facade - there are several options, and they say its important


# How to fit this to another model?
# Configuration space
# termination_cost_threshold parameter to scenario
# Choosing the model itself

# TODO: can write the code in a more general way, to combine with the other experiments (the smac should be just the fitter I guess)
def append_params(accumulated_params: list, cur_params: list):
    if not accumulated_params:
        return cur_params[:]
    for i in range(len(accumulated_params)):
        accumulated_params[i].value += cur_params[i].value
    return accumulated_params

# def compare_param(param: Parameter, target_params: dict):
#     return (param.value - float(target_params[param.name()])) ** 2

def compare_param(param: Parameter, target_param: Parameter):
    return (param.value - float(target_param.value)) ** 2



# TODO: give model_class as input
def target_function_generator(target_params: list[Parameter], num_of_samples: int=2, model_class: type[GraphModel]=None):
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

        # If target_params was a list:
        final_res = list(compare_param(out_param, target_param) for out_param, target_param in zip(
           avg_output_params, target_params))
        # final_res = [compare_param(param, target_params) for param in avg_output_params]
        # print("*** Final res: ")
        # print(final_res)
        # Return the targets
        return final_res
    return target_function


# TODO: determine this based on the input graph
# Configuration space for Erdos renyi


def extractParamsFromConfig(config: Configuration, model_class: type[GraphModel],
                            cur_res:dict={}):
    for param in model_class.input_parameters():
        param_name = param.name()
        if param_name in cur_res:
            cur_res[param_name] += config[param_name]
        else:
            cur_res[param_name] = config[param_name]
    return cur_res

def generate_config_space(target_parameters: list[Parameter], model_class: type[GraphModel]) -> ConfigurationSpace:
    # configspace = ConfigurationSpace({
    #     "n": (1000, 15000),
    #     "d": (1, 15)
    # })
    config = {}

    for in_param, target_param in zip(
            model_class.input_parameters(), target_parameters):
        if in_param.name() == "n":
            n = float(target_param.value)
            config["n"] = (n , n + math.sqrt(n))
        elif in_param.name() == "d":
            config["d"] = (1, 15)
        else:
            raise NotImplementedError()            


    # for param in model_class.input_parameters():
    #     if param.name() == "n":
    #         n = float(target_features["n"])
    #         config["n"] = (n , n + math.sqrt(n))
    #     elif param.name() == "d":
    #         config["d"] = (1, 15)
    #     else:
    #         raise NotImplementedError()
    configspace = ConfigurationSpace(config)
    print('config is: ', config)
    return configspace

def uniObjective(n_trials):
    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def multiObjective(n_trials: int, target_parameters: list[Parameter], model_class: type[GraphModel], num_of_samples: int=10):
    
    target_function = target_function_generator(target_parameters, num_of_samples=num_of_samples, model_class=model_class)
    configspace = generate_config_space(target_parameters, model_class)
    objectives = [param.name() for param in model_class.input_parameters()]
     #TODO: explore more the termination_cost_threshold, or objective_limits
     # My issue is - it stops when one of the objectives is below the cost (not all)
    scenario = Scenario(configspace, deterministic=True, n_trials=n_trials, objectives=objectives, termination_cost_threshold=[100, 1])
    smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    # print('*** Incumbent***')
    # print(incumbent)

    # Taking average over resulting incumbents
    avg_incumbent = {}
    for config in incumbent:
        avg_incumbent = extractParamsFromConfig(config, model_class, avg_incumbent)

    for param in model_class.input_parameters():
        avg_incumbent[param.name()] /= len(incumbent)

    final_res = [param(avg_incumbent[param.name()]) for param in model_class.input_parameters()]

    print('final res is: ', final_res)
    print('Final cost:', target_function(avg_incumbent, seed=0))
    return final_res

# Function taken from ParameterFitterRunner
def writeResults(fitted_parameters, output_file, model_class, target_features: list[Parameter], fitter_name):
    row_data = {}
    # row_data['Graph'] = param_dict['Graph']
    row_data['Fitter'] = fitter_name

    parameter_classes = [input_param.output_parameter()
                            for input_param in model_class.input_parameters()]
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
    base_input = '../output_data/target_params/erdos-renyi'
    base_output = f"../output_data/fitted_params/smac/{model}"
    model_class = ErdosRenyi

    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        
        print("Working", input_file)

        custom_fitter_config = {}
        with open(input_file) as input_dicts_file:
            target_features = list(csv.DictReader(input_dicts_file))
            target_features = target_features[0]
        print("param dict is: ", target_features)
        print("Input file: ", input_file)
        
        # TODO: this should be the fitter class
        parameters = []
        parameter_classes = [input_param.output_parameter()
                             for input_param in model_class.input_parameters()]
        for parameter_class in parameter_classes:
            value = target_features[parameter_class.name()]
            parameter = parameter_class(value)
            parameters.append(parameter)
        fitter = multiObjective(n_trials=100, target_parameters=parameters, model_class=model_class, num_of_samples=10)
        
        #runner = ParameterFitterRunner(
        #    param_dict, model_class, fitter_class, output_file, custom_fitter_config)
        #runner.execute()
        fitted_parameters = fitter
        writeResults(fitted_parameters, output_file, model_class, target_features=target_features, fitter_name="smac")


local_run()

# Questions:
# The configuration space
# Target function - what should we minimize - compare to a single input graph?
# Facade - make should to choose the right one...
# number of trials / mean over a batch of samples - how many??? Whats the connection?
