# Trying to understand how to use SMAC
from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac import BlackBoxFacade, Scenario, Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue
from models import ErdosRenyi, GraphModel, ALL_MODELS
from parameters import Parameter
from smacParametersSpec import PARAMS_SPEC
import sys
import os
import csv
import math
import numpy as np
from typing import Literal, TypeAlias, Dict
import random
import argparse
# facade - there are several options, and they say its important


n_trials = 60
num_of_samples = 30

FittedParameters:TypeAlias = list[list[Parameter]]

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



def target_function_generator(target_params: list[Parameter], num_of_samples: int, model_class: type[GraphModel]=None, params_weights: dict=None, is_multi_obj: bool=True):
    # Generates a target function for a specific input graph (works for any model)
    def target_function(config: Configuration, seed: int):
        # This is the evaluation function, that given a certain configuration,
        # returns the cost for each objective (parameter)
        input_params = []
        for input_param in model_class.input_parameters():
            non_transformed_value = config[input_param.name()]
            value = PARAMS_SPEC[input_param.name()].transform(non_transformed_value)
            input_params.append(input_param(value))

        model = model_class(*input_params)

        accumulated_params = []
        for i in range(num_of_samples):
            g = model.generate()
            cur_output_params = model.measure_output_parameters(g)
            accumulated_params = append_params(accumulated_params, cur_output_params)

        avg_output_params = [param.__class__(param.value / num_of_samples) for param in accumulated_params]

        # Get cost for each output parameter (= feature)
        params_costs = {out_param.name(): compare_param(out_param, target_param) for out_param, target_param in zip(
           avg_output_params, target_params)}
           
        if is_multi_obj:
            # Return a list of all the costs
            final_res = list(params_costs.values())
        else:
            # Return a weighted sum of the costs
            final_res = sum(cost * (params_weights[param]) ** 2 for param, cost in params_costs.items())
            # TODO: from some reason girg doesn't work without this assertion
            assert final_res is not None, "Returned None"
            assert np.isfinite(final_res), f"Non-finite value: {value}"
        # print("*** Final res: ")
        # print(final_res)
        return final_res
    return target_function


def extract_params_from_config(config: Configuration, model_class: type[GraphModel],
                            cur_res:dict={}):
    """
    Get a smac config, extract parameters for the models and add them to the current result
    """
    for param in model_class.input_parameters():
        param_name = param.name()
        if param_name in cur_res:
            cur_res[param_name] += config[param_name]
        else:
            cur_res[param_name] = config[param_name]
    return cur_res

def generate_weights(target_parameters: list[Parameter], model_class: type[GraphModel]) -> Dict[str, float]:
    # Calculate the weight for each parameter in the cost function
    # The weight is according to the the range of the output feature
    weights = {}
    
    features_dict = {}
    for in_param, target_param in zip(
        model_class.input_parameters(), target_parameters):
        features_dict[in_param.output_parameter().name()] = target_param.value

    for in_param, target_param in zip(
            model_class.input_parameters(), target_parameters):
        value = float(target_param.value)
        weight = PARAMS_SPEC[in_param.name()].generate_weight(value, model_class, features_dict)
        weights[in_param.output_parameter().name()] = weight
  
    print('Weights are: ', weights)
    return weights


def generate_config_space(target_parameters: list[Parameter], model_class: type[GraphModel]) -> ConfigurationSpace:
    # Generate the configuration space, based on the input graph and model
    # Example output:
    # configspace = ConfigurationSpace({
    #     "n": (1000, 15000),
    #     "d": (1, 15)
    # })
    config = {}
    configspace = ConfigurationSpace()
    
    features_dict = {}
    for in_param, target_param in zip(
        model_class.input_parameters(), target_parameters):
        features_dict[in_param.output_parameter().name()] = float(target_param.value)

    for out_param_name, target_param_value in features_dict.items():
        if out_param_name == 'n':
            n = target_param_value
            configspace.add(PARAMS_SPEC['n'].generate_config(n, model_class, features_dict))
        elif out_param_name == 'd':
            value = target_param_value
            configspace.add(PARAMS_SPEC['d'].generate_config(value, model_class, features_dict))
        elif out_param_name == 'heterogeneity':
            value = target_param_value
            configspace.add(PARAMS_SPEC['beta'].generate_config(value, model_class, features_dict))
        elif out_param_name == 'cc':
            value = target_param_value
            configspace.add(PARAMS_SPEC['t'].generate_config(value, model_class, features_dict))
        else:
            raise NotImplementedError(out_param_name)  

    # TODO: should add temprature here
    # TODO: consider create a parameers extended class with this info ( + thresholds )
    
    
    print('config is: ', config)
    return configspace

PARAM_NAME_TO_COST_THRESHOLD = { 'n': 100, 'd': 0.1, 'beta': np.inf, 't': np.inf}
# TODO: does termination callback ever have effect?
class TerminationCallback(Callback):
    def build_threshold(self, input_parameters: list[Parameter]):
        self.thresholds = [PARAM_NAME_TO_COST_THRESHOLD[param.name()] for param in input_parameters]


    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        for trial_value in smbo.runhistory.values():
            costs_vector = trial_value.cost
            if all(c <= t for c, t in zip(costs_vector, self.thresholds)):
                # TODO: Add logging
                print("Early stopping: all objectives below thresholds.")
                #smbo.solver.terminate = True
                #break
                return False
        return True
        

def uniObjective_example(n_trials: int):
    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=n_trials)
    # Use SMAC to find the best configuration/hyperparameters
    smac = BlackBoxFacade(scenario, target_function=target1)
    incumbent = smac.optimize()
    return incumbent
    
def config_to_params(config: Configuration, model_class: type[GraphModel])->list[Parameter]:
    """
    Takes a configuration, returns as a list of parameters according to the model's input parameters
    """
    res = []
    for param in model_class.input_parameters():
        non_transformed_value = config[param.name()]
        value = PARAMS_SPEC[param.name()].transform(non_transformed_value)
        res.append(param(value))
    print('*** Result after transformation: ', res)
    return res
    
    
# TODO: get rid of avg mode
def extract_res_from_incumbent(incumbent, model_class: type[GraphModel], mode='random')->FittedParameters:
    final_res = []
    if type(incumbent) == list:
        if mode == 'first':
            final_res = [config_to_params(incumbent[0], model_class)]
        if mode == 'random':
            chosen_incumbent = random.choice(incumbent)
            final_res = [config_to_params(chosen_incumbent, model_class)]
        if mode == 'all':
            for config in incumbent:
                config_params = config_to_params(config, model_class)
                final_res.append(config_params)
        if mode == 'avg':
            # TODO: should also transform here
            avg_incumbent = {}
            for config in incumbent:
                avg_incumbent = extract_params_from_config(config, model_class, avg_incumbent)

            for param in model_class.input_parameters():
                avg_incumbent[param.name()] /= len(incumbent)

            final_res = [[param(avg_incumbent[param.name()]) for param in model_class.input_parameters()]]
    else:
        # This is usually for uni objective
        final_res = [config_to_params(incumbent, model_class)]

    return final_res
    
# TODO: change function name (it now also handles uni obj)
def multiObjective(n_trials: int, target_parameters: list[Parameter], model_class: type[GraphModel], output_directory:str, num_of_samples: int=10, is_multi_obj: bool=False)->tuple[FittedParameters, int]:
    """
    Runs the smac multi-objective optimization
    Notice output_directory should be unique for each input file
    Returns a list of all the resulting incumbents
    """
    # params_weights is only required for uni objective
    params_weights = generate_weights(target_parameters, model_class)
    target_function = target_function_generator(target_parameters, num_of_samples=num_of_samples, model_class=model_class, params_weights=params_weights, is_multi_obj=is_multi_obj)
    configspace = generate_config_space(target_parameters, model_class)
    objectives = [param.name() for param in model_class.input_parameters()]
    if is_multi_obj:
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            objectives=objectives,
            output_directory=output_directory
            )
        callback = TerminationCallback()
        callback.build_threshold(model_class.input_parameters())
        smac = BlackBoxFacade(scenario, target_function=target_function, callbacks=[callback])
    else:
        # No objectives parameter
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            output_directory=output_directory
            )
        # TODO: should set termination criteria of its own
        smac = BlackBoxFacade(scenario, target_function=target_function)
    incumbent = smac.optimize()
    print('*** Incumbent***')
    print(incumbent)
    
    # TODO: we can get more info on the run using smac
    #
    # All evaluated configurations (perhaps can be used to determine best budget)
    # for config, info in runhistory.data.items():
    #    print("Config:", config)
    #    print("Obj values:", info.cost)
    # Some extra info
    runhistory = smac.runhistory
    used_budget = len(runhistory._data)
    #print('Total used budget: ', used_budget)
    # This yields all configs, perhaps can look at their costs for diminishing returns of budget...
    #print(runhistory._data)
    # Cost of a specific config:
    #for config in incumbent:
    #    print('Cost: ', runhistory.get_cost(config))
    #used_trials = smac.stats.ta_runs
    


    return extract_res_from_incumbent(incumbent, model_class), used_budget

def writeResultsWrapper(fitted_parameters: [Parameter], output_file:str, *args, **kwargs):
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
def writeResults(fitted_parameters: list[Parameter], output_file:str, model_class:type[GraphModel], target_features: list[Parameter], fitter_name: str, used_budget: int):
    row_data = {}
    # row_data['Graph'] = param_dict['Graph']
    row_data['Fitter'] = fitter_name
    row_data['used_budget'] = used_budget

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


# TODO: consider writing a general local run (to run experiments, just without the run package,
# which is parallel and without prints...)
# todo: since I run locally, should make sure it doesnt run again existing files
def local_run(model_name: str, is_multi_obj:bool, mode: Literal['all', 'compact']='all'):
    import glob
    from collections import namedtuple

    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    model_class = model_choices[model_name]
    input_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"../output_data/target_params/{model_name}/*")]
    
    if mode == "compact":
        input_files = input_files[:1]
    base_input = f'../output_data/target_params/{model_name}'
    base_output = f"../output_data/fitted_params/smac/{model_name}"
    

    for i in input_files:
        input_file = os.path.join(base_input, f'{i}.csv')
        output_file = os.path.join(base_output, f'{i}.csv')
        
        if (os.path.exists(output_file)):
            continue
        
        print("Working", input_file)

        custom_fitter_config = {}
        with open(input_file) as input_dicts_file:
            target_features = list(csv.DictReader(input_dicts_file))
            target_features = target_features[0]
        print("param dict is: ", target_features)
        print("Input file: ", input_file)
        
        parameters = []
        # TODO: make sure we compare to the target parameters (and not the input)
        # TODO: this TODO is important
        parameter_classes = [input_param.output_parameter()
                             for input_param in model_class.input_parameters()]
        for parameter_class in parameter_classes:
            value = target_features[parameter_class.name()]
            parameter = parameter_class(value)
            parameters.append(parameter)
        output_directory = os.path.join(base_output, "smac_output", i)
        fitter = multiObjective(
            n_trials=n_trials,
            target_parameters=parameters,
            model_class=model_class,
            num_of_samples=num_of_samples,
            output_directory=output_directory,
            is_multi_obj=is_multi_obj
            )
        
        fitted_parameters, used_budget = fitter
        writeResultsWrapper(fitted_parameters, output_file, model_class, target_features=target_features, fitter_name="smac", used_budget=used_budget)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_file', type=str)
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    parser.add_argument('--model', type=str.lower,
                        choices=model_choices.keys(), required=True)
    parser.add_argument('--multi-obj', help='True for multi-variate optimization, false otherwise (default)', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    # model_class = model_choices[args.model]
    local_run(model_name=args.model, is_multi_obj=args.multi_obj, mode='all')
    # local_run(model_name='chung-lu-pl', mode='compact')

# Questions:
# Parameters for beta, temperature - config space? cost threshold?
