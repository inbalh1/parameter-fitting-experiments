import csv
import argparse
from pathlib import Path

from parameter_fitters import ParameterFitter, RobbinsMonroFinal
import multiprocessing
from models import *
from parameters import *


class ParameterFitterRunner:
    def __init__(self, target_features: dict, model_class: type[GraphModel], fitter_class: type[ParameterFitter],
                 output_file: str, custom_fitter_config):
        self.target_features = target_features # A dict of the target features (cc, etc)
        self.model_class = model_class
        self.fitter_class = fitter_class
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.custom_fitter_config = custom_fitter_config
        self.fitter_params = None # Will be determined during execute

        networkit.engineering.setNumberOfThreads(1)
        # logger = multiprocessing.log_to_stderr(logging.INFO)
        # logging.basicConfig(
        #    level=logging.INFO,
        #    format='%(asctime)s\t%(levelname)s:%(name)s:%(process)s:%(message)s')

    def execute(self):
        # logging.basicConfig(
        #    level=logging.INFO,
        #    format='%(asctime)s\t%(levelname)s:%(name)s:%(process)s:%(message)s')


        # if model_class == RealWorld:
        #    row_data['file_path'] = target_features['file_path']
        #    return [row_data]

        parameters = []
        parameter_classes = [input_param.output_parameter()
                             for input_param in self.model_class.input_parameters()]
        for parameter_class in parameter_classes:
            value = self.target_features[parameter_class.name()]
            parameter = parameter_class(value)
            parameters.append(parameter)
        fitter = self.fitter_class(self.model_class, parameters, **self.custom_fitter_config)
        logger = multiprocessing.get_logger()
        logger.info("Starting parameter fitting")
        self.fitted_parameters = fitter.run()
        logger.info("Finished parameter fitting")


    def writeResults(self):
        row_data = {}
        # row_data['Graph'] = target_features['Graph']
        row_data['Fitter'] = self.fitter_class.name()

        parameter_classes = [input_param.output_parameter()
                             for input_param in self.model_class.input_parameters()]
        for parameter_class in parameter_classes:
            value = self.target_features[parameter_class.name()]
            parameter = parameter_class(value)
            row_data['target_' + parameter_class.name()] = parameter.value

        for fitted_param in self.fitted_parameters:
            row_data[fitted_param.name()] = fitted_param.value

        averaging_iterations, total_iterations, flips = fitter.statistics()
        smoothing_iterations = total_iterations - averaging_iterations
        row_data['averaging_iterations'] = averaging_iterations
        row_data['smoothing_iterations'] = smoothing_iterations
        row_data['total_iterations'] = total_iterations
        for flip_count, param in zip(flips, parameter_classes):
            row_data['flips_' + param.name()] = flip_count
        for key, value in self.custom_fitter_config.items():
            row_data[key] = value

        with open(self.output_file, "w") as results_file:
            fieldnames = sorted(set(row_data.keys()))
            dict_writer = csv.DictWriter(results_file, fieldnames)
            dict_writer.writeheader()
            dict_writer.writerow(row_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    parser.add_argument('--model', type=str.lower,
                        choices=model_choices.keys(), required=True)
    parser.add_argument('--alpha', type=float, required=False)
    parser.add_argument('--threshold', type=float, required=False)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model_class = model_choices[args.model]
    alpha = args.alpha
    threshold = args.threshold

    fitter_class = RobbinsMonroFinal
    custom_fitter_config = {}
    if alpha is not None:
        custom_fitter_config["alpha"] = alpha
    if threshold is not None:
        custom_fitter_config["threshold"] = threshold

    with open(input_file) as input_dicts_file:
        target_features = list(csv.DictReader(input_dicts_file))[0]
    runner = ParameterFitterRunner(
        target_features, model_class, fitter_class, output_file, custom_fitter_config)
    runner.execute()
    runner.writeResults()
