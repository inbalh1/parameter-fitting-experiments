import csv
import argparse
from pathlib import Path
import sys

sys.path.insert(0, '../')
from models import *
from parameters import *


class GraphSamplerAndMeasurer():
    def __init__(self, param_dict, model_class, seed: int | None, samples: int, combine: bool, output_file):
        self.param_dict = param_dict
        self.model_class = model_class
        self.seed = seed
        self.samples = samples
        self.combine = combine
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        networkit.engineering.setNumberOfThreads(1)

    def execute(self):
        if self.seed is None:
            self.seed = 1234
        random.seed(self.seed)

        input_params = []
        for input_param_class in self.model_class.input_parameters():
            input_params.append(input_param_class(
                self.param_dict[input_param_class.name()]))

        output = {}
        output.update(self.param_dict)
        # output["graph"] = self.output_file.stem
        fieldnames = sorted(set(output.keys()))
        dict_writer = csv.DictWriter(sys.stdout, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerow(output)

        all_feature_vals = []
        for _ in range(self.samples):
            iteration_seed = random.randrange(10000)
            generator = self.model_class(*input_params, seed=iteration_seed)
            g = generator.generate()

            all_input_params = NumberOfVertices, NumberOfEdges, AverageDegree, PowerlawBeta, Temperature
            all_features = [input_param.output_parameter()
                            for input_param in all_input_params]
            features = [feature.measure(g) for feature in all_features]
            feature_vals = {}
            for feature in features:
                feature_vals[feature.name()] = feature.value
            feature_vals["seed"] = iteration_seed
            all_feature_vals.append(feature_vals)

        fieldnames = sorted(set(all_feature_vals[0].keys()))
        if self.combine:
            all_means = {}
            for fieldname in fieldnames:
                if fieldname == "seed":
                    continue
                vals = [feature_vals[fieldname]
                        for feature_vals in all_feature_vals]
                mean = sum(vals) / len(vals)
                all_means[fieldname] = mean
            fieldnames = sorted(set(all_means.keys()))
            with open(self.output_file, "w") as results_file:
                dict_writer = csv.DictWriter(results_file, fieldnames)
                dict_writer.writeheader()
                dict_writer.writerow(all_means)
        else:
            with open(self.output_file, "w") as results_file:
                dict_writer = csv.DictWriter(results_file, fieldnames)
                dict_writer.writeheader()
                for feature_vals in all_feature_vals:
                    dict_writer.writerow(feature_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_file', type=str)
    model_choices = {model.name().lower(): model for model in ALL_MODELS}
    parser.add_argument('--model', type=str.lower,
                        choices=model_choices.keys(), required=True)
    parser.add_argument('--seed', required=False, type=int)
    parser.add_argument('--samples', required=True, type=int)
    parser.add_argument(
        '--combine', action="store_true", default=False)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)

    args, unknown = parser.parse_known_args()
    input_file = args.input_file
    output_file = args.output_file
    seed = args.seed
    samples = args.samples
    combine = args.combine

    model_class = model_choices[args.model]

    model_param_parser = argparse.ArgumentParser()
    for param_class in model_class.input_parameters():
        model_param_parser.add_argument(
            f'--{param_class.name()}', type=str)
    model_args = model_param_parser.parse_args(unknown)

    if input_file is not None:
        with open(input_file, "r") as input_dicts_file:
            param_dict = list(csv.DictReader(input_dicts_file))[0]
    else:
        param_dict = vars(model_args)

    runner = GraphSamplerAndMeasurer(
        param_dict, model_class, seed, samples, combine, output_file)
    runner.execute()
