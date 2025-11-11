#!/usr/bin/env python3
import os
import glob
import run
from experimentsUtils import setup

setup(is_colab=True)

all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

alpha_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# run parameter fitting

run.group("fit_parameters_alpha")
for model in all_models:
    run.add(
        "fit_parameters_alpha_[[model]]",
        "python3 src/fit_parameters.py --model [[model]] --alpha [[alpha]] output_data/target_params/[[model]]/[[input]].csv [[file]]",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*")],
            "alpha": alpha_range,
            "file": "output_data/fitted_params_ablation_alpha/[[model]]/[[input]]_alpha=[[alpha]].csv",
            "model": model,
        },
        creates_file="[[file]]",
    )

# Generate samples based on fitted parameters
run.run()

run.group("fitted_sample_and_measure_alpha")

for model in all_models:
    run.add(
        "fitted_sample_and_measure_alpha_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params_ablation_alpha/[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params_ablation_alpha/{model}/*")],
            "file": "output_data/fitted_features_ablation_alpha/[[model]]/[[input]].csv",
        },
        creates_file="[[file]]",
    )
run.run()


run.group("fit_parameters_threshold")
threshold_range = [0.001, 0.005, 0.01, 0.05, 0.1]

for model in all_models:
    run.add(
        "fit_parameters_threshold_[[model]]",
        "python3 src/fit_parameters.py --model [[model]] --threshold [[threshold]] output_data/target_params/[[model]]/[[input]].csv [[file]]",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*")],
            "threshold": threshold_range,
            "file": "output_data/fitted_params_ablation_threshold/[[model]]/[[input]]_threshold=[[threshold]].csv",
            "model": model,
        },
        creates_file="[[file]]",
    )

# Generate samples based on fitted parameters
run.run()

run.group("fitted_sample_and_measure_threshold")

for model in all_models:
    run.add(
        "fitted_sample_and_measure_threshold_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params_ablation_threshold/[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params_ablation_threshold/{model}/*")],
            "file": "output_data/fitted_features_ablation_threshold/[[model]]/[[input]].csv",
        },
        creates_file="[[file]]",
    )
run.run()


# names of all output stats (might have changed by previous runs)
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/*/")
]

######################################################################
# some postprocessing
run.group("post")

# merge csv
run.add(
    "merge_csv",
    "scripts/merge-csv.sh [[output]]",
    {"output": output_names},
    creates_file="[[output]].csv",
)

# merge csv force
run.add(
    "merge_csv_force",
    "scripts/merge-csv.sh [[output]]",
    {"output": output_names},
)

run.run()
