#!/usr/bin/env python3
import os
import glob
import run
from experimentsUtils import setup

setup(is_colab=True)

# reduce largest connected component and convert to different format
run.add(
    "clean_graphs",
    "python3 src/clean_graphs.py input_data/konect/[[input]] [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"input_data/konect/*")],
        "file": "input_data/clean/real-world/[[input]].networkit"},
    creates_file="[[file]]",
)

run.run()

run.add(
    "measure_target_features",
    "python3 src/measure_target_params.py input_data/clean/real-world/[[input]].networkit [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"input_data/clean/real-world/*")],
        "file": "output_data/target_params/real-world/[[input]].csv",
    },
    creates_file="[[file]]",
)
run.run()

all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

for model in all_models:
    run.add(
        "fit_parameters_[[model]]",
        "python3 src/fit_parameters.py --model [[model]] output_data/target_params/real-world/[[input]].csv [[file]]",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/real-world/*")],
            "file": "output_data/fitted_params/real-world-[[model]]/[[input]].csv",
            "model": model
        },
        creates_file="[[file]]",
    )

run.run()

for model in all_models:
    run.add(
        "fitted_sample_and_measure_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params/real-world-[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params/real-world-{model}/*")],
            "file": "output_data/fitted_features/real-world-[[model]]/[[input]].csv",
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
