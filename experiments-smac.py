#!/usr/bin/env python3
import os
import glob
import run
import multiprocessing

# Try using more than 1 core
run.use_cores(max(1, multiprocessing.cpu_count() - 2))

# To create target params - run from the other experiments files (synthetic data: experiments-models, real graphs: experiments-konect)


all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

# Fit parameters - currently this isn't performed here
# for model in all_models:
#    run.add(
#        "fit_parameters_smac_[[model]]",
#        "python3 src/smacFitter.py --model [[model]] output_data/target_params/[[model]]/[[input]].csv [[file]]",
#        {
#            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*.csv")],
#            "file": "output_data/fitted_params/smac/[[model]]/[[input]].csv",
#            "model": model
#        },
#        creates_file="[[file]]",
#    )

run.run()

for model in all_models:
    run.add(
        "fitted_sample_and_measure_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params/smac/[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params/smac/{model}/*.csv")],
            "file": "output_data/fitted_features/smac/[[model]]/[[input]].csv",
        },
        creates_file="[[file]]",
    )
run.run()


# names of all output stats (might have changed by previous runs) - for SMAC
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/smac/*/")
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
