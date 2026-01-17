#!/usr/bin/env python3
import os
import glob
import run
import multiprocessing

run.use_cores(multiprocessing.cpu_count() - 2)

# The goal is to compare the erdos-renyi results for the optimal solution of MLE


all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

# Notice the use of the extended model - is just because we assume for the MLE that we've got number of edges as well...
model = "erdos-renyi"
run.add(
    "fit_parameters_MLE_[[model]]",
    "python3 src/MLE.py --model [[model]]-extended output_data/target_params/real-world/[[input]].csv [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/real-world/*")],
        "file": "output_data/fitted_params/MLE/real-world-[[model]]/[[input]].csv",
        "model": model
    },
    creates_file="[[file]]",
)

run.run()

# Generate samples based on fitted parameters
# Not sure whether in here I need the model extended
run.add(
    "fitted_sample_and_measure_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params/MLE/real-world-[[model]]/[[input]].csv --output_file [[file]]",
    {
        "model": "erdos-renyi",
        "seed": [9381],
        "samples": 50,
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params/MLE/real-world-{model}/*")],
        "file": "output_data/fitted_features/MLE/real-world-[[model]]/[[input]].csv",
    },
    creates_file="[[file]]",
)
run.run()


# names of all output stats (might have changed by previous runs)
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/MLE/*/")
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
