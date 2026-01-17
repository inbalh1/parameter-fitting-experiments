#!/usr/bin/env python3
import os
import glob
import run
import multiprocessing

# Try using more than 1 core
run.use_cores(max(1, multiprocessing.cpu_count() - 2))

run.add(
    "sample_and_measure_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --t [[t]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": 10000,
        "d": [2, 3, 5, 7, 10],
        "beta": [2.1, 2.25, 2.4, 2.55, 2.75, 3.00, 3.35, 3.9, 5.1, 25.0],
        "t": [0.01, 0.4, 0.53, 0.62, 0.7, 0.76, 0.82, 0.88, 0.94, 0.9999],
        "seed": [993],
        "samples": 50,
        "name": "[[model]]_n=[[n]]_d=[[d]]_t=[[t]]_beta=[[beta]]_seed=[[seed]]",
        "file": "output_data/target_params/[[model]]/[[name]].csv",
        "model": "girg-1d"
    },
    stdout_file="output_data/attributes/[[model]]/[[name]].csv"
)


run.add(
    "sample_and_measure_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": list(range(1000, 10000+1, 500)),
        "d": list(range(2, 10+1)),
        "seed": [993],
        "samples": 50,
        "name": "[[model]]_n=[[n]]_d=[[d]]_seed=[[seed]]",
        "file": "output_data/target_params/[[model]]/[[name]].csv",
        "model": "erdos-renyi"
    },
    stdout_file="output_data/attributes/[[model]]/[[name]].csv"
)


run.add(
    "sample_and_measure_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": list(range(1000, 10000+1, 1000)),
        "d": list(range(2, 10+1, 2)),
        "beta": [2.1, 2.25, 2.4, 2.55, 2.75, 3.00, 3.35, 3.9, 5.1, 25.0],
        "seed": [993],
        "samples": 50,
        "name": "[[model]]_n=[[n]]_d=[[d]]_beta=[[beta]]_seed=[[seed]]",
        "file": "output_data/target_params/[[model]]/[[name]].csv",
        "model": "chung-lu-pl"
    },
    stdout_file="output_data/attributes/[[model]]/[[name]].csv"
)

run.run()


all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

# run parameter fitting
for model in all_models:
    run.add(
        "fit_parameters_[[model]]",
        "python3 src/fit_parameters.py --model [[model]] output_data/target_params/[[model]]/[[input]].csv [[file]]",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*")],
            "file": "output_data/fitted_params/[[model]]/[[input]].csv",
            "model": model
        },
        creates_file="[[file]]",
    )

# Generate samples based on fitted parameters
run.run()
for model in all_models:
    run.add(
        "fitted_sample_and_measure_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params/[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params/{model}/*")],
            "file": "output_data/fitted_features/[[model]]/[[input]].csv",
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
