#!/usr/bin/env python3
import os
import glob
import run
import multiprocessing

run.use_cores(multiprocessing.cpu_count() - 2)

# The goal is to compare the erdos-renyi results for the optimal solution of MLE

# The same sample_and_measure experiment from before
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



run.run()


all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]

# Notice the use of the extended model - is just because we assume for the MLE that we've got number of edges as well...
model = "erdos-renyi"
run.add(
    "fit_parameters_MLE_[[model]]",
    "python3 src/MLEFitter.py --model [[model]]-extended output_data/target_params/[[model]]/[[input]].csv [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*")],
        "file": "output_data/fitted_params/MLE/[[model]]/[[input]].csv",
        "model": model
    },
    creates_file="[[file]]",
)

run.run()


all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]
for model in all_models:
    run.add(
        "fit_parameters_smac_[[model]]",
        "python3 src/smacFitter.py --model [[model]] output_data/target_params/[[model]]/[[input]].csv [[file]] > output.txt",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/target_params/{model}/*")],
            "file": "output_data/fitted_params/smac/[[model]]/[[input]].csv",
            "model": model
        },
        creates_file="[[file]]",
    )

run.run()


# Generate samples based on fitted parameters
# Not sure whether in here I need the model extended
all_fitters = ['MLE', 'smac']
for fitter in all_fitters:
    run.add(
        name="fitted_sample_and_measure_[[fitter]]_[[model]]",
        command="python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file output_data/fitted_params/[[fitter]]/[[model]]/[[input]].csv --output_file [[file]]",
        arguments_descr={
            "fitter": fitter,
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"output_data/fitted_params/{fitter}/{model}/*")],
            "file": "output_data/fitted_features/[[fitter]]/[[model]]/[[input]].csv",
        },
        creates_file="[[file]]",
    )
run.run()


# names of all output stats (might have changed by previous runs)
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/*/*/")
]

# TODO: find a way to pass fitter name
# Only for smac
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
