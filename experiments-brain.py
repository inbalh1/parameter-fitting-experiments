#!/usr/bin/env python3
import os
import glob
import run
from experimentsUtils import setup

setup(is_colab=True)

# TODO: should unzip the brain network data first (from python or colab?)
# Should we save the unzipped infomation?

# reduce largest connected component and convert to different format
clean_input_path = 'input_data/brain'
clean_output_path = 'input_data/clean/brain'

# Remove zip files from input files
all_items = glob.glob(f"{clean_input_path}/*")
zip_items = glob.glob(f"{clean_input_path}/*.zip")
non_zip_items = list(set(all_items) - set(zip_items))

edges_files = glob.glob(f"{clean_input_path}/*.edges")

def find_edges_file(folder):
    """Return the path to the .edges file inside a two-level-deep folder."""
    matches = glob.glob(os.path.join(folder, "*", "*.edges"))
    return matches[0] if matches else None

def get_graph_name(edges_file_path):
    f = edges_file_path
    return os.path.splitext(os.path.basename(f))[0]

run.add(
    "clean_graphs",
    f"python3 src/clean_graphs.py {clean_input_path}/[[input]].edges [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in edges_files],
        "file": f"{clean_output_path}/[[input]].networkit"},
    creates_file="[[file]]",
)

run.run()

measure_input_path = clean_output_path
measure_output_path = 'output_data/target_params/brain'

run.add(
    "measure_target_features",
    f"python3 src/measure_target_params.py {measure_input_path}/[[input]].networkit [[file]]",
    {
        "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{measure_input_path}/*")],
        "file": f"{measure_output_path}/[[input]].csv",
    },
    creates_file="[[file]]",
)
run.run()

all_models = ["erdos-renyi", "chung-lu-pl", "girg-1d"]
fit_input_path = measure_output_path
fit_output_path = 'output_data/fitted_params/brain'

for model in all_models:
    run.add(
        "fit_parameters_[[model]]",
        f"python3 src/fit_parameters.py --model [[model]] {fit_input_path}/[[input]].csv [[file]]",
        {
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{fit_input_path}/*")],
            "file": f"{fit_output_path}/[[model]]/[[input]].csv",
            "model": model
        },
        creates_file="[[file]]",
    )

run.run()


features_input_path = fit_output_path
features_output_path = 'output_data/fitted_features/brain'

for model in all_models:
    run.add(
        "fitted_sample_and_measure_[[model]]",
        f"python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --samples [[samples]] --input_file {features_input_path}/[[model]]/[[input]].csv --output_file [[file]]",
        {
            "model": model,
            "seed": [9381],
            "samples": 50,
            "input": [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{features_input_path}-{model}/*")],
            "file": f"{features_output_path}/[[model]]/[[input]].csv",
        },
        creates_file="[[file]]",
    )

run.run()

# names of all output stats (might have changed by previous runs)
# TODO: does it work like this?
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/brain/*/")
] + [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/brain/")
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
