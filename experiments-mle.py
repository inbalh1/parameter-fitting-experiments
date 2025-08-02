#!/usr/bin/env python3
import os
import glob
import run
import multiprocessing
import random

run.use_cores(multiprocessing.cpu_count() - 2)


# Constants
num_of_samples = 50
base_output_path = "output_data/target_params/mle"
# Train constants
train_seed = [321]
n_range = (500, 12000)
d_range=(2, 12)
beta_low_range = (1.8, 5.5)
beta_weights = [0.9, 0.1] 
beta_high_range = (5.5, 30)
t_low_range = (0, 0.9)
t_high_range = (0.9, 0.9999)
t_weights = [0.5, 0.5]


# Test constants
test_seed = [993]



def generate_random_params(n_range=None, d_range=None, beta=False, t=False,
                            total_combinations=500, seed=None):
    if seed is not None:
        random.seed(seed)

    seen = set()

    while len(seen) < total_combinations:

        combo = []
        if n_range:
            n = random.randint(*n_range)
            combo.append(n)
        if d_range:
            d = random.randint(*d_range)
            combo.append(d)
        if beta:
            selected_beta_range = random.choices(
                [beta_low_range, beta_high_range],
                weights=beta_weights,
                k=1)[0]
            # Sample beta from the selected range
            beta = round(random.uniform(*selected_beta_range), 3)
            combo.append(beta)
        if t:
            selected_t_range = random.choices(
                [t_low_range, t_high_range],
                weights=t_weights,
                k=1)[0]
            # Sample t from the selected range
            t = round(random.uniform(*selected_t_range), 4)
            combo.append(t)

        combo_tuple = tuple(combo)
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            yield combo_tuple




girg_generator = generate_random_params(n_range, d_range,
                beta=True, t=True, total_combinations=500, seed=321)
for param_comb in girg_generator:
    n, d, beta, t = param_comb
    run.add(
        "train_data_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --t [[t]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
        {
            "n": n,
            "d": d,
            "beta": beta,
            "t": t,
            "seed": train_seed,
            "samples": num_of_samples,
            "name": "[[model]]_n=[[n]]_d=[[d]]_t=[[t]]_beta=[[beta]]_seed=[[seed]]",
            "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
            "model": "girg-1d"
        },
        stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
    )

chung_lu_pl_generator = generate_random_params(n_range, d_range, beta=True, total_combinations=500, seed=321)
for param_comb in chung_lu_pl_generator:
    n, d, beta = param_comb
    run.add(
        "train_data_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
        {
            "n": n,
            "d": d,
            "beta": beta,
            "seed": train_seed,
            "samples": num_of_samples,
            "name": "[[model]]_n=[[n]]_d=[[d]]_beta=[[beta]]_seed=[[seed]]",
            "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
            "model": "chung-lu-pl"
        },
        stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
    )

erdos_renyi_generator = generate_random_params(n_range, d_range, total_combinations=500, seed=321)
for param_comb in erdos_renyi_generator:
    n, d = param_comb
    run.add(
        "train_data_[[model]]",
        "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --samples [[samples]] --combine --output_file [[file]]",
        {
            "n": n,
            "d": d,
            "seed": train_seed,
            "samples": num_of_samples,
            "name": "[[model]]_n=[[n]]_d=[[d]]_seed=[[seed]]",
            "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
            "model": "erdos-renyi"
        },
        stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
    )


run.add(
    "test_data_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --t [[t]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": 10000,
        "d": [2, 3, 5, 7, 10],
        "beta": [2.1, 2.25, 2.4, 2.55, 2.75, 3.00, 3.35, 3.9, 5.1, 25.0],
        "t": [0.01, 0.4, 0.53, 0.62, 0.7, 0.76, 0.82, 0.88, 0.94, 0.9999],
        "seed": test_seed,
        "samples": num_of_samples, 
        "name": "[[model]]_n=[[n]]_d=[[d]]_t=[[t]]_beta=[[beta]]_seed=[[seed]]",
        "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
        "model": "girg-1d"
    },
    stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
)


run.add(
    "test_data_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": list(range(1000, 10000+1, 500)),
        "d": list(range(2, 10+1)),
        "seed": test_seed,
        "samples": num_of_samples,
        "name": "[[model]]_n=[[n]]_d=[[d]]_seed=[[seed]]",
        "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
        "model": "erdos-renyi"
    },
    stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
)


run.add(
    "test_data_[[model]]",
    "python3 src/sample_and_measure.py --model [[model]] --seed [[seed]] --n [[n]] --d [[d]] --beta [[beta]] --samples [[samples]] --combine --output_file [[file]]",
    {
        "n": list(range(1000, 10000+1, 1000)),
        "d": list(range(2, 10+1, 2)),
        "beta": [2.1, 2.25, 2.4, 2.55, 2.75, 3.00, 3.35, 3.9, 5.1, 25.0],
        "seed": test_seed,
        "samples": num_of_samples,
        "name": "[[model]]_n=[[n]]_d=[[d]]_beta=[[beta]]_seed=[[seed]]",
        "file": os.path.join(base_output_path, "[[model]]/[[name]].csv"),
        "model": "chung-lu-pl"
    },
    stdout_file="output_data/attributes/mle/[[model]]/[[name]].csv"
)

run.run()



# names of all output stats (might have changed by previous runs)
output_names = [
    os.path.dirname(dir) for dir in glob.glob("output_data/*/*/")
]

######################################################################
