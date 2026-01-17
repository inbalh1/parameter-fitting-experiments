#!/bin/bash

# List of datasets
datasets=("train_data_erdos-renyi" "train_data_chung-lu-pl" "train_data_girg-1d")

# Loop over datasets and run the script
for dataset in "${datasets[@]}"; do
    echo "Running experiment with $dataset..."
    python3 experiments-mle.py "$dataset"
done

echo "All experiments completed."
