import pandas as pd
import numpy as np

# Read data
tbl_true = pd.read_csv("output_data/target_params/erdos-renyi.csv")
tbl_fitted = pd.read_csv("output_data/fitted_features/MLE/erdos-renyi.csv")
tbl_fitting_process = pd.read_csv("output_data/fitted_params/MLE/erdos-renyi.csv")
tbl_generation_params = pd.read_csv("output_data/attributes/erdos-renyi.csv")

# Aggregate numeric columns by 'graph' (compute mean)
tbl_fitted = tbl_fitted.groupby("graph").mean().reset_index()

# Merge tables
tbl = tbl_true.merge(tbl_fitted, on="graph", suffixes=("_true", "_fitted"))
tbl = tbl_generation_params.merge(tbl, on="graph")
tbl = tbl.merge(tbl_fitting_process, on="graph", suffixes=("", "_fitting"))

# Compute absolute difference for 'n'
tbl["n_diff"] = abs(tbl["n_true"] - tbl["n_fitted"])

print (tbl.shape)
print(tbl[tbl["n_diff"] > 10].loc[:, ['n', 'target_n', 'd', 'target_d', "n_diff"]].sort_values(ascending=False, by="n_diff").head(30))

# Compute Pearson correlation
pearson_corr = np.corrcoef(tbl["n_true"], tbl["n_fitted"])[0, 1]

# Compute mean absolute difference (MAD)
n_mean_diff = tbl["n_diff"].mean()

# Print results
print(f"n Pearson correlation: {pearson_corr}")
print(f"n mean absolute difference: {n_mean_diff}")
