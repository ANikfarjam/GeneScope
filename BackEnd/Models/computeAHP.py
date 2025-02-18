from rich.progress import Progress
import BackEnd.Models.ahpFunctions as ahpFunctions
import pandas as pd
import numpy as np

# Load datasets
mutated_df = pd.read_csv("../../data/mutatedDataSet.CSV", index_col=0)  # Cancer data
benign_df = pd.read_csv("../../data/normalDataset.CSV", index_col=0)  # Healthy data
mutated_df.fillna(0, inplace=True)
benign_df.fillna(0, inplace=True)
num_samples = min(mutated_df.shape[1], benign_df.shape[1])

# Use progress bar to track computations
results = {}

with Progress() as progress:
    task = progress.add_task("[cyan]Computing AHP scores...", total=8)
    t_test, p_value = ahpFunctions.compute_t_test(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    results["T_Score"] = t_test
    results["p_values"] = p_value
    progress.update(task, advance=1)

    results["Entropy"] = ahpFunctions.compute_entropy(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    progress.update(task, advance=1)

    results["ROC_AUC"] = ahpFunctions.compute_roc_auc(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    progress.update(task, advance=1)

    results["Wilcoxon"] = ahpFunctions.compute_wilcoxon(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    progress.update(task, advance=1)

    results["SNR"] = ahpFunctions.compute_snr(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    progress.update(task, advance=1)

    scores, eigenvalues = ahpFunctions.compute_ahp_scores(mutated_df.iloc[:, :num_samples].values, benign_df.iloc[:, :num_samples].values)
    progress.update(task, advance=1)
    
    results["AHP_Score"] = scores
    progress.update(task, advance=1)

    results["Eigen_Values"]= eigenvalues
    progress.update(task, advance=1)

# Create DataFrame with results
results_df = pd.DataFrame({"Gene": mutated_df.index.tolist(), **results})

print(results_df.head())
results_df.to_csv("ahp.csv", index=False)
