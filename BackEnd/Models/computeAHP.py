import modAhpFunctions as ahpFunctions
import pandas as pd
import numpy as np
import pickle

# Load datasets
mutated_df = pd.read_csv("../../data/ModelDataSets/cancerExpressions.csv", index_col=0).T  # Cancer data
benign_df = pd.read_csv("../../data/ModelDataSets/helthyExpressions.csv", index_col=0).T  # Healthy data

# Fill NaN values
mutated_df.fillna(0, inplace=True)
benign_df.fillna(0, inplace=True)

# Ensure both datasets have the same number of samples
num_samples = min(mutated_df.shape[1], benign_df.shape[1])

# Compute AHP scores
final_scores, priority_vectors, pairwise_matrices = ahpFunctions.compute_ahp_scores(
    mutated_df.iloc[:, :num_samples].values, 
    benign_df.iloc[:, :num_samples].values  # Corrected from DCIS_dataset
)

# Convert priority vectors into a DataFrame
results_df = pd.DataFrame({"Gene": mutated_df.index, **priority_vectors, 'Scores':final_scores})

# Display and save results
print('Resulting DF:\n', results_df.head())
results_df.to_csv("Mod_ahp_scores.csv", index=False)
# np.save("pairwise_matrices.npy", pairwise_matrices)
with open('pwm_pickl.pkl', 'wb') as f:
    pickle.dump(pairwise_matrices, f)