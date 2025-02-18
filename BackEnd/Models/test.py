import BackEnd.Models.ahpFunctions as ahpFunctions 
import pandas as pd
import numpy as np

#load datasets
mutated_df = pd.read_csv("./data/mutatedDataSet.CSV", index_col=0)  # Cancer data
benign_df = pd.read_csv("./data/normalDataset.CSV", index_col=0)  # Healthy data
# Compute AHP scores
#top_genes, ahp_scores = ahp.ahp_gene_selection(mutated_df.values, benign_df.values, top_n=mutated_df.shape[0])

# Create DataFrame with results
results_df = pd.DataFrame({
    "Gene": mutated_df.index.tolist(),  # Ensure it works correctly
    "T_Score": ahpFunctions.compute_t_test(mutated_df.values, benign_df.values),
    "Entropy": ahpFunctions.compute_entropy(mutated_df.values, benign_df.values),
    "ROC_AUC": ahpFunctions.compute_roc_auc(mutated_df.values, benign_df.values),
    "Wilcoxon": ahpFunctions.compute_wilcoxon(mutated_df.values, benign_df.values),
    "SNR": ahpFunctions.compute_snr(mutated_df.values, benign_df.values),
    "AHP_Score": ahpFunctions.compute_ahp_weighted_ranking(mutated_df.values, benign_df.values)
})

print(results_df.head())