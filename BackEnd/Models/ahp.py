import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, rankdata
from sklearn.metrics import roc_auc_score

def compute_t_test(cancer_samples, normal_samples):
    t_values, _ = ttest_ind(cancer_samples, normal_samples, axis=1, equal_var=False)
    return np.abs(t_values)

def compute_entropy(cancer_samples, normal_samples):
    variances = np.var(np.hstack([cancer_samples, normal_samples]), axis=1)
    return variances

def compute_roc_auc(cancer_samples, normal_samples):
    labels = np.hstack([np.ones(cancer_samples.shape[1]), np.zeros(normal_samples.shape[1])])
    auc_scores = [roc_auc_score(labels, np.hstack([cancer_samples[i], normal_samples[i]])) for i in range(cancer_samples.shape[0])]
    return np.array(auc_scores)

def compute_wilcoxon(cancer_samples, normal_samples):
    ranks = rankdata(np.hstack([cancer_samples, normal_samples]), axis=1)
    cancer_ranks = ranks[:, :cancer_samples.shape[1]]
    normal_ranks = ranks[:, cancer_samples.shape[1]:]
    return np.abs(np.mean(cancer_ranks, axis=1) - np.mean(normal_ranks, axis=1))

def compute_snr(cancer_samples, normal_samples):
    mean_diff = np.abs(np.mean(cancer_samples, axis=1) - np.mean(normal_samples, axis=1))
    std_sum = np.std(cancer_samples, axis=1) + np.std(normal_samples, axis=1)
    return mean_diff / (std_sum + 1e-10)

def ahp_gene_selection(cancer_samples, normal_samples, top_n=10):
    metrics = {
        't_test': compute_t_test(cancer_samples, normal_samples),
        'entropy': compute_entropy(cancer_samples, normal_samples),
        'roc_auc': compute_roc_auc(cancer_samples, normal_samples),
        'wilcoxon': compute_wilcoxon(cancer_samples, normal_samples),
        'snr': compute_snr(cancer_samples, normal_samples)
    }
    
    # Normalize metrics
    for key in metrics:
        metrics[key] = rankdata(metrics[key]) / len(metrics[key])
    
    # Compute final AHP score (equal weights assumed)
    scores = sum(metrics.values()) / len(metrics)
    
    # Select top N genes
    top_genes = np.argsort(scores)[-top_n:][::-1]
    return top_genes, scores[top_genes]
def compute_ahp_weighted_ranking(cancer_samples, normal_samples):
    """
    Computes AHP weighted ranking for gene significance.
    """
    # Compute metrics
    t_test = compute_t_test(cancer_samples, normal_samples)
    entropy = compute_entropy(cancer_samples, normal_samples)
    roc_auc = compute_roc_auc(cancer_samples, normal_samples)
    wilcoxon = compute_wilcoxon(cancer_samples, normal_samples)
    snr = compute_snr(cancer_samples, normal_samples)

    # Stack into 2D array (rows = metrics, columns = genes)
    scores = np.vstack([t_test, entropy, roc_auc, wilcoxon, snr])  # Shape: (5, num_genes)

    print("Before Normalization:")
    print("NaN in scores:", np.isnan(scores).sum())  # Check NaN before normalization

    # Check for NaN values in individual metrics
    for i, metric_name in enumerate(["T-Test", "Entropy", "ROC_AUC", "Wilcoxon", "SNR"]):
        print(f"{metric_name} NaN count: {np.isnan(scores[i]).sum()}")

    # Min-Max Normalization (Avoid division by zero)
    min_vals = np.nanmin(scores, axis=1, keepdims=True)
    max_vals = np.nanmax(scores, axis=1, keepdims=True)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Prevent division by zero

    normalized_scores = (scores - min_vals) / range_vals  # Normalize safely

    print("After Normalization:")
    print("NaN in normalized scores:", np.isnan(normalized_scores).sum())

    # Define equal weights
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Adjustable weights

    # Compute final AHP ranking
    final_rankings = np.dot(weights, normalized_scores)

    print("Final AHP Rankings NaN count:", np.isnan(final_rankings).sum())

    return np.nan_to_num(final_rankings)  # Ensure no NaN in final output


# Example Usage:
# cancer_samples and normal_samples are NumPy arrays where rows are genes and columns are samples.
# cancer_samples = np.random.rand(1000, 50)  # Example data (1000 genes, 50 samples)
# normal_samples = np.random.rand(1000, 50)
# top_genes, scores = ahp_gene_selection(cancer_samples, normal_samples, top_n=20)
# print("Top genes:", top_genes)
