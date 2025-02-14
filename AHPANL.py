import numpy as np
from scipy.stats import ttest_ind, wilcoxon, entropy
from sklearn.metrics import roc_auc_score

# Functions for AHP Analysis

def compute_t_test(sample1, sample2):
    """
    Computes the two-sample t-test score between two independent samples.
    - **Purpose**: Identifies statistically significant differences in gene expression between two groups.
    - **Output**: t-score and p-value indicating statistical significance.
    """
    t_score, _ = ttest_ind(sample1, sample2, equal_var=False)
    return t_score

def compute_entropy(sample1, sample2):
    """
    Computes the entropy for two independent samples.
    - **Purpose**: Measures disorder in gene expression; higher entropy indicates better class separation.
    - **Output**: Entropy values for both samples.
    """
    entropy1 = entropy(np.histogram(sample1, bins=10)[0])
    entropy2 = entropy(np.histogram(sample2, bins=10)[0])
    return entropy1, entropy2

def compute_wilcoxon(sample1, sample2):
    """
    Computes the Wilcoxon rank-sum test between two independent samples.
    - **Purpose**: A non-parametric test ranking genes based on median expression differences.
    - **Output**: Wilcoxon statistic and p-value.
    """
    stat, p_value = wilcoxon(sample1, sample2,zero_method="pratt", correction=True)
    return stat, p_value

def compute_snr(sample1, sample2):
    """
    Computes the Signal-to-Noise Ratio (SNR) between two samples.
    - **Purpose**: Measures the difference in mean expression levels relative to standard deviation.
    - **Output**: SNR value, indicating how well a gene differentiates between two conditions.
    """
    mean_diff = np.mean(sample1) - np.mean(sample2)
    std_sum = np.std(sample1) + np.std(sample2)
    return mean_diff / std_sum

# def compute_ahp_weighted_ranking(t_scores, entropy1, entropy2, snrs):
#     """
#     Combines statistical rankings using a modified AHP approach.
#     - **Purpose**: Integrates different statistical measures into a single weighted ranking system.
#     - **Output**: A final ranking score for gene significance.
#     """
#     scores = np.array([t_scores, entropy1, entropy2, snrs])
#     normalized_scores = scores / np.sum(scores, axis=1, keepdims=True)
#     weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weighting for now
#     final_rankings = np.dot(weights, normalized_scores)
#     return final_rankings

def compute_ahp_weighted_ranking(t_scores, entropy1, entropy2, snrs):
    """
    Combines statistical rankings using a modified AHP approach.
    - Purpose: Integrates different statistical measures into a single weighted ranking system.
    - Output: A final ranking score for gene significance.
    """
    scores = np.array([t_scores, entropy1, entropy2, snrs]).reshape(1, -1)  # Convert to 2D

    # Compute sum, check if it is zero
    score_sum = np.sum(scores, axis=1, keepdims=True)
    score_sum[score_sum == 0] = 1  # Prevent division by zero

    normalized_scores = scores / score_sum  # Normalize safely
    weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weighting for now
    final_rankings = np.dot(weights, normalized_scores.T)  # Use .T to align dimensions correctly

    return final_rankings[0]

# def compute_ahp_weighted_ranking(t_scores, entropies, wilcoxon_stats, snrs):
#     """
#     Combines statistical rankings using a modified AHP approach.
#     - **Purpose**: Integrates different statistical measures into a single weighted ranking system.
#     - **Output**: A final ranking score for gene significance.
#     """
#     scores = np.array([t_scores, entropies, wilcoxon_stats, snrs])
#     normalized_scores = scores / np.sum(scores, axis=1, keepdims=True)
#     weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weighting for now
#     final_rankings = np.dot(weights, normalized_scores)
#     return final_rankings