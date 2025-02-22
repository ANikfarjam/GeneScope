import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, rankdata, entropy
from sklearn.metrics import roc_auc_score
from scipy.linalg import eig
import tqdm

def compute_t_test(cancer_samples, normal_samples):
    t_values, p_value = ttest_ind(cancer_samples, normal_samples, axis=1, equal_var=False, nan_policy='omit')
    return np.nan_to_num(np.abs(t_values), nan=0.0), p_value

def compute_entropy(cancer_samples, normal_samples):
    cancer_probs = np.nan_to_num(cancer_samples / (np.sum(cancer_samples, axis=1, keepdims=True) + 1e-10), nan=0.0)
    normal_probs = np.nan_to_num(normal_samples / (np.sum(normal_samples, axis=1, keepdims=True) + 1e-10), nan=0.0)
    entropy_scores = np.array([entropy(cancer_probs[i], normal_probs[i]) if np.sum(cancer_probs[i]) > 0 and np.sum(normal_probs[i]) > 0 else 0.0 for i in range(cancer_samples.shape[0])])
    return np.nan_to_num(entropy_scores, nan=0.0)

def compute_roc_auc(cancer_samples, normal_samples):
    labels = np.hstack([np.ones(cancer_samples.shape[1]), np.zeros(normal_samples.shape[1])])
    auc_scores = np.array([
        roc_auc_score(labels, np.hstack([cancer_samples[i], normal_samples[i]])) if len(np.unique(labels)) > 1 else 0.5
        for i in range(cancer_samples.shape[0])
    ])
    return np.nan_to_num(auc_scores, nan=0.5)

def compute_wilcoxon(cancer_samples, normal_samples):
    ranks = rankdata(np.hstack([cancer_samples, normal_samples]), axis=1)
    cancer_ranks = ranks[:, :cancer_samples.shape[1]]
    normal_ranks = ranks[:, cancer_samples.shape[1]:]
    wilcoxon_scores = np.abs(np.mean(cancer_ranks, axis=1) - np.mean(normal_ranks, axis=1))
    return np.nan_to_num(wilcoxon_scores, nan=0.0)

def compute_snr(cancer_samples, normal_samples):
    mean_diff = np.abs(np.mean(cancer_samples, axis=1) - np.mean(normal_samples, axis=1))
    std_sum = np.std(cancer_samples, axis=1) + np.std(normal_samples, axis=1) + 1e-10
    return np.nan_to_num(mean_diff / std_sum, nan=0.0)

def construct_pairwise_matrix(scores):
    print("constructing pariwize Metrix!")

    num_genes = len(scores)
    matrix = np.ones((num_genes, num_genes))

    # Compute pairwise absolute differences using broadcasting
    score_diffs = np.abs(scores[:, None] - scores[None, :])

    # Normalize the differences
    max_distance = np.nanmax(score_diffs) + 1e-10
    normalized_diff = score_diffs / (max_distance + 1e-10)

    # Compute scaling factor in a numerically stable way
    scaled_diff = np.exp(np.clip(normalized_diff * np.log(10), 0, 10))

    # Construct the pairwise comparison matrix efficiently
    matrix = np.where(scores[:, None] >= scores[None, :], scaled_diff, 1 / scaled_diff)

    return np.nan_to_num(matrix, nan=1.0)



def compute_priority_vector(matrix):
    print("Coputing priority Vactor")
    eigvals, eigvecs = eig(matrix)
    max_eigval_index = np.argmax(eigvals.real)
    principal_eigvec = eigvecs[:, max_eigval_index].real
    priority_vector = np.abs(principal_eigvec) / np.sum(np.abs(principal_eigvec))
    return np.nan_to_num(priority_vector, nan=0.0), eigvals.real

def compute_ahp_scores(cancer_samples, normal_samples):
    metrics = {
        't_test': compute_t_test(cancer_samples, normal_samples)[0],  # Extract first item
        'entropy': compute_entropy(cancer_samples, normal_samples),
        'roc_auc': compute_roc_auc(cancer_samples, normal_samples),
        'wilcoxon': compute_wilcoxon(cancer_samples, normal_samples),
        'snr': compute_snr(cancer_samples, normal_samples)
    }

    pairwise_matrices = {k: construct_pairwise_matrix(v) for k, v in metrics.items()}
    priority_vectors = {k: compute_priority_vector(v) for k, v in pairwise_matrices.items()}

    eigenvalues = {k: priority_vectors[k][1] for k in priority_vectors}
    priority_vectors = {k: priority_vectors[k][0] for k in priority_vectors}

    final_scores = sum(priority_vectors.values()) / len(priority_vectors)

    return final_scores, eigenvalues

