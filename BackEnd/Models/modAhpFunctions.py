import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, entropy
from sklearn.metrics import roc_auc_score
from scipy.sparse import save_npz, lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm

### === STATISTICAL METRICS COMPUTATION FUNCTIONS === ###

def compute_t_test(cancer_samples, normal_samples):
    t_values, _ = ttest_ind(cancer_samples, normal_samples, axis=1, equal_var=False, nan_policy='omit')
    return np.nan_to_num(np.abs(t_values), nan=0.0)

def compute_entropy(cancer_samples, normal_samples):
    cancer_probs = np.nan_to_num(cancer_samples / (np.sum(cancer_samples, axis=1, keepdims=True) + 1e-10))
    normal_probs = np.nan_to_num(normal_samples / (np.sum(normal_samples, axis=1, keepdims=True) + 1e-10))
    entropy_scores = np.array([entropy(cancer_probs[i], normal_probs[i]) for i in range(cancer_samples.shape[0])])
    return np.nan_to_num(entropy_scores, nan=0.0)

def compute_roc_auc(cancer_samples, normal_samples):
    labels = np.hstack([np.ones(cancer_samples.shape[1]), np.zeros(normal_samples.shape[1])])
    auc_scores = np.array([
        roc_auc_score(labels, np.hstack([cancer_samples[i], normal_samples[i]])) if len(np.unique(labels)) > 1 else 0.5
        for i in range(cancer_samples.shape[0])
    ])
    return np.nan_to_num(auc_scores, nan=0.5)

def compute_snr(cancer_samples, normal_samples):
    mean_diff = np.abs(np.mean(cancer_samples, axis=1) - np.mean(normal_samples, axis=1))
    std_sum = np.std(cancer_samples, axis=1) + np.std(normal_samples, axis=1) + 1e-10
    return mean_diff / std_sum


### === DYNAMIC PAIRWISE MATRIX COMPUTATION === ###

def compute_pair(i, j, scores, max_diff):
    """Dynamically computes values for the matrix without storing in memory."""
    diff = np.abs(scores[i] - scores[j])
    norm_diff = diff / max_diff
    value = np.exp(np.clip(norm_diff * np.log(10), 0, 10))
    return value if scores[i] >= scores[j] else 1 / value

def construct_sparse_pairwise_matrix(scores):
    """Dynamically constructs a sparse pairwise matrix row by row to prevent memory overload."""
    num_genes = len(scores)
    sparse_matrix = lil_matrix((num_genes, num_genes), dtype=np.float32)

    max_diff = np.nanmax(scores) + 1e-10

    for i in tqdm(range(num_genes), desc="Computing Pairwise Matrix"):
        for j in range(i, num_genes):  # Compute only upper triangular part
            if i == j:
                sparse_matrix[i, j] = 1.0  # Diagonal is always 1
            else:
                val = compute_pair(i, j, scores, max_diff)
                sparse_matrix[i, j] = val
                sparse_matrix[j, i] = 1 / val  # Use symmetry

    return sparse_matrix.tocsr()

### === AHP SCORE COMPUTATION === ###

def compute_priority_vector(sparse_matrix):
    """Computes priority vector using eigen decomposition."""
    eigvals, eigvecs = eigs(sparse_matrix, k=1, which='LM')
    priority_vector = np.abs(eigvecs[:, 0].real)
    return priority_vector / np.sum(priority_vector)

def compute_ahp_scores(cancer_samples, normal_samples, output_prefix="AHP_results"):
    """Computes AHP scores dynamically to prevent memory leaks."""
    metrics = {
        't_test': compute_t_test(cancer_samples, normal_samples),
        'entropy': compute_entropy(cancer_samples, normal_samples),
        'roc_auc': compute_roc_auc(cancer_samples, normal_samples),
        'snr': compute_snr(cancer_samples, normal_samples)
    }
    
    pairwise_matrices = {}
    priority_vectors = {}
    
    for key, values in tqdm(metrics.items(), desc="Processing Metrics"):
        print(f"Processing pairwise matrix for {key}")
        pairwise_matrices[key] = construct_sparse_pairwise_matrix(values)
        priority_vectors[key] = compute_priority_vector(pairwise_matrices[key])

    final_scores = sum(priority_vectors.values()) / len(priority_vectors)
    np.save(f"{output_prefix}_eigen_vectors.npy", priority_vectors)

    return final_scores, priority_vectors, pairwise_matrices
