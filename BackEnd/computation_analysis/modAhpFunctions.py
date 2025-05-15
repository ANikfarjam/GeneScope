import numpy as np
from scipy.stats import ttest_ind, entropy, ranksums
from sklearn.metrics import roc_auc_score
from scipy.sparse import save_npz, lil_matrix, csr_matrix
from tqdm import tqdm

#### === STATISTICAL METRICS COMPUTATION FUNCTIONS === ###

def compute_t_test(cancer_samples, normal_samples):
    t_values, _= ttest_ind(cancer_samples, normal_samples, axis=1, equal_var=False, nan_policy='omit')
    return np.nan_to_num(np.abs(t_values), nan=0.0)

def compute_entropy(cancer_samples, normal_samples):
    sum_cancer = np.sum(cancer_samples, axis=1, keepdims=True)
    sum_normal = np.sum(normal_samples, axis=1, keepdims=True)

    valid_indices = (sum_cancer > 0).flatten() & (sum_normal > 0).flatten()
    
    cancer_probs = np.zeros_like(cancer_samples)
    normal_probs = np.zeros_like(normal_samples)
    
    cancer_probs[valid_indices] = np.nan_to_num(cancer_samples[valid_indices] / (sum_cancer[valid_indices] + 1e-10))
    normal_probs[valid_indices] = np.nan_to_num(normal_samples[valid_indices] / (sum_normal[valid_indices] + 1e-10))

    entropy_scores = np.zeros(cancer_samples.shape[0])
    entropy_scores[valid_indices] = np.array([entropy(cancer_probs[i], normal_probs[i]) for i in np.where(valid_indices)[0]])
    
    return np.nan_to_num(entropy_scores, nan=0.0)

def compute_roc_auc(cancer_samples, normal_samples):
    labels = np.hstack([np.ones(cancer_samples.shape[1]), np.zeros(normal_samples.shape[1])])
    auc_scores = np.zeros(cancer_samples.shape[0])

    for i in range(cancer_samples.shape[0]):
        if len(np.unique(labels)) > 1:
            auc_scores[i] = roc_auc_score(labels, np.hstack([cancer_samples[i], normal_samples[i]]))
        else:
            auc_scores[i] = 0.5  # Default AUC if only one class exists

    return np.nan_to_num(auc_scores, nan=0.5)

def compute_snr(cancer_samples, normal_samples):
    mean_diff = np.abs(np.mean(cancer_samples, axis=1) - np.mean(normal_samples, axis=1))
    std_sum = np.std(cancer_samples, axis=1) + np.std(normal_samples, axis=1) + 1e-10
    return mean_diff / std_sum

def compute_wilcoxon(cancer_samples, normal_samples):
    """Computes Wilcoxon rank-sum test for each gene and returns both the statistic and p-value."""
    
    wilcoxon_scores = np.zeros(cancer_samples.shape[0])
    p_values = np.ones(cancer_samples.shape[0])
    
    for i in range(cancer_samples.shape[0]):
        if np.any(cancer_samples[i]) and np.any(normal_samples[i]):  
            result = ranksums(cancer_samples[i], normal_samples[i])
            wilcoxon_scores[i] = result.statistic
            p_values[i] = result.pvalue
    
    return np.nan_to_num(np.abs(wilcoxon_scores), nan=0.0), np.nan_to_num(p_values, nan=1.0)

### === FUNCTION TO UPLOAD PAIRWISE MATRIX TO FIREBASE === ###
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
    
### === COMPUTE AHP SCORES === ###
def compute_ahp_scores(cancer_samples, normal_samples):
    t_test = compute_t_test(cancer_samples, normal_samples)
    wilcoxon_scores, wilcoxon_p = compute_wilcoxon(cancer_samples, normal_samples)

    metrics = {
        't_test': t_test,
        'entropy': compute_entropy(cancer_samples, normal_samples),
        'roc_auc': compute_roc_auc(cancer_samples, normal_samples),
        'Wilcoxon': wilcoxon_scores,
        'Wilcoxon_p': wilcoxon_p,
        'snr': compute_snr(cancer_samples, normal_samples)
    }

    return metrics
