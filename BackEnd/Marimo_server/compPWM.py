from rich.progress import Progress
import numpy as np
from scipy.sparse.linalg import eigs
import modAhpFunctions
import pickle



metrics = np.genfromtxt("./AHPresults/Mod_ahp_scores.csv", delimiter=',', names=True, dtype=None, encoding='utf-8')


def process_metric(metric_name):
    """Sequentially processes the metric while parallelizing the sparse dictionary construction."""
    with Progress() as progress:
        task = progress.add_task(f'Processing metrics for {metric_name}', total=4)
        print(f'Constructing PWM for {metric_name}')
        pwm = modAhpFunctions.construct_sparse_pairwise_matrix(metrics[metric_name])
        progress.update(task, advance=1)
        print(f'Savig PWM for {metric_name}')
        with open(f'./AHPresults/{metric_name}_pwm.pkl', 'wb') as f:
            pickle.dump(pwm, f)
        progress.update(task, advance=1)
        print(f'Computing eigvals and eigvecs for {metric_name}')
        eigvals, eigvecs = eigs(pwm, k=1, which='LM')
        priority_vectors = np.abs(eigvecs[:, 0].real) / np.sum(np.abs(eigvecs[:, 0].real))
        progress.update(task, advance=1)
        print(f'Saving eigenvalues for {metric_name} to a file')
        with open(f"./AHPresults/{metric_name}_eigvals.pkl", 'wb') as f:
            pickle.dump([eigvals, eigvecs], f)
        with open(f"./AHPresults/{metric_name}_priorityV.pkl", "wb") as f:
            pickle.dump(priority_vectors, f)
        progress.update(task, advance=1)
        print(f'Finished processing {metric_name}')

if __name__ == "__main__":
    metric_names = ['t_test', 'entropy', 'roc_auc', 'snr']
    
    for metric in metric_names:
        process_metric(metric)
    print("All metrics processed successfully!")
