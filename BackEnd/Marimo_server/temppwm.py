from rich.progress import Progress
import numpy as np
from scipy.sparse.linalg import eigs
import modAhpFunctions
import pickle
metrics = np.genfromtxt("./AHPresults/Mod_ahp_scores.csv", delimiter=',', names=True, dtype=None, encoding='utf-8')
def comput_PWM(metric_name):
    print(f'Constructing PWM for {metric_name}')
    pwm = modAhpFunctions.construct_sparse_pairwise_matrix(metrics[metric_name])
    print(f'Savig PWM for {metric_name}')
    with open(f'./AHPresults/{metric_name}_pwm.pkl', 'wb') as f:
        pickle.dump(pwm, f)
if __name__ == "__main__":
    metric_names = ['t_test', 'entropy', 'roc_auc', 'snr']
    
    for metric in metric_names:
        comput_PWM(metric)
    print("All metrics processed successfully!")