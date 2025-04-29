import pickle
import os
import numpy as np
import pandas as pd


# Define paths
metrics = ['t_test', 'entropy', 'roc_auc', 'snr']
results = []

RI_large = 1.98  

for metric in metrics:
    eigval_path = f'./AHPresults/{metric}_eigvals.pkl'
    if os.path.exists(eigval_path):
        with open(eigval_path, 'rb') as f:
            eigvals, eigvecs = pickle.load(f)
            # Get principal eigenvalue (real part)
            principal_eigenvalue = np.real(eigvals[0])

            # Load PWM to get n (matrix size)
            pwm_path = f'./AHPresults/{metric}_pwm.pkl'
            with open(pwm_path, 'rb') as pwm_file:
                pwm = pickle.load(pwm_file)
                n = pwm.shape[0]

            # Calculate CI
            CI = (principal_eigenvalue - n) / (n - 1)
            # Calculate CR
            CR = CI / RI_large

            results.append({
                'Metric': metric,
                'Principal Eigenvalue': principal_eigenvalue,
                'Matrix Size': n,
                'Consistency Index (CI)': CI,
                'Consistency Ratio (CR)': CR
            })
    else:
        print(f"Eigenvalue file not found for {metric}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('./AHPresults/AHP_consistency_results.csv', index=False)

print("Consistency results saved to './AHPresults/AHP_consistency_results.csv'")
