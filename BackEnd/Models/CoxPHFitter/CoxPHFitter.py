from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

# Load your clinical dataset
clinical = pd.read_csv('./model_data.csv')

# Create event column (1 = dead, 0 = alive)
clinical['event'] = clinical['vital_status'].str.lower().apply(lambda x: 1 if x == 'dead' else 0)

# Approximate follow-up time (in days) from year_of_diagnosis to assumed 2025 endpoint
clinical['approx_diagnosis_year'] = clinical['year_of_diagnosis']
clinical['duration'] = (2025 - clinical['approx_diagnosis_year']) * 365.25

# Drop rows with missing key values
clinical = clinical.dropna(subset=['duration', 'event'])

# Select features (keep age_at_diagnosis as categorical)
features = [
    'duration', 'event', 'age_at_diagnosis', 'Stage',
    'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m',
    'paper_miRNA.Clusters', 'ethnicity', 'race'
]

# Subset and copy the data
cox_data = clinical[features].copy()

# Convert all categorical variables (including age_at_diagnosis) to dummy/one-hot encoding
categorical_cols = [
    'age_at_diagnosis', 'Stage', 'ajcc_pathologic_t',
    'ajcc_pathologic_n', 'ajcc_pathologic_m',
    'paper_miRNA.Clusters', 'ethnicity', 'race'
]
cox_data = pd.get_dummies(cox_data, columns=categorical_cols, drop_first=True)

# Drop dummy variables with near-zero variance (to prevent convergence issues)
low_variance_cols = [col for col in cox_data.columns if col not in ['event', 'duration'] and cox_data[col].var() < 1e-4]
cox_data = cox_data.drop(columns=low_variance_cols)

# Fit the Cox model with L2 regularization (penalizer)
cph = CoxPHFitter(penalizer=0.1)  # you can adjust penalizer strength if needed
cph.fit(cox_data, duration_col='duration', event_col='event')

# Export Cox model summary to CSV
cph.summary.to_csv("./result/cox_model_summary.csv", index=True)
# Extract the comparison metrics portion of the Cox summary
comparison_metrics = cph.summary[['cmp to', 'z', 'p', '-log2(p)']]

# Save to CSV
comparison_metrics.to_csv("./result/cox_comparison_metrics.csv", index=True)
# Optional: Plot hazard ratios
cph.plot()
