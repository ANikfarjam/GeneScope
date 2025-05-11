import pandas as pd
import pickle as pkl
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
import marimo as mo


stagedata = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# --------------- Your Data Preparation ---------------
corr_df = stagedata[['Stage', 'year_of_diagnosis', 'ajcc_pathologic_t', 'ajcc_pathologic_n',
                     'ajcc_pathologic_m', 'paper_miRNA.Clusters', 'ethnicity', 'race',
                     'age_at_diagnosis', 'vital_status']].copy()

# Encode categorical columns
for cols in ['Stage', 'vital_status', 'ethnicity', 'race', 'paper_miRNA.Clusters']:
    corr_df[cols] = corr_df[cols].astype('category')

# Drop rows with missing values
corr_df = corr_df.dropna()

# Now calculate association
stage_corr = []
vital_corr = []

featuress = [col for col in corr_df.columns if col not in ['Stage', 'vital_status']]

for _col in featuress:
    if corr_df[_col].dtype.name == 'category' or corr_df[_col].dtype == object:
        # Use Cramer's V for categorical
        corr_stage = cramers_v(corr_df['Stage'], corr_df[_col])
        corr_vital = cramers_v(corr_df['vital_status'], corr_df[_col])
    else:
        # Use Pearson correlation for numeric (like age)
        corr_stage = np.corrcoef(corr_df['Stage'].cat.codes, corr_df[_col])[0, 1]
        corr_vital = np.corrcoef(corr_df['vital_status'].cat.codes, corr_df[_col])[0, 1]

    stage_corr.append(corr_stage)
    vital_corr.append(corr_vital)

# Results DataFrame
correlation_results = pd.DataFrame({
    'Feature': featuress,
    'Association_with_Stage': stage_corr,
    'Association_with_Vital_Status': vital_corr
}).dropna()

# --------------- Plot ---------------
fig = go.Figure()

fig.add_trace(go.Bar(
    x=correlation_results['Feature'],
    y=correlation_results['Association_with_Stage'],
    name='Association with Stage'
))

fig.add_trace(go.Bar(
    x=correlation_results['Feature'],
    y=correlation_results['Association_with_Vital_Status'],
    name='Association with Vital Status'
))

fig.update_layout(
    barmode='group',
    title='Feature Associations with Stage and Vital Status (Cramér\'s V and Pearson)',
    xaxis_title='Features',
    yaxis_title='Association Strength',
    width=1000,
    height=600
)

#data = mo.ui.table(stagedata.head(200))
corr_plot = mo.ui.plotly(fig)

# Save raw DataFrame, which *is* pickle-able
with open('./pkl_files/stage_data.pkl', 'wb') as f:
    pkl.dump(stagedata.head(200), f)

with open('./pkl_files/corr_plot.pkl', 'wb') as f:
    pkl.dump(corr_plot, f)
with open('./pkl_files/corr_df.pkl', 'wb') as f:
    pkl.dump(corr_df, f)
print('Done1')