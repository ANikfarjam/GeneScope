import numpy as np
import pandas as pd
from rich.progress import Progress

with Progress() as progress:
    task1 = progress.add_task('Loading data frames', total=6)
    
    # Load data
    data = pd.read_csv('../../Marimo_server/AHPresults/stage_dataSet.csv')
    ahp_data = pd.read_csv('../../Marimo_server/AHPresults/final_Mod_ahp_scores.csv')
    progress.update(task1, advance=1)
    
    # Extract top 2000 genes
    ahp_data.sort_values(by='Scores', inplace=True)
    top_genes = ahp_data.Gene.to_list()[:2000]
    data = data.loc[:, ['Stage', 'age_at_diagnosis','gender','ethnicity','vital_status'] + [gene for gene in top_genes if gene in data.columns]]
    progress.update(task1, advance=1)

    copy_df = data.copy()
    initial_state_counts = copy_df['Stage'].value_counts()

    # Calculate weights
    N_min = initial_state_counts.min()
    weights = N_min / initial_state_counts
    
    # Calculate weighted Initial State Probabilities
    weighted_initial_probabilities = (initial_state_counts * weights) / (initial_state_counts * weights).sum()

    # Save weighted initial probabilities
    weighted_initial_probabilities.to_csv('./probabilitiesResults/weighted_initial_p.csv')
    progress.update(task1, advance=1)

    # Calculate weighted Transition Probabilities
    df = copy_df.sort_values(by=['Stage'])
    transition_counts = pd.crosstab(df['Stage'].shift(1), df['Stage'])
    weighted_transition_counts = transition_counts.mul(weights, axis=0)
    transition_probabilities = weighted_transition_counts.div(weighted_transition_counts.sum(axis=1), axis=0)
    transition_probabilities.to_csv('./probabilitiesResults/weighted_ts_p.csv')
    progress.update(task1, advance=1)

    # Calculate Emission Probabilities for Numeric Data
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric['Stage'] = df['Stage']
    emission_probabilities_numeric = df_numeric.groupby('Stage').mean()

    # Save numeric emission probabilities
    emission_probabilities_numeric.to_csv('./probabilitiesResults/numeric_em_p.csv')

    # Calculate Emission Probabilities for Categorical Data
    categorical_columns = ['gender', 'ethnicity', 'vital_status']
    categorical_emission_probs = pd.DataFrame(index=emission_probabilities_numeric.index)

    for col in categorical_columns:
        col_probs = {}
        for stage, stage_data in df.groupby('Stage'):
            category_counts = stage_data[col].value_counts(normalize=True)
            for category, prob in category_counts.items():
                categorical_emission_probs.loc[stage, f'{col}_{category}'] = prob

    # Save categorical emission probabilities
    categorical_emission_probs.to_csv('./probabilitiesResults/categorical_em_p.csv')

    print('All tasks completed successfully with weighted probability adjustments!')