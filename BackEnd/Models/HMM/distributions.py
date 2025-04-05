from rich.progress import Progress
import numpy as np
import pandas as pd

with Progress() as progress:
    task1 = progress.add_task('Loading data frames', total=6)
    
    # Load data
    data = pd.read_csv('../../Marimo_server/AHPresults/stage_dataSet.csv')
    print('Cancer Stage dataset loaded!')
    progress.update(task1, advance=1)
    
    # Load AHP results
    ahp_data = pd.read_csv('../../Marimo_server/AHPresults/final_Mod_ahp_scores.csv')
    print('AHP Analysis data loaded!')
    progress.update(task1, advance=1)
    
    # Extract top 2000 genes
    print('Extracting top 2000 genes!')
    ahp_data.sort_values(by='Scores', inplace=True)
    progress.update(task1, advance=1)
    
    top_genes = ahp_data.Gene.to_list()[:2000]
    progress.update(task1, advance=1)
    
    # Keep 'Stage' and the genes that match the top_genes list
    data = data.loc[:, ['Stage', 'age_at_diagnosis','gender','ethnicity','vital_status'] + [gene for gene in top_genes if gene in data.columns]]
    progress.update(task1, advance=1)
    
    print('Done!')
    print(data.head())
    
    task2 = progress.add_task('Calculate initial/transitional/Emission probability distribution', total=6)
    copy_df = data.copy()
    
    # Calculate Initial State Probabilities
    initial_state_counts = copy_df['Stage'].value_counts()
    progress.update(task2, advance=1)
    
    initial_state_probabilities = initial_state_counts / len(copy_df)
    progress.update(task2, advance=1)
    print('Initial probabilities calculated successfully:')
    print(initial_state_probabilities)
    
    # Calculate Transition Probabilities
    df = copy_df.sort_values(by=['Stage'])
    progress.update(task2, advance=1)
    
    # Create a transition matrix by counting transitions
    transition_counts = pd.crosstab(df['Stage'].shift(1), df['Stage'])
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    
    progress.update(task2, advance=1)
    print("Transition Matrix (Probabilities):")
    print(transition_probabilities)
    
    # Calculate Emission Probabilities for Numeric Data
    # Drop non-numeric columns before calculating the mean
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Include the 'Stage' column for grouping
    df_numeric['Stage'] = df['Stage']
    
    # Calculate Emission Probabilities (Mean values per stage)
    emission_probabilities_numeric = df_numeric.groupby('Stage').mean()
    
    progress.update(task2, advance=1)
    print("Emission Probabilities for Numeric Data (Mean values per stage):")
    print(emission_probabilities_numeric)
    
    # Calculate Emission Probabilities for Categorical Data
    categorical_columns = ['gender', 'ethnicity', 'vital_status']
    categorical_emission_probs = pd.DataFrame(index=emission_probabilities_numeric.index)
    
    for col in categorical_columns:
        col_probs = {}
        for stage, stage_data in df.groupby('Stage'):
            category_counts = stage_data[col].value_counts(normalize=True)
            for category, prob in category_counts.items():
                categorical_emission_probs.loc[stage, f'{col}_{category}'] = prob

    print("Emission Probabilities for Categorical Data:")
    print(categorical_emission_probs)
    
    # Combine Numeric and Categorical Emission Probabilities
    combined_emission_probabilities = pd.concat([emission_probabilities_numeric, categorical_emission_probs], axis=1)
    
    progress.update(task2, advance=1)
    
    print('Saving data!')
    
    # Save all emission probabilities to a single CSV file
    combined_emission_probabilities.to_csv('./probabilitiesResults/combined_em_p.csv')
    initial_state_probabilities.to_csv('./probabilitiesResults/intial_p.csv', index=True)
    transition_probabilities.to_csv('./probabilitiesResults/ts_p.csv', index=True)
    
    print('Done! All tasks completed successfully!')
