from rich.progress import Progress
import numpy as np
import pandas as pd

with Progress() as progress:
    task1 = progress.add_task('Loading data frames', total=5)
    data = pd.read_csv('../../Marimo_server/AHPresults/stage_dataSet.csv')
    print('Cancer Stage dataset loaded!')
    progress.update(task1, advance=1)
    
    ahp_data = pd.read_csv('../../Marimo_server/AHPresults/final_Mod_ahp_scores.csv')
    print('AHP Analysis data loaded!')
    progress.update(task1, advance=1)
    
    print('Extracting top 2000 genes!')
    ahp_data.sort_values(by='Scores', inplace=True)
    progress.update(task1, advance=1)
    
    top_genes = ahp_data.Gene.to_list()[:2000]
    progress.update(task1, advance=1)
    
    # Keep 'Stage' and the genes that match the top_genes list
    data = data.loc[:, ['Stage'] + [gene for gene in top_genes if gene in data.columns]]
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
    
    # Calculate Emission Probabilities
    # Use mean values for each gene per stage
    emission_probabilities = df.groupby('Stage').mean()
    
    progress.update(task2, advance=1)
    print("Emission Probabilities (Mean values per stage):")
    print(emission_probabilities)

print('Saving datas!')
initial_state_probabilities.to_csv('./probabilitiesResults/intial_p.csv', index=False)
transition_probabilities.to_csv('./probabilitiesResults/ts_p.csv', index=False)
emission_probabilities.to_csv('./probabilitiesResults/em_p.csv', index=False)
print('Done! All task completeded successfully!')