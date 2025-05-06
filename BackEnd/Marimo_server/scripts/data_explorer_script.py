import marimo as mo
import pandas as pd
import plotly.express as px
import pickle as pkl


# Load your full CSVs
ahp_df = pd.read_csv('../AHPresults/final_Mod_ahp_scores.csv')
stage_df = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
stage_df = stage_df.drop(stage_df.columns[29], axis=1)
valid_stages = stage_df['Stage'].value_counts()
valid_stages = valid_stages[valid_stages >= 14].index
ahp_df = ahp_df.iloc[:600,:]
# Sample 14 rows from each of those stages
sampled_df = (
    stage_df[stage_df['Stage'].isin(valid_stages)]
    .groupby("Stage", group_keys=False)
    .apply(lambda x: x.sample(n=14, random_state=42))
    .reset_index(drop=True)
)
ahp_df.to_pickle('./pkl_files/ahp_df.pkl') 
sampled_df.to_pickle('./pkl_files/stage_df.pkl')