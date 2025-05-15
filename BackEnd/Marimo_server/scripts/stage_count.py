import pandas as pd
import marimo as mo
import pickle as pk

stagedata = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
stage_count = stagedata['Stage'].value_counts().reset_index()
stage_count.to_csv('pkl_files/stage_count.csv', index=False)

