import pandas as pd
import marimo as mo
import pickle as pk

stagedata = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
stage_count = mo.ui.table(stagedata['Stage'].value_counts().to_dict())

with open('./pkl_files/stg_count.pkl', 'wb') as f:
    pk.dump(stage_count, f)
print("done!")