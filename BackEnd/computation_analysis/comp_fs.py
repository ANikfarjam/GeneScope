import numpy as np
import pickle
import os
from os import path
import pandas as pd 
#loading priority vectores for the final score calculation
priority_vectors = []
pv_files_path='./AHPresults'
pv_files = [path.join(root, f) for root, dirs, files in os.walk(pv_files_path) for f in files if f.endswith('_priorityV.pkl')]
for f in pv_files:
    with open(f, 'br') as file: 
        priority_vectors.append(pickle.load(file))
#print(pv_files[0])
final_scores = sum(priority_vectors) / len(priority_vectors)

#add final scores to the our ahp score table
ahp_df = pd.read_csv('./AHPresults/Mod_ahp_scores.csv')
ahp_df['Scores'] = final_scores
ahp_df.to_csv('./AHPresults/final_Mod_ahp_scores.csv', index=False)