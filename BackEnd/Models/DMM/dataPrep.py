"""
This file creates train test val for gene expression and clinical vatiable
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import tqdm

# Confirm path
print("Working directory:", os.getcwd())

# Load data
data = pd.read_csv('./fina_Stage_unaugmented.csv')
data.dropna(inplace=True)
gene_exp = data.iloc[:, -2000:].columns
gene_exp_df = data[['Stage'] + list(gene_exp)]
clinical = data.iloc[:, :-2000]

print(gene_exp_df['Stage'].unique())
print(len(gene_exp_df['Stage'].unique()))

# Function to create datasets
def create_dataSet(df, type):
    for stage in tqdm.tqdm(df['Stage'].unique(), total=len(df['Stage'].unique()), desc=f'Preparing {type} dataset', unit='stage'):
        stage_df = df[df['Stage'] == stage]

        if stage_df.empty:
            print(f"⚠️ No data found for {stage}")
            continue
        
        train_df, temp_df = train_test_split(stage_df, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        stage_dir = f"./model_data/{type}/Stage_{stage.replace('-', '_').replace(' ', '_')}"
        os.makedirs(stage_dir, exist_ok=True)

        train_df.to_csv(f"{stage_dir}/train.csv", index=False)
        val_df.to_csv(f"{stage_dir}/val.csv", index=False)
        test_df.to_csv(f"{stage_dir}/test.csv", index=False)

        print(f"✅ Saved train/val/test for {stage} → {stage_dir}")

# Run for both gene expression and clinical datasets
create_dataSet(gene_exp_df, 'gene_exp')
create_dataSet(clinical, 'clinical')
