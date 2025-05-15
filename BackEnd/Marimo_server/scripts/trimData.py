import pandas as pd

healthy = pd.read_csv('../../../data/ModelDataSets/helthyExpressions.csv')
cancer = pd.read_csv('../../../data/ModelDataSets/cancerExpressions.csv')
healthy.head(10).to_csv('healthy_trimed.csv')
cancer.head(10).to_csv('cancer_trimed.csv')