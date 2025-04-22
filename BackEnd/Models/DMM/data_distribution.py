import pandas as pd

data = pd.read_csv('./fina_Stage_unaugmented.csv')
print(data.iloc[:,:-2000].info(verbose=True, show_counts=True))
print(data.iloc[:,-2000:-1500].info(verbose=True, show_counts=True))