import pandas as pd
import plotly.express as px 
import pickle as pkl
miRna_df = pd.read_csv('miRNA_list.txt', sep='\t')
print(miRna_df)

miRna_genes = miRna_df['Gene name'].to_list()
# columns_to_keep = ['Stage', 'paper_miRNA.Clusters'] + miRna_genes
cancer = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv')
columns_to_keep = ['Stage', 'paper_miRNA.Clusters'] + miRna_genes
cancer = cancer[columns_to_keep]
print(cancer.head())

# Group by Stage and take the mean of all genes
grouped = cancer.groupby('Stage').mean(numeric_only=True).reset_index()
grouped_long = grouped.melt(id_vars='Stage', var_name='miRNA', value_name='Expression')

grouped2 = cancer.groupby(['Stage','paper_miRNA.Clusters']).mean(numeric_only=True).reset_index()

# Now plot
fig = px.bar(
    grouped_long,
    x='Stage',
    y='Expression',
    color='miRNA',
    barmode='group',
    title='Mean miRNA Expression per Stage',
)

grouped.to_csv('resul_miRNA.csv', index=False)
grouped2.to_csv('resul_miRNA_withClusters.csv', index=False)

with open('./miRna_visuals.pkl', 'wb') as f:
    pkl.dump(fig, f)



