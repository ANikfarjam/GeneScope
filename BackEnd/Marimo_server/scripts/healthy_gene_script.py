import marimo as mo
import pandas as pd
import plotly.express as px
import pickle as pkl

# healthy_dataSet = pd.read_csv("../../data/ModelDataSets/helthyExpressions.csv")
# healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
healthy_dataSet = pd.read_csv(
    '../data/helthyExpressions.csv', sep=",", index_col=0
)
# healthy_dataSet.set_index(healthy_dataSet.columns[0], inplace=True)
# Extract genes and compute mean expression
genes = healthy_dataSet.columns
average = healthy_dataSet.mean(axis=0)  # Compute mean across all samples

# Create a new DataFrame with gene names and their average expression
plot_df = pd.DataFrame({"Genes": genes, "avg_expr_level": average})

# Sort and select the top 300 genes by expression level
plot_df = plot_df.sort_values(by="avg_expr_level", ascending=False).iloc[:300]

# Create an interactive bar plot
fig = mo.ui.plotly(
    px.bar(
        plot_df,
        x="Genes",
        y="avg_expr_level",
        title="Gene Expression Visualization for Top 300 Genes",
        labels={"avg_expr_level": "Mean Expression Level"},
        color="avg_expr_level",  # Optional: color for better visualization
    ).update_layout(xaxis_tickangle=-45)
)

# Load gene descriptions
polished_df = pd.read_csv('../comparisonMLMTX/description_genes_healthy.csv')
html_table = polished_df.to_html(classes='healthy-table', index=False, escape=False)

with open('./pkl_files/healthy_table.pkl', 'wb') as f:
    pkl.dump(html_table, f)

with open('./pkl_files/healthy_fig.pkl', 'wb') as f:
    pkl.dump(fig, f)