import numpy as np
import pandas as pd
import plotly.express as px
import pickle as pkl
#load Data
ahp_df = pd.read_csv('../AHPresults/final_Mod_ahp_scores.csv')

# Sort and take top 200 genes
ahp_top = ahp_df.sort_values(by="Scores", ascending=False).iloc[:200, :]
# Read biomarker file
biomarker = pd.read_csv('../AHPresults/fina_Stage_unaugmented.csv', low_memory=False)


# Load your top genes list
top_20_gene = ahp_top['Gene'].to_list()[:20]

# Filter to only those genes that exist in the biomarker DataFrame
existing_genes = [gene for gene in top_20_gene if gene in biomarker.columns]

# Optionally warn about missing genes
missing_genes = [gene for gene in top_20_gene if gene not in biomarker.columns]
if missing_genes:
    print("Warning: The following genes are missing from the biomarker dataset:", missing_genes)

# Select only the available columns
biomarker = biomarker.loc[:, ['Stage'] + existing_genes]
biomarker = biomarker.groupby(by='Stage').mean().reset_index()
biomarker = biomarker.set_index('Stage')

# bio_graph = px.imshow(
#     biomarker,
#     text_auto=True,  # this is the correct Plotly arg for annotations
#     aspect="auto",
#     color_continuous_scale="Cividis",
#     title="Mean Expression of Top Biomarkers by Cancer Stage"
# )

# bio_graph.update_layout(
#     xaxis_title="Genes",
#     yaxis_title="Cancer Stage",
#     font=dict(size=12)
# )

# bio_graph.show()

# Log transform for visualization only
biomarker_viz = np.log1p(biomarker)

# Convert real values to strings for annotation
text_labels = biomarker.round(2).astype(str)

# Plot
bio_graph = px.imshow(
    biomarker_viz,
    text_auto=False,
    color_continuous_scale="blues",
    aspect="auto",
    title="Log-Scaled Mean Expression of Top Biomarkers by Cancer Stage"
)

bio_graph.update_layout(
    xaxis_title="Genes",
    yaxis_title="Cancer Stage",
    font=dict(size=12)
)

with open('pkl_files/ahp_hitmap.pkl', 'wb') as f:
    pkl.dump(bio_graph, f)

print("Done!")