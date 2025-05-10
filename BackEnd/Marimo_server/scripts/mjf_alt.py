import pandas as pd
import marimo as mo
import altair as alt
import pickle as pkl


#load Data
ahp_df = pd.read_csv('../AHPresults/final_Mod_ahp_scores.csv')

# Sort and take top 200 genes
ahp_top = ahp_df.sort_values(by="Scores", ascending=False).iloc[:200, :]

# Ensure "Gene" column is retained and treated as string
ahp_top_scaled = ahp_top.copy()
# Scale only selected numeric features if needed
scale_cols = ["Scores", "t_test", "Wilcoxon"]  # modify as needed
for col in scale_cols:
    if col in ahp_top_scaled.columns:
        ahp_top_scaled[col] *= 1e6
ahp_top_scaled["Gene"] = ahp_top_scaled["Gene"].astype(str)
ahp_top_scaled["Scores"] = ahp_top_scaled["Scores"].astype(float)
# Save the scaled DataFrame only
with open('./pkl_files/ahp_top_scaled.pkl', 'wb') as f:
    pkl.dump(ahp_top_scaled, f)

# Selection for interactive brushing
# brush = alt.selection_interval(encodings=["x", "y"])
# # Scatter Plot (Interactive)
# chart = mo.ui.altair_chart(
#     alt.Chart(ahp_top_scaled)
#     .mark_circle()
#     .encode(
#         x="Scores:Q",
#         y="t_test:Q",
#         color="entropy:Q",
#         size="Wilcoxon",
#         tooltip=[
#             "Gene:N",
#             "Scores:Q",
#             "t_test:Q",
#             "entropy:Q",
#             "roc_auc:Q",
#             "Wilcoxon",
#             "Wilcoxon_p",
#             "snr:Q",
#         ],
#     )
#     .add_params(brush)
# )

# # Display chart and dynamically updating table
# alt_plot = mo.vstack([chart, mo.ui.table(chart.value)])

# with open('./pkl_files/alt_plot.pkl', 'wb') as f:
#     pkl.dump(alt_plot, f)

print("done!")