import pandas as pd
import marimo as mo
import plotly.express as px
import pickle as pkl

# Load AHP data
ahp_df = pd.read_csv('../AHPresults/final_Mod_ahp_scores.csv')
ahp_top = ahp_df.sort_values(by="Scores", ascending=False).iloc[:200, :]

# Scale selected numeric columns
ahp_top_scaled = ahp_top.copy()
scale_cols = ["Scores", "t_test", "Wilcoxon"]
for col in scale_cols:
    if col in ahp_top_scaled.columns:
        ahp_top_scaled[col] *= 1e6

ahp_top_scaled["Gene"] = ahp_top_scaled["Gene"].astype(str)
ahp_top_scaled["Scores"] = ahp_top_scaled["Scores"].astype(float)

#  Create Plotly scatter plot
fig = px.scatter(
    ahp_top_scaled,
    x="Scores",
    y="t_test",
    color="entropy",
    size="Wilcoxon",
    hover_data=["Gene", "roc_auc", "Wilcoxon_p", "snr"],
    title="Top AHP Genes Scatter Plot"
)
fig.update_layout(template="plotly_white", title_x=0.5)

#  Convert table to styled HTML
html_table = ahp_top_scaled.to_html(classes='ahp-table', index=False, escape=False)

styled_table = f"""
<style>
.ahp-table {{
    font-family: Arial;
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
}}

.ahp-table th, .ahp-table td {{
    border: 1px solid #ccc;
    padding: 6px;
    text-align: left;
}}

.scroll-container {{
    max-height: 500px;
    overflow-y: auto;
}}
</style>

<div class="scroll-container">{html_table}</div>
"""

#  Create tabs layout
tab_view = mo.ui.tabs({
    "Scatter Plot": mo.ui.plotly(fig),
    "Table": mo.Html(styled_table)
})

#  Save the tab layout
with open('./pkl_files/ahp_tabs.pkl', 'wb') as f:
    pkl.dump(tab_view, f)

print("✅ ahp_tabs.pkl saved with Plotly scatter + HTML table.")
