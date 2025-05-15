import pandas as pd
import marimo as mo
import plotly.express as px
import pickle as pkl

# Load data
model_data = pd.read_csv('../AHPresults/fina_Stage_unaugmented2.csv', low_memory=False)
stage_p_df = pd.read_csv('../../Models/gbst/result/stage_result.csv', index_col=0)
model_matrix = pd.read_csv('../../Models/gbst/result/classification_mtrx.csv', index_col=0)
gdp_matrix = pd.read_csv('../../Models/gbst/result/gdb_p_result.csv')
manually_calc_prop = pd.read_csv('../AHPresults/weighted_initial_p.csv')

# Format 'count' column
manually_calc_prop['count'] = manually_calc_prop['count'].round(2)
manually_calc_prop.rename(columns={'count': 'calculate_p'}, inplace=True)

# Merge stage probabilities
merged_df = stage_p_df.merge(manually_calc_prop, left_on=stage_p_df.index, right_on=manually_calc_prop['Stage'])
merged_df.drop(columns='Stage', inplace=True)

# Stage probability plot
stage_fig = px.bar(
    merged_df,
    x=merged_df.index,
    y=['Estimated Probability', 'calculate_p'],
    title="Stage Diagnosis Probability",
    labels={"value": "Probability", "index": "Cancer Stage"},
    barmode="group",
    template="plotly_white"
)

# GDP heatmap
columns_to_sum = gdp_matrix.columns.drop('True Stage')
gdp_agg = gdp_matrix.groupby('True Stage')[columns_to_sum].mean().reset_index()
gdg_fig = px.imshow(
    gdp_agg.iloc[:, 1:],
    labels=dict(color="Prob."),
    x=columns_to_sum,
    y=gdp_agg['True Stage'],
    color_continuous_scale="plasma",
    title="Model Predicted Stage Probabilities"
)

# HTML model matrix
styled_html_table = model_matrix.to_html(classes='matrix-table', index=True)

styled_block = f"""
<style>
.matrix-table {{
    font-family: Arial;
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
}}
.matrix-table th, .matrix-table td {{
    border: 1px solid #ccc;
    padding: 6px;
    text-align: center;
}}
.scroll-box {{
    max-height: 500px;
    overflow-y: auto;
    margin-top: 20px;
}}
</style>
<div class="scroll-box">{styled_html_table}</div>
"""

# Insight box
insight = """
<div style="
    font-family: Arial, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    background-color: #f9f1f1;
    padding: 20px;
    border-left: 5px solid #963131;
    margin-top: 20px;
    max-width: 900px;
">
<b>Model Interpretation:</b><br><br>
The heatmap illustrates how well the model assigned high probability to the correct breast cancer stage. Each diagonal value represents a correct prediction, where the predicted stage aligns with the true stage.

Most of the strong signals appear along the diagonal, indicating that the model correctly classifies the majority of cases. Minor off-diagonal spikes are visible but relatively low — these reflect slight misclassifications, which may be attributed to the imbalance in our dataset. Since advanced stages had fewer examples, prediction confidence may be weaker in those cases.

Overall, the model demonstrates a strong ability to differentiate between stages based on clinical and gene expression data.
</div>
"""

# Create full display tab
cat_boost = mo.ui.tabs({
    'Stage Probability': mo.ui.plotly(stage_fig),
    'Stage Classification Heatmap': mo.vstack([
        mo.ui.plotly(gdg_fig),
        mo.Html(styled_block),
        mo.Html(insight)
    ])
})

# Save it
with open('pkl_files/catboos.pkl', 'wb') as f:
    pkl.dump(cat_boost, f)

print("✅ Updated: catboos.pkl saved with heatmap, matrix, and model explanation.")
