import pandas as pd
import plotly.express as px
import marimo as mo
import pickle as pkl

#import data
cox_hazerdus_p = pd.read_csv('../../Models/CoxPHFitter/result/cox_comparison_metrics.csv')
cox_hazerdus_p.drop(columns='cmp to', inplace=True)
cox_summry = pd.read_csv('../../Models/CoxPHFitter/result/cox_model_summary.csv',index_col=0)
cox_summry = cox_summry.T
# Replace any values > 10 with NaN (or clip at 10 if you prefer)
cox_summry_numeric = cox_summry.select_dtypes(include='number')
cox_summry[cox_summry_numeric.columns] = cox_summry_numeric.where(cox_summry_numeric <= 10)

# cox_summry.set_index('covariate',inplace=True)
summery_fig = px.imshow(cox_summry)

# Filter features where p-value < 0.05 (i.e., reject the null hypothesis)
significant_features_df = cox_summry.T
significant_features_df = significant_features_df[significant_features_df["p"] < 0.05][["coef", "exp(coef)", "p"]]

# Sort by p-value for better readability
significant_features_df = significant_features_df.sort_values(by="p")
sig_html = significant_features_df.to_html(classes="cox-html")
styled = f"""
<style>
.cox-html {{
    font-family: Arial;
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
}}
.cox-html th, .cox-html td {{
    border: 1px solid #ccc;
    padding: 6px;
    text-align: center;
}}
.scroll-container {{
    max-height: 400px;
    overflow-y: auto;
    margin-top: 20px;
}}
</style>
<div class="scroll-container">{sig_html}</div>
"""
hazard_fig = px.bar(significant_features_df, x=significant_features_df.index, y='coef', color='p')

features_md = mo.md(f"""
We set the null hypothesis for each feature to zero. This means the feature has no association with the hazard (risk of death in this context). In simpler terms: Rejecting the null hypothesis (p-value < 0.05).

Features That Accept the Null Hypothesis
Based on the contents of your cox_model_summary.csv, here are the features that fail to show statistical significance (p-value ≥ 0.05), meaning they accept the null hypothesis and are not significantly associated with survival risk:

""")

insight = mo.md(f"""

Below is a table of features rejcting the null hypothesis. Aming to extract the the fignificant importance by analyzing ther HR and Log hazerdus ratio.


{
styled
}


ℹ️ Why Do We Use `coef` (log scale) in the Model?

The Cox model is based on the following formula:

$$
h(t \mid X) = h_0(t) \cdot \exp(\eta_1 X_1 + \eta_2 X_2 + \dots + \eta_n X_n)
$$

- `coef` is the **β (beta coefficient)** learned by the model for each variable.
- It represents the **logarithmic effect** of the variable on the hazard (risk of event).
- We use the log scale because:
  - It makes the math additive for multiple variables.
  - It ensures the final hazard ratio is always **positive** (since risk can't be negative).
- To interpret the effect in real terms, we take the exponential:  
  **`exp(coef)` = hazard ratio (HR)**

#### 🔁 Interpretation Examples:
- `coef = 1.0` → `exp(coef) = 2.72` → **2.72× higher risk**
- `coef = -0.5` → `exp(coef) = 0.61` → **39% lower risk**
""")

conclussion = mo.md(
"""
<span style="color:brown">Conclusion</span>

* Based on the features we extracted it seams like when cancer is at stage that its mutated LympthNodes are at category N3B, in this category cancer cells in lymph nodes are in the armpit and lymph nodes behind the breastbone the number of mutated lymphnodes are more than 10. With coef of 9.64, patients diagnosed with this catgory, are at really high mortality risk.

* Genes miRNA that are part of cluster C5 are posing more danger for matients mortality. We already stablished that the noncoding gene that was highly expressed was the one that regulates emune system.

* Finally the cinier patient obviously are more at risk. 
"""
)

cox_table = pd.read_csv('../../Models/CoxPHFitter/result/cox_model_summary.csv',index_col=0).T
cox_html_table = cox_table.to_html(classes="cox-html")

styled_table = f"""
<style>
.cox-html {{
    font-family: Arial;
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
}}
.cox-html th, .cox-html td {{
    border: 1px solid #ccc;
    padding: 6px;
    text-align: center;
}}
.scroll-container {{
    max-height: 400px;
    overflow-y: auto;
    margin-top: 20px;
}}
</style>
<div class="scroll-container">{cox_html_table}</div>
"""
cox_stack = mo.vstack([
    mo.ui.tabs({
        'Hazerdus Probabilities': hazard_fig,
        'Model Summery': mo.vstack([summery_fig, mo.Html(styled_table)])
    }),
    insight, conclussion
])

with open('pkl_files/cox.pkl', 'wb') as f:
    pkl.dump(cox_stack, f)

print("Done!")