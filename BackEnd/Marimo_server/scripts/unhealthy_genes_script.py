import marimo as mo
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
import pickle as pkl
import pandas as pd
import plotly.express as px

plot2_df = pd.read_csv('../AHPresults/cancer_mean_exp.csv')
fig2 = mo.ui.plotly(
    px.bar(
        plot2_df,
        x="Genes",
        y="avg_expr_level",
        title="Gene Expression Visualization for Top 300 Genes",
        labels={"avg_expr_level": "Mean Expression Level"},
        color="avg_expr_level",  # Optional: color for better visualization
    ).update_layout(xaxis_tickangle=-45)
)

# with open('./pkl_files/unhealthy_df.pkl', 'wb') as f:
#     pkl.dump(plot2_df,f)
plot2_df.to_pickle('./pkl_files/unhealthy_df.pkl')

with open('./pkl_files/unhealthy_fig.pkl', 'wb') as f:
    pkl.dump(fig2,f)