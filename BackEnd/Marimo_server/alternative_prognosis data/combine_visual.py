import marimo as mo
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
import pickle as pkl
import pandas as pd
import plotly.express as px

data = pd.read_csv('./model_data_with_tnm_metrics.csv')


prog_fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=[m for m in unique_measurements],
    vertical_spacing=0.15
)

age_colors = {
    'Young Adults': 'rgba(148, 103, 189, 0.7)',
    'Adults': 'rgba(44, 160, 44, 0.7)',
    'Middle-Aged Adults': 'rgba(31, 119, 180, 0.7)',
    'Seniors': 'rgba(255, 127, 14, 0.7)'
}
line_styles = ['solid', 'dash', 'dot', 'dashdot']
line_color_map = {}

# Step 8: Bar traces (age group)
for i, measurement in enumerate(unique_measurements, start=1):
    m_df = bar_data[bar_data['Measurement'] == measurement]
    for age_group in all_ages:
        subset = m_df[m_df['age_group'] == age_group]
        if not subset.empty:
            prog_fig.add_trace(go.Bar(
                x=subset['Stage'],
                y=subset['mean_value'],
                name=age_group,
                marker_color=age_colors.get(age_group, 'gray'),
                showlegend=(i == 1)
            ), row=i, col=1)

# Step 9: Line traces (race)
for i, measurement in enumerate(unique_measurements, start=1):
    r_df = race_data[race_data['Measurement'] == measurement]
    for j, race in enumerate(r_df['race'].unique()):
        subset = r_df[r_df['race'] == race]
        if race not in line_color_map:
            line_color_map[race] = f"rgba({50+j*30}, {100+j*20}, {150+j*10}, 1)"
        prog_fig.add_trace(go.Scatter(
            x=subset['Stage'],
            y=subset['mean_value'],
            mode='lines+markers',
            name=race,
            line=dict(dash=line_styles[j % len(line_styles)],
                      color=line_color_map[race]),
            showlegend=(i == 1)
        ), row=i, col=1)

# Step 10: Final layout
prog_fig.update_layout(
    height=850,
    title='Mean Tumor Size and Lymph Node Count by Stage\nAge (Bars) and Race (Lines)',
    barmode='group',
    font=dict(size=13),
    legend_title_text='Group',
    margin=dict(t=100)
)
prog_fig.update_yaxes(title_text="Mean Value (cm / node count)", row=1)
prog_fig.update_yaxes(title_text="Mean Value (cm / node count)", row=2)
prog_fig.update_xaxes(title_text="Cancer Stage", tickangle=45)

mo.ui.plotly(prog_fig)  