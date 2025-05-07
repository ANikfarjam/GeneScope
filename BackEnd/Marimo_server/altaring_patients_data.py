
import pandas as pd

model_data = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
alternate_table = model_data[['Stage',  'year_of_diagnosis','ajcc_pathologic_t', 'ajcc_pathologic_n','ajcc_pathologic_m','paper_miRNA.Clusters','ethnicity','race', 'age_at_diagnosis', 'vital_status']]
#convert age from days to year
alternate_table['age_at_diagnosis']=alternate_table['age_at_diagnosis'].apply(lambda age: float(age/365))
"""

0–4 years Infants/Toddlers
5–14 years Childhood
15–19 years Adolescents
20-29 Young Adults
30-49 Adults
50–64 Middle-Aged Adults
65 -  Seniors

"""
def age_group(age):
    if age <= 4:
        return 'Infants/Toddlers'
    elif 4 < age <= 14:
        return 'Childhood'
    elif 14 < age <= 19:
        return 'Adolescents'
    elif 19 < age <= 29:
        return 'Young Adults'
    elif 29 < age <= 49:
        return 'Adults'
    elif 49 < age <= 64:
        return 'Middle-Aged Adults'
    else:
        return 'Seniors'
alternate_table['age_at_diagnosis'] = alternate_table['age_at_diagnosis'].apply(age_group)
###################################################
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# If it's already categorized, just rename the column
agg_df.rename(columns={'age_at_diagnosis': 'age_group'}, inplace=True)

# Then melt the data for plotting
df_melted = agg_df.melt(
    id_vars=['Stage', 'age_group', 'race'],
    value_vars=['tumor_size_min_cm', 'lymph_nodes_min'],
    var_name='Measurement',
    value_name='Value'
)
# Grouped mean for age_group bars
bar_data = df_melted.groupby(['Stage', 'age_group', 'Measurement']).agg(
    mean_value=('Value', 'mean')
).reset_index()

# Grouped mean for race lines
race_data = df_melted.groupby(['Stage', 'race', 'Measurement']).agg(
    mean_value=('Value', 'mean')
).reset_index()

# Create subplots
unique_measurements = bar_data['Measurement'].unique()
prog_fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=[f"{m}" for m in unique_measurements],
    vertical_spacing=0.15
)

# Define custom colors
age_colors = {
    'Young Adults': 'rgba(148, 103, 189, 0.7)',
    'Adults': 'rgba(44, 160, 44, 0.7)',
    'Middle-Aged Adults': 'rgba(31, 119, 180, 0.7)',
    'Seniors': 'rgba(255, 127, 14, 0.7)'
}
line_styles = ['solid', 'dash', 'dot', 'dashdot']
line_color_map = {}

# Add bar plots (age_group)
for i, measurement in enumerate(unique_measurements, start=1):
    m_df = bar_data[bar_data['Measurement'] == measurement]
    for age_groups in m_df['age_group'].unique():  
        subset = m_df[m_df['age_group'] == age_groups]
        prog_fig.add_trace(go.Bar(
            x=subset['Stage'],
            y=subset['mean_value'],
            name=age_groups,  
            marker_color=age_colors.get(age_groups, 'gray'),
            showlegend=(i == 1)
        ), row=i, col=1)


# Add line plots (race)
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

# Final touches
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
