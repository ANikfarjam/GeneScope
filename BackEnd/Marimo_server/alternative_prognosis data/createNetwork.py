import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pickle

# === Load Data ===
expr_df = pd.read_csv("../AHPresults/fina_Stage_unaugmented.csv")
ahp_df = pd.read_csv("../AHPresults/final_Mod_ahp_scores.csv")
miRNA_df = pd.read_csv('./resul_miRNA.csv')

# === Extract Genes Safely ===
miRNA_genes = pd.Index(miRNA_df.iloc[:, 2:].columns).drop_duplicates()
top_genes = ahp_df.sort_values(by='Scores', ascending=False)['Gene'].head(5).tolist()

# === Validate Columns ===
all_cols = set(expr_df.columns)
valid_genes = [g for g in top_genes + list(miRNA_genes) if g in all_cols]

# === Filter Dataset ===
filtered_df = expr_df[['Stage', 'paper_miRNA.Clusters'] + valid_genes].dropna()

# === Prepare Graph ===
G = nx.Graph()

clusters = sorted(filtered_df['paper_miRNA.Clusters'].unique())
stages = sorted(filtered_df['Stage'].unique())
genes = top_genes

# Identify stage vs cluster genes
stage_genes_set = set()
for stage in stages:
    sub = filtered_df[filtered_df['Stage'] == stage]
    top = sub[genes].mean().sort_values(ascending=False).head(5).index
    stage_genes_set.update(top)

stage_genes = sorted(stage_genes_set)
cluster_genes = sorted(set(genes) - stage_genes_set)

# === Add Nodes ===
G.add_nodes_from(clusters, bipartite='cluster')
G.add_nodes_from(stages, bipartite='stage')
G.add_nodes_from(stage_genes, bipartite='stage_gene')
G.add_nodes_from(cluster_genes, bipartite='cluster_gene')

# === Add Edges ===
# Cluster ↔ Stage
cluster_stage_pairs = filtered_df.groupby(['paper_miRNA.Clusters', 'Stage']).size()
for (cluster, stage), weight in cluster_stage_pairs.items():
    G.add_edge(cluster, stage, weight=weight)

# Cluster ↔ Cluster Genes
for cluster in clusters:
    sub = filtered_df[filtered_df['paper_miRNA.Clusters'] == cluster]
    gene_means = sub[cluster_genes].mean().sort_values(ascending=False).head(5)
    for gene, val in gene_means.items():
        G.add_edge(cluster, gene, weight=round(val * 100, 2))  # Scaled weight

# Stage ↔ Stage Genes
for stage in stages:
    sub = filtered_df[filtered_df['Stage'] == stage]
    gene_means = sub[stage_genes].mean().sort_values(ascending=False).head(5)
    for gene, val in gene_means.items():
        G.add_edge(stage, gene, weight=round(val * 100, 2))  # Scaled weight

# === Manual Positions ===
pos = {}

def evenly_spaced_positions(nodes, y_val):
    count = len(nodes)
    x_vals = np.linspace(-2, 2, count)
    return {node: (x, y_val) for node, x in zip(sorted(nodes), x_vals)}

# Layering
pos.update(evenly_spaced_positions(stage_genes, 2.0))
pos.update(evenly_spaced_positions(stages, 1.0))
pos.update(evenly_spaced_positions(clusters, 0.0))
pos.update(evenly_spaced_positions(cluster_genes, -1.5))

# === Create Edge Traces ===
edge_traces = []
annotations = []

def add_edge_trace(edge_list, color, label):
    edge_x, edge_y, edge_text = [], [], []
    for u, v, weight in edge_list:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{u} → {v}<br>Mean: {weight:.2f}")
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        annotations.append(dict(
            x=mid_x,
            y=mid_y,
            text=f"{weight:.2f}",
            showarrow=False,
            font=dict(size=8, color=color),
            opacity=0.7
        ))

    trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color=color),
        hoverinfo='text',
        mode='lines',
        text=edge_text,
        name=label
    )
    edge_traces.append(trace)

# Categorize edges
cluster_stage_edges = []
cluster_gene_edges = []
stage_gene_edges = []

for u, v, data in G.edges(data=True):
    types = {G.nodes[u]['bipartite'], G.nodes[v]['bipartite']}
    weight = data['weight']
    if {'cluster', 'stage'} == types:
        cluster_stage_edges.append((u, v, weight))
    elif {'cluster', 'cluster_gene'} == types:
        cluster_gene_edges.append((u, v, weight))
    elif {'stage', 'stage_gene'} == types:
        stage_gene_edges.append((u, v, weight))

# Add categorized edges
add_edge_trace(cluster_stage_edges, color="gray", label="Cluster → Stage")
add_edge_trace(cluster_gene_edges, color="red", label="Cluster → Gene")
add_edge_trace(stage_gene_edges, color="blue", label="Stage → Gene")

# === Node Trace ===
node_x, node_y, node_text, node_color = [], [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    typ = G.nodes[node].get("bipartite")
    if typ == "cluster":
        node_color.append("red")
    elif typ == "stage":
        node_color.append("blue")
    elif typ == "stage_gene":
        node_color.append("green")
    else:
        node_color.append("orange")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='bottom center',
    textfont=dict(size=10, color='black'),
    hoverinfo='text',
    marker=dict(
        color=node_color,
        size=12,
        line_width=1
    ),
    name="Nodes"
)

# === Final Figure ===
network_plt = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title='Layered Network: Clusters, Stages, and Top Genes',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=40, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=850,
        annotations=annotations
    )
)

# === Save and Show ===
network_plt.write_html("network_plot.html")
with open("network_plot.pkl", "wb") as f:
    pickle.dump(network_plt, f)

print("✅ Network plot saved as 'network_plot.html' and 'network_plot.pkl'")

network_plt.show()