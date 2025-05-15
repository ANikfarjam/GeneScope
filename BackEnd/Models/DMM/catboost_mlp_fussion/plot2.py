from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Fusion Model Architecture')
dot.attr(rankdir='LR')  # left to right layout
dot.attr('node', shape='box', style='filled', color='lightgray', fontname='Helvetica')

# Gene MLP Branch
dot.node('G0', 'Gene Input\n(2000-dim)', color='lightblue')
dot.node('G1', 'Dense(256)', color='lightblue')
dot.node('G2', 'Dropout', color='lightblue')
dot.node('G3', 'Dense(10)\nGene Embedding', color='lightblue')

# Clinical / CatBoost Branch
dot.node('C0', 'Clinical Input', color='lightyellow')
dot.node('C1', 'CatBoost\nModel', color='orange')
dot.node('C2', 'Extracted CatBoost\nFeatures (10-dim)', color='orange')

# Fusion MLP
dot.node('F0', 'Concat Layer\n[Gene + Clinical]', color='lightgreen')
dot.node('F1', 'Fusion Dense(512)', color='lightgreen')
dot.node('F2', 'Dropout', color='lightgreen')
dot.node('F3', 'Fusion Dense(192)', color='lightgreen')
dot.node('F4', 'Dropout', color='lightgreen')
dot.node('F5', 'Softmax Output\n(10 classes)', color='lightgreen')

# Edges for Gene MLP
dot.edges([('G0', 'G1'), ('G1', 'G2'), ('G2', 'G3')])

# Edges for CatBoost
dot.edges([('C0', 'C1'), ('C1', 'C2')])

# Inputs to Fusion
dot.edge('G3', 'F0')
dot.edge('C2', 'F0')

# Fusion MLP layers
dot.edge('F0', 'F1')
dot.edge('F1', 'F2')
dot.edge('F2', 'F3')
dot.edge('F3', 'F4')
dot.edge('F4', 'F5')

# Render to file or view directly
dot.render('fusion_model_graphviz', format='png', cleanup=True)
dot.view()
