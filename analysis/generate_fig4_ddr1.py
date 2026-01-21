#!/usr/bin/env python3
"""
Generate Figure 4: DDR1 Subnetwork Visualization with RAG-GNN Embeddings
Uses proper RAG-GNN embeddings from corrected analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('figures', exist_ok=True)

np.random.seed(42)

# Set2 color palette
SET2_COLORS = [
    '#66c2a5',  # teal
    '#fc8d62',  # orange
    '#8da0cb',  # purple-blue
    '#e78ac3',  # pink
    '#a6d854',  # lime green
    '#ffd92f',  # yellow
    '#e5c494',  # tan
    '#b3b3b3',  # gray
    '#1b9e77',  # dark teal
    '#d95f02',  # dark orange
    '#7570b3',  # purple
    '#e7298a',  # magenta
    '#66a61e',  # green
    '#a6761d',  # brown
]

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("="*70)
print("GENERATING FIGURE 4: DDR1 SUBNETWORK WITH RAG-GNN EMBEDDINGS")
print("="*70)

# =====================================================
# STEP 1: Load data
# =====================================================
print("\n[1/5] Loading network data...")

# Load network
G = nx.read_edgelist('figures/protein_network.edgelist')

# Load annotations
protein_df = pd.read_csv('figures/protein_annotations.csv')
proteins = protein_df['protein'].tolist()
categories = protein_df['category'].tolist()
category_ids = protein_df['category_id'].tolist()

n_nodes = len(proteins)
print(f"  Network: {n_nodes} proteins, {G.number_of_edges()} interactions")

# Create adjacency matrix
adj_matrix = nx.to_numpy_array(G, nodelist=proteins)

# =====================================================
# STEP 2: Generate RAG-GNN Embeddings (same as proper analysis)
# =====================================================
print("\n[2/5] Generating RAG-GNN embeddings...")

def gnn_message_passing(adj_matrix, n_layers=3, hidden_dim=128):
    """GNN message passing with normalized adjacency"""
    n_nodes = adj_matrix.shape[0]
    A = adj_matrix + np.eye(n_nodes)
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    H = np.random.randn(n_nodes, hidden_dim) * 0.1
    H[:, 0] = np.log1p(degrees)
    H[:, 1] = np.array([nx.clustering(G, p) for p in proteins])

    for layer in range(n_layers):
        np.random.seed(42 + layer)
        W = np.random.randn(H.shape[1], hidden_dim) * np.sqrt(2.0 / H.shape[1])
        H_agg = A_norm @ H
        H = H_agg @ W
        H = np.maximum(0, H)
        H = (H - H.mean(axis=1, keepdims=True)) / (H.std(axis=1, keepdims=True) + 1e-8)

    return H

# GNN embeddings
gnn_embeddings = gnn_message_passing(adj_matrix, n_layers=3, hidden_dim=128)

# Create knowledge base
def create_knowledge_base(proteins, categories, n_docs_per_protein=5):
    documents = []
    doc_metadata = []

    pathway_knowledge = {
        'Cell cycle': ["regulates G1/S phase transition", "controls mitotic checkpoint", "mediates DNA replication"],
        'Apoptosis': ["induces mitochondrial permeabilization", "activates caspase cascade", "regulates BCL2 family"],
        'DNA repair': ["participates in homologous recombination", "mediates non-homologous end joining", "activates ATM/ATR"],
        'RTK signaling': ["transduces growth factor signals", "activates downstream MAPK and PI3K", "undergoes autophosphorylation"],
        'Transcription': ["binds DNA regulatory elements", "recruits coactivators", "integrates signaling inputs"],
        'PI3K-AKT-MTOR': ["phosphorylates phosphatidylinositol", "activates AKT", "regulates mTORC1"],
        'MAPK signaling': ["propagates RAS-RAF-MEK-ERK cascade", "regulates proliferation", "activates AP-1"],
        'Wnt signaling': ["stabilizes beta-catenin", "activates TCF/LEF", "regulates stem cells"],
        'TGF-beta signaling': ["activates SMAD factors", "regulates EMT", "controls cell cycle arrest"],
        'Notch signaling': ["undergoes proteolytic cleavage", "releases NICD", "activates HES/HEY"],
        'JAK-STAT': ["mediates cytokine signaling", "activates STAT factors", "regulates immunity"],
        'ECM-adhesion': ["mediates cell-matrix adhesion", "transduces mechanical signals", "regulates focal adhesion"],
        'Angiogenesis': ["regulates endothelial proliferation", "mediates VEGF signaling", "controls vessel sprouting"],
        'Other': ["participates in signaling networks", "interacts with partners", "regulates effectors"]
    }

    for protein, category in zip(proteins, categories):
        knowledge = pathway_knowledge.get(category, pathway_knowledge['Other'])
        for func_desc in knowledge[:n_docs_per_protein]:
            doc = f"{protein} {func_desc}. Involved in {category.lower()} pathway."
            documents.append(doc)
            doc_metadata.append({'protein': protein, 'category': category})

    return documents, doc_metadata

documents, doc_metadata = create_knowledge_base(proteins, categories)

# Document embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=128, stop_words='english', ngram_range=(1, 2))
doc_embeddings = tfidf.fit_transform(documents).toarray()
doc_embeddings = StandardScaler().fit_transform(doc_embeddings)

# Retrieval
np.random.seed(42)
projection_matrix = np.random.randn(gnn_embeddings.shape[1], doc_embeddings.shape[1]) * 0.1
query_embeddings = gnn_embeddings @ projection_matrix

top_k = 10
retrieved_features = np.zeros((n_nodes, doc_embeddings.shape[1]))
for i in range(n_nodes):
    query_norm = query_embeddings[i] / (np.linalg.norm(query_embeddings[i]) + 1e-10)
    doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = doc_norms @ query_norm
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]
    attention = np.exp(top_scores) / (np.sum(np.exp(top_scores)) + 1e-10)
    retrieved_features[i] = np.sum(doc_embeddings[top_indices] * attention[:, np.newaxis], axis=0)

# Fusion
alpha = 0.6
gnn_norm = StandardScaler().fit_transform(gnn_embeddings)
ret_norm = StandardScaler().fit_transform(retrieved_features)
combined = np.hstack([alpha * gnn_norm, (1 - alpha) * ret_norm])
svd = TruncatedSVD(n_components=128, random_state=42)
rag_gnn_embeddings = StandardScaler().fit_transform(svd.fit_transform(combined))

print(f"  RAG-GNN embeddings shape: {rag_gnn_embeddings.shape}")

# =====================================================
# STEP 3: Extract DDR1 subnetwork
# =====================================================
print("\n[3/5] Extracting DDR1 subnetwork...")

central_protein = 'DDR1'

if central_protein not in G.nodes():
    print(f"  {central_protein} not found, using alternative...")
    degrees = dict(G.degree())
    central_protein = max(degrees, key=degrees.get)

print(f"  Central protein: {central_protein}")

# Get neighbors
neighbors_1hop = list(G.neighbors(central_protein))
neighbors_2hop = []
for n in neighbors_1hop[:10]:
    neighbors_2hop.extend([n2 for n2 in G.neighbors(n) if n2 != central_protein and n2 not in neighbors_1hop])

neighbors_2hop = list(set(neighbors_2hop))[:20]

# Create subnetwork
subnetwork_nodes = [central_protein] + neighbors_1hop + neighbors_2hop
subG = G.subgraph(subnetwork_nodes).copy()

print(f"  Subnetwork: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
print(f"  1-hop neighbors: {len(neighbors_1hop)}")
print(f"  2-hop neighbors: {len(neighbors_2hop)}")

# =====================================================
# STEP 4: Compute properties
# =====================================================
print("\n[4/5] Computing node properties...")

# Categories for subnetwork
node_categories = {}
node_category_ids = {}
for node in subG.nodes():
    if node in proteins:
        idx = proteins.index(node)
        node_categories[node] = categories[idx]
        node_category_ids[node] = category_ids[idx]
    else:
        node_categories[node] = 'Other'
        node_category_ids[node] = 13

# Embedding similarities using RAG-GNN embeddings
central_idx = proteins.index(central_protein)
central_emb = rag_gnn_embeddings[central_idx]

node_similarities = {}
for node in subG.nodes():
    if node in proteins:
        idx = proteins.index(node)
        node_emb = rag_gnn_embeddings[idx]
        similarity = 1 - cosine(central_emb, node_emb)
        node_similarities[node] = max(0, similarity)  # Clamp to non-negative
    else:
        node_similarities[node] = 0.0

# Find top 5 most similar
sorted_sims = sorted([(n, s) for n, s in node_similarities.items() if n != central_protein],
                     key=lambda x: x[1], reverse=True)
print(f"\n  Top 5 most similar proteins to {central_protein}:")
for i, (protein, sim) in enumerate(sorted_sims[:5], 1):
    print(f"    {i}. {protein}: {sim:.3f} ({node_categories[protein]})")

# Betweenness
betweenness = nx.betweenness_centrality(subG)

# =====================================================
# STEP 5: Create visualization
# =====================================================
print("\n[5/5] Creating network visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color map for categories
category_colors = {
    0: '#E74C3C',   # Cell cycle - red
    1: '#3498DB',   # Apoptosis - blue
    2: '#2ECC71',   # DNA repair - green
    3: '#9B59B6',   # RTK signaling - purple
    4: '#F1C40F',   # Transcription - yellow
    5: '#E67E22',   # PI3K-AKT-MTOR - orange
    6: '#1ABC9C',   # MAPK signaling - turquoise
    7: '#34495E',   # Wnt signaling - dark gray
    8: '#16A085',   # TGF-beta signaling - teal
    9: '#8E44AD',   # Notch signaling - violet
    10: '#2980B9',  # JAK-STAT - blue
    11: '#27AE60',  # ECM-adhesion - green
    12: '#C0392B',  # Angiogenesis - dark red
    13: '#95A5A6',  # Other - gray
}

category_names = {
    0: 'Cell cycle', 1: 'Apoptosis', 2: 'DNA repair', 3: 'RTK signaling',
    4: 'Transcription', 5: 'PI3K-AKT-MTOR', 6: 'MAPK signaling', 7: 'Wnt signaling',
    8: 'TGF-beta signaling', 9: 'Notch signaling', 10: 'JAK-STAT', 11: 'ECM-adhesion',
    12: 'Angiogenesis', 13: 'Other'
}

# Layout
pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)

# Important nodes to label
important_nodes = [central_protein] + sorted(
    [n for n in neighbors_1hop if n in subG.nodes()],
    key=lambda x: betweenness.get(x, 0),
    reverse=True
)[:5]

# ============== PANEL A: Functional category ==============
ax1 = axes[0]

# Draw edges
nx.draw_networkx_edges(subG, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax1)

# Draw nodes by category
for cat_id in set(node_category_ids.values()):
    nodelist = [n for n in subG.nodes() if node_category_ids[n] == cat_id]
    if nodelist:
        node_sizes = [600 if n == central_protein else 250 for n in nodelist]
        nx.draw_networkx_nodes(
            subG, pos, nodelist=nodelist,
            node_color=category_colors[cat_id],
            node_size=node_sizes, alpha=0.85,
            edgecolors='black',
            linewidths=[1.5 if n == central_protein else 0.5 for n in nodelist],
            ax=ax1
        )

# Labels
labels_dict = {n: n for n in important_nodes}
nx.draw_networkx_labels(subG, pos, labels_dict, font_size=8, font_weight='bold',
                        font_color='black', ax=ax1)

ax1.set_title('(A) Functional category', fontsize=12, fontweight='bold', loc='left')
ax1.axis('off')

# Legend
present_categories = set(node_category_ids.values())
legend_elements = [
    mpatches.Patch(facecolor=category_colors[cat_id], edgecolor='black',
                   label=category_names[cat_id])
    for cat_id in sorted(present_categories)
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=7, frameon=False)

# ============== PANEL B: Embedding similarity ==============
ax2 = axes[1]

# Draw edges
nx.draw_networkx_edges(subG, pos, width=0.5, alpha=0.3, edge_color='gray', ax=ax2)

# Color nodes by similarity
similarities = [node_similarities[n] for n in subG.nodes()]
node_sizes = [600 if n == central_protein else 250 for n in subG.nodes()]

nodes = nx.draw_networkx_nodes(
    subG, pos,
    node_color=similarities,
    node_size=node_sizes,
    cmap='RdYlGn',
    vmin=0.0, vmax=1.0,
    alpha=0.85,
    edgecolors='black',
    linewidths=[1.5 if n == central_protein else 0.5 for n in subG.nodes()],
    ax=ax2
)

# Labels
nx.draw_networkx_labels(subG, pos, labels_dict, font_size=8, font_weight='bold',
                        font_color='black', ax=ax2)

ax2.set_title('(B) Embedding similarity', fontsize=12, fontweight='bold', loc='left')
ax2.axis('off')

# Colorbar
cbar = plt.colorbar(nodes, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Cosine similarity', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()

# =====================================================
# Save figure
# =====================================================
plt.savefig('figures/Fig-4-DDR1-Network.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig-4-DDR1-Network.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig-4-DDR1-Network.svg', dpi=300, bbox_inches='tight')

print("\n  Saved: figures/Fig-4-DDR1-Network.pdf/png/svg")

plt.close()

# Save subnetwork data
subnetwork_data = {
    'central_protein': central_protein,
    'central_category': node_categories[central_protein],
    'n_nodes': subG.number_of_nodes(),
    'n_edges': subG.number_of_edges(),
    'n_1hop': len(neighbors_1hop),
    'n_2hop': len(neighbors_2hop),
    'top_5_similar': [(p, float(s), node_categories[p]) for p, s in sorted_sims[:5]]
}

import json
with open('figures/Fig-4-subnetwork-data.json', 'w') as f:
    json.dump(subnetwork_data, f, indent=2)

print("\n" + "="*70)
print("SUCCESS! Figure 4 generated with RAG-GNN embeddings")
print("="*70)
print(f"\nSubnetwork summary:")
print(f"  Central protein: {central_protein} ({node_categories[central_protein]})")
print(f"  Nodes: {subG.number_of_nodes()}")
print(f"  Edges: {subG.number_of_edges()}")
print(f"  1-hop neighbors: {len(neighbors_1hop)}")
print(f"  2-hop neighbors: {len(neighbors_2hop)}")
