#!/usr/bin/env python3
"""
Fix Benchmark Figure - Adjust radar plot and spacing
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import json
import warnings
warnings.filterwarnings('ignore')

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
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("Loading benchmark results...")

# Load results
results_df = pd.read_csv('figures/fair_benchmark_results.csv')

# Load network and embeddings for visualization
G = nx.read_edgelist('figures/protein_network.edgelist')
protein_df = pd.read_csv('figures/protein_annotations.csv')
proteins = protein_df['protein'].tolist()
categories = protein_df['category'].tolist()
category_ids = protein_df['category_id'].tolist()

n_nodes = len(proteins)
adj_matrix = nx.to_numpy_array(G, nodelist=proteins)

# Regenerate embeddings for visualization
degrees_safe = np.sum(adj_matrix, axis=1) + 1
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
A_norm = D_inv_sqrt @ (adj_matrix + np.eye(n_nodes)) @ D_inv_sqrt

np.random.seed(42)
H0 = np.random.randn(n_nodes, 128)
H = H0.copy()
for _ in range(3):
    H = np.tanh(A_norm @ H)
gnn_only_emb = StandardScaler().fit_transform(H)

# RAG-GNN embeddings
np.random.seed(123)
n_docs = 100
doc_embeddings = np.random.randn(n_docs, 64)
neighborhood_pattern = A_norm @ A_norm
retrieval_scores = neighborhood_pattern @ np.random.randn(n_nodes, n_docs)
top_k = 10
retrieved_features = np.zeros((n_nodes, 64))
for i in range(n_nodes):
    top_docs = np.argsort(retrieval_scores[i])[-top_k:]
    retrieved_features[i] = doc_embeddings[top_docs].mean(axis=0)
retrieved_norm = StandardScaler().fit_transform(retrieved_features)
combined = np.hstack([0.6 * gnn_only_emb, 0.4 * retrieved_norm])
rag_svd = TruncatedSVD(n_components=128, random_state=42)
rag_emb = StandardScaler().fit_transform(rag_svd.fit_transform(combined))

print("Creating figure...")

# Create figure with better spacing
fig = plt.figure(figsize=(16, 14))  # Increased height

# Use GridSpec with more space between rows
gs = GridSpec(3, 4, figure=fig,
              height_ratios=[1, 1, 1.1],  # Give bottom row slightly more space
              hspace=0.45,  # Increased vertical space between rows
              wspace=0.3)

methods = results_df['Method'].tolist()
n_methods = len(methods)
x_pos = np.arange(n_methods)
bar_width = 0.7

method_colors = {
    'RAG-GNN': SET2_COLORS[0],
    'GNN-only': SET2_COLORS[1],
    'GCN': SET2_COLORS[2],
    'GAT': SET2_COLORS[3],
    'GraphSAGE': SET2_COLORS[4],
    'DeepWalk': SET2_COLORS[5],
    'Node2Vec': SET2_COLORS[6],
    'LINE': SET2_COLORS[7],
    'Spectral': SET2_COLORS[8],
    'Raw Features': SET2_COLORS[9],
}
colors = [method_colors[m] for m in methods]

# --- Panel A: Silhouette Scores ---
ax1 = fig.add_subplot(gs[0, 0:2])
bars1 = ax1.bar(x_pos, results_df['Silhouette'], bar_width, color=colors, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Silhouette score')
ax1.set_title('(A) Functional clustering quality', fontweight='bold', loc='left')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.set_ylim(-0.25, 0.1)
bars1[0].set_edgecolor('red')
bars1[0].set_linewidth(2)
for i, (bar, val) in enumerate(zip(bars1, results_df['Silhouette'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.02,
             f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)

# --- Panel B: Link Prediction AUROC ---
ax2 = fig.add_subplot(gs[0, 2:4])
bars2 = ax2.bar(x_pos, results_df['Link Pred AUROC'], bar_width, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Random')
ax2.set_ylabel('AUROC')
ax2.set_title('(B) Link prediction performance', fontweight='bold', loc='left')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.set_ylim(0.4, 1.05)
best_lp = results_df['Link Pred AUROC'].idxmax()
bars2[best_lp].set_edgecolor('red')
bars2[best_lp].set_linewidth(2)
for i, (bar, val) in enumerate(zip(bars2, results_df['Link Pred AUROC'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# --- Panel C: Node Classification AUROC ---
ax3 = fig.add_subplot(gs[1, 0:2])
bars3 = ax3.bar(x_pos, results_df['Node Class AUROC'], bar_width,
                yerr=results_df['Node Class Std'], capsize=3,
                color=colors, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Random')
ax3.set_ylabel('AUROC')
ax3.set_title('(C) Node classification (fair labels)', fontweight='bold', loc='left')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods, rotation=45, ha='right')
ax3.set_ylim(0.35, 0.95)

# --- Panel D: Link Prediction AUPRC ---
ax4 = fig.add_subplot(gs[1, 2:4])
bars4 = ax4.bar(x_pos, results_df['Link Pred AUPRC'], bar_width, color=colors, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('AUPRC')
ax4.set_title('(D) Link prediction precision-recall', fontweight='bold', loc='left')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, rotation=45, ha='right')
ax4.set_ylim(0.4, 1.05)

# --- Panel E: RAG-GNN Embeddings ---
ax5 = fig.add_subplot(gs[2, 0])
pca = PCA(n_components=2, random_state=42)
rag_2d = pca.fit_transform(rag_emb)
unique_cats = sorted(set(categories))
cat_colors = {cat: SET2_COLORS[i % len(SET2_COLORS)] for i, cat in enumerate(unique_cats)}
point_colors = [cat_colors[cat] for cat in categories]
ax5.scatter(rag_2d[:, 0], rag_2d[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.set_title('(E) RAG-GNN embeddings', fontweight='bold', loc='left')

# --- Panel F: GNN-only Embeddings ---
ax6 = fig.add_subplot(gs[2, 1])
gnn_2d = pca.fit_transform(gnn_only_emb)
ax6.scatter(gnn_2d[:, 0], gnn_2d[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax6.set_xlabel('PC1')
ax6.set_ylabel('PC2')
ax6.set_title('(F) GNN-only embeddings', fontweight='bold', loc='left')

# --- Panel G: Radar Chart - FIXED with much smaller radius ---
ax7 = fig.add_subplot(gs[2, 2], projection='polar')

metrics = ['Silhouette', 'Link Pred AUROC', 'Link Pred AUPRC', 'Node Class AUROC']
rag_vals = results_df[results_df['Method'] == 'RAG-GNN'][metrics].values[0]
gnn_vals = results_df[results_df['Method'] == 'GNN-only'][metrics].values[0]
gcn_vals = results_df[results_df['Method'] == 'GCN'][metrics].values[0]

# Normalize to 0-1 range
all_vals = np.vstack([results_df[metrics].values])
min_vals = all_vals.min(axis=0)
max_vals = all_vals.max(axis=0)
range_vals = max_vals - min_vals + 1e-10

rag_norm = (rag_vals - min_vals) / range_vals
gnn_norm = (gnn_vals - min_vals) / range_vals
gcn_norm = (gcn_vals - min_vals) / range_vals

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

rag_norm = np.append(rag_norm, rag_norm[0])
gnn_norm = np.append(gnn_norm, gnn_norm[0])
gcn_norm = np.append(gcn_norm, gcn_norm[0])

# Plot with much smaller radius - scale to fit within 0.8 of the plot area
scale_factor = 0.45  # Scale down to 45% for much more margin
ax7.plot(angles, rag_norm * scale_factor, 'o-', linewidth=1, color=SET2_COLORS[0], label='RAG-GNN', markersize=3)
ax7.fill(angles, rag_norm * scale_factor, alpha=0.15, color=SET2_COLORS[0])
ax7.plot(angles, gnn_norm * scale_factor, 's-', linewidth=1, color=SET2_COLORS[1], label='GNN-only', markersize=3)
ax7.fill(angles, gnn_norm * scale_factor, alpha=0.15, color=SET2_COLORS[1])
ax7.plot(angles, gcn_norm * scale_factor, '^-', linewidth=1, color=SET2_COLORS[2], label='GCN', markersize=3)
ax7.fill(angles, gcn_norm * scale_factor, alpha=0.15, color=SET2_COLORS[2])

# Set labels with shorter names - rotate to avoid overlap
short_labels = ['Silhouette', 'LP AUROC', 'LP AUPRC', 'NC AUROC']
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(short_labels, fontsize=7)

# Adjust radial limits - include 0.8 line
ax7.set_ylim(0, 0.55)
ax7.set_yticks([0.1, 0.2, 0.3, 0.4])
ax7.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=6)

# Title aligned with E and F using figure text
ax7.set_title('')  # Remove default title
fig.text(0.515, 0.34, '(G) Method comparison', fontweight='bold', fontsize=11, ha='left')

# Legend positioned inside lower right
ax7.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=7, frameon=True)

# --- Panel H: RAG-GNN Improvement ---
ax8 = fig.add_subplot(gs[2, 3])

gnn_only_row = results_df[results_df['Method'] == 'GNN-only'].iloc[0]
rag_row = results_df[results_df['Method'] == 'RAG-GNN'].iloc[0]

metrics_short = ['Silhouette', 'LP AUROC', 'NC AUROC']
rag_deltas = [
    rag_row['Silhouette'] - gnn_only_row['Silhouette'],
    rag_row['Link Pred AUROC'] - gnn_only_row['Link Pred AUROC'],
    rag_row['Node Class AUROC'] - gnn_only_row['Node Class AUROC']
]

bar_colors = [SET2_COLORS[0] if d > 0 else SET2_COLORS[1] for d in rag_deltas]
bars8 = ax8.barh(metrics_short, rag_deltas, color=bar_colors, edgecolor='black', linewidth=0.5, height=0.5)
ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax8.set_xlabel('Δ vs GNN-only')
ax8.set_title('(H) RAG-GNN improvement', fontweight='bold', loc='left')
ax8.set_xlim(-0.2, 0.1)

for bar, val in zip(bars8, rag_deltas):
    offset = 0.008 if val > 0 else -0.008
    ax8.text(val + offset, bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()

# Save figures
fig.savefig('figures/Fig-Benchmark.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-Benchmark.svg', format='svg', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-Benchmark.png', format='png', dpi=300, bbox_inches='tight')

print("✓ Saved: figures/Fig-Benchmark.pdf")
print("✓ Saved: figures/Fig-Benchmark.svg")
print("✓ Saved: figures/Fig-Benchmark.png")

plt.close()
print("Done!")
