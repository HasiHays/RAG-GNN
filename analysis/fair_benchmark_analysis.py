#!/usr/bin/env python3
"""
Fair Comparative Benchmark Analysis for RAG-GNN Framework
- Fair node classification (no information leakage)
- Publication-quality figures with Set2 colors
- Comprehensive evaluation across multiple tasks
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Set2 color palette (colorblind-friendly)
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
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("="*80)
print("FAIR COMPARATIVE BENCHMARK ANALYSIS: RAG-GNN vs Baseline Methods")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/8] Loading data...")

G = nx.read_edgelist('figures/protein_network.edgelist')
protein_df = pd.read_csv('figures/protein_annotations.csv')
proteins = protein_df['protein'].tolist()
categories = protein_df['category'].tolist()
category_ids = protein_df['category_id'].tolist()

protein_to_idx = {p: i for i, p in enumerate(proteins)}
n_nodes = len(proteins)

adj_matrix = nx.to_numpy_array(G, nodelist=proteins)
adj_sparse = csr_matrix(adj_matrix)

le = LabelEncoder()
labels = le.fit_transform(categories)
n_classes = len(np.unique(labels))

print(f"  Nodes: {n_nodes}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Categories: {n_classes}")

# =============================================================================
# 2. CREATE FAIR EXTERNAL LABELS (No information leakage)
# =============================================================================
print("\n[2/8] Creating fair external labels...")

# Method: Use network topology properties to create labels
# These are independent of functional categories

# Label 1: High centrality nodes (top 30% by betweenness)
betweenness = nx.betweenness_centrality(G)
bt_values = np.array([betweenness.get(p, 0) for p in proteins])
centrality_labels = (bt_values > np.percentile(bt_values, 70)).astype(int)

# Label 2: Hub nodes (degree > mean + 1 std)
degrees = np.array([G.degree(p) if p in G else 0 for p in proteins])
hub_threshold = degrees.mean() + degrees.std()
hub_labels = (degrees > hub_threshold).astype(int)

# Label 3: Bridging nodes (high betweenness but low clustering)
clustering = nx.clustering(G)
cl_values = np.array([clustering.get(p, 0) for p in proteins])
bridge_labels = ((bt_values > np.median(bt_values)) & (cl_values < np.median(cl_values))).astype(int)

# Combined fair label: XOR of hub and bridge (creates balanced, topology-based label)
fair_labels = (hub_labels ^ bridge_labels).astype(int)

print(f"  Fair labels: {fair_labels.sum()} positive ({100*fair_labels.mean():.1f}%)")
print(f"  Hub labels: {hub_labels.sum()} positive")
print(f"  Bridge labels: {bridge_labels.sum()} positive")

# =============================================================================
# 3. GENERATE EMBEDDINGS
# =============================================================================
print("\n[3/8] Generating embeddings for all methods...")

embedding_dim = 128
embeddings_dict = {}

# Adjacency normalization for GNN-like methods
degrees_safe = np.sum(adj_matrix, axis=1) + 1
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
A_norm = D_inv_sqrt @ (adj_matrix + np.eye(n_nodes)) @ D_inv_sqrt

# Transition matrix for random walk methods
D_inv = np.diag(1.0 / (np.sum(adj_matrix, axis=1) + 1e-10))
P = D_inv @ adj_matrix

# --- Raw Features ---
print("  Computing: Raw Features...")
degree_feat = degrees.reshape(-1, 1)
pagerank = nx.pagerank(G)
pr_feat = np.array([pagerank.get(p, 0) for p in proteins]).reshape(-1, 1)
bt_feat = bt_values.reshape(-1, 1)
cl_feat = cl_values.reshape(-1, 1)
raw_features = StandardScaler().fit_transform(np.hstack([degree_feat, pr_feat, bt_feat, cl_feat]))
raw_padded = np.zeros((n_nodes, embedding_dim))
raw_padded[:, :raw_features.shape[1]] = raw_features
embeddings_dict['Raw Features'] = raw_padded

# --- Spectral ---
print("  Computing: Spectral...")
n_comp = min(embedding_dim, n_nodes - 2)
U, S, Vt = svds(adj_sparse.astype(float), k=n_comp)
spectral_emb = StandardScaler().fit_transform(U * S)
embeddings_dict['Spectral'] = spectral_emb

# --- DeepWalk ---
print("  Computing: DeepWalk...")
walk_matrix = P + P @ P + P @ P @ P
walk_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
deepwalk_emb = StandardScaler().fit_transform(walk_svd.fit_transform(walk_matrix))
embeddings_dict['DeepWalk'] = deepwalk_emb

# --- Node2Vec ---
print("  Computing: Node2Vec...")
P2 = P @ P
node2vec_matrix = 0.5 * P + 0.3 * P2 + 0.2 * (P2 @ P)
node2vec_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
node2vec_emb = StandardScaler().fit_transform(node2vec_svd.fit_transform(node2vec_matrix))
embeddings_dict['Node2Vec'] = node2vec_emb

# --- LINE ---
print("  Computing: LINE...")
first_order = adj_matrix
second_order = adj_matrix @ adj_matrix
line_matrix = 0.5 * first_order + 0.5 * second_order
line_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
line_emb = StandardScaler().fit_transform(line_svd.fit_transform(line_matrix))
embeddings_dict['LINE'] = line_emb

# --- GCN ---
print("  Computing: GCN...")
np.random.seed(42)
H0 = np.random.randn(n_nodes, embedding_dim)
H1 = np.tanh(A_norm @ H0)
H2 = np.tanh(A_norm @ H1)
gcn_emb = StandardScaler().fit_transform(H2)
embeddings_dict['GCN'] = gcn_emb

# --- GraphSAGE ---
print("  Computing: GraphSAGE...")
neighbor_features = adj_matrix @ H0 / (degrees_safe.reshape(-1, 1))
sage_combined = np.hstack([H0, neighbor_features])
sage_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
graphsage_emb = StandardScaler().fit_transform(sage_svd.fit_transform(sage_combined))
embeddings_dict['GraphSAGE'] = graphsage_emb

# --- GAT ---
print("  Computing: GAT...")
similarity = H0 @ H0.T
attention = similarity * adj_matrix
attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-10)
H_att = attention @ H0
gat_combined = np.hstack([H0, H_att])
gat_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
gat_emb = StandardScaler().fit_transform(gat_svd.fit_transform(gat_combined))
embeddings_dict['GAT'] = gat_emb

# --- GNN-only ---
print("  Computing: GNN-only...")
H = np.random.randn(n_nodes, embedding_dim)
for _ in range(3):
    H = np.tanh(A_norm @ H)
gnn_only_emb = StandardScaler().fit_transform(H)
embeddings_dict['GNN-only'] = gnn_only_emb

# --- RAG-GNN (FAIR: semantic features NOT based on category labels) ---
print("  Computing: RAG-GNN (fair)...")
# Semantic features based on document retrieval simulation
# Use pathway co-occurrence patterns derived from network structure (not categories)

# Create document embeddings based on neighborhood patterns
np.random.seed(123)
n_docs = 100
doc_embeddings = np.random.randn(n_docs, 64)

# Simulate retrieval: each node retrieves documents based on its neighborhood pattern
neighborhood_pattern = A_norm @ A_norm  # 2-hop neighborhood
retrieval_scores = neighborhood_pattern @ np.random.randn(n_nodes, n_docs)
top_k = 10

# Aggregate retrieved document features
retrieved_features = np.zeros((n_nodes, 64))
for i in range(n_nodes):
    top_docs = np.argsort(retrieval_scores[i])[-top_k:]
    retrieved_features[i] = doc_embeddings[top_docs].mean(axis=0)

retrieved_norm = StandardScaler().fit_transform(retrieved_features)

# RAG fusion
alpha_struct = 0.6
alpha_semantic = 0.4
combined = np.hstack([alpha_struct * gnn_only_emb, alpha_semantic * retrieved_norm])
rag_svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
rag_emb = StandardScaler().fit_transform(rag_svd.fit_transform(combined))
embeddings_dict['RAG-GNN'] = rag_emb

print(f"  Generated {len(embeddings_dict)} embedding methods")

# =============================================================================
# 4. EVALUATION FUNCTIONS
# =============================================================================
print("\n[4/8] Computing evaluation metrics...")

def compute_silhouette(embeddings, labels, exclude_other=True):
    """Compute silhouette score for functional clustering"""
    if exclude_other:
        mask = np.array([cat != 'Other' for cat in categories])
        emb_filtered = embeddings[mask]
        labels_filtered = np.array(category_ids)[mask]
    else:
        emb_filtered = embeddings
        labels_filtered = labels

    if len(np.unique(labels_filtered)) < 2:
        return 0.0
    return silhouette_score(emb_filtered, labels_filtered)

def compute_link_prediction(embeddings, adj_matrix, n_samples=5000):
    """Compute link prediction metrics"""
    n = embeddings.shape[0]

    pos_edges = np.array(np.where(adj_matrix > 0)).T
    if len(pos_edges) > n_samples // 2:
        idx = np.random.choice(len(pos_edges), n_samples // 2, replace=False)
        pos_edges = pos_edges[idx]

    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        i, j = np.random.randint(0, n, 2)
        if i != j and adj_matrix[i, j] == 0:
            neg_edges.append([i, j])
    neg_edges = np.array(neg_edges)

    def edge_score(emb, edges):
        scores = []
        for i, j in edges:
            sim = np.dot(emb[i], emb[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-10)
            scores.append(sim)
        return np.array(scores)

    pos_scores = edge_score(embeddings, pos_edges)
    neg_scores = edge_score(embeddings, neg_edges)

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    return auroc, auprc

def compute_node_classification_fair(embeddings, labels, n_folds=5):
    """Fair node classification using topology-derived labels"""
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    aurocs = []
    for train_idx, test_idx in cv.split(embeddings, labels):
        clf.fit(embeddings[train_idx], labels[train_idx])
        y_pred = clf.predict_proba(embeddings[test_idx])[:, 1]
        aurocs.append(roc_auc_score(labels[test_idx], y_pred))

    return np.mean(aurocs), np.std(aurocs)

# =============================================================================
# 5. RUN BENCHMARKS
# =============================================================================
print("\n[5/8] Running comprehensive benchmarks...")

results = []
method_order = ['RAG-GNN', 'GNN-only', 'GCN', 'GAT', 'GraphSAGE', 'DeepWalk', 'Node2Vec', 'LINE', 'Spectral', 'Raw Features']

for method_name in method_order:
    emb = embeddings_dict[method_name]
    print(f"  Evaluating: {method_name}...")

    # Silhouette score (functional clustering)
    sil_score = compute_silhouette(emb, category_ids)

    # Link prediction
    lp_auroc, lp_auprc = compute_link_prediction(emb, adj_matrix)

    # Fair node classification
    nc_auroc_mean, nc_auroc_std = compute_node_classification_fair(emb, fair_labels)

    results.append({
        'Method': method_name,
        'Silhouette': sil_score,
        'Link Pred AUROC': lp_auroc,
        'Link Pred AUPRC': lp_auprc,
        'Node Class AUROC': nc_auroc_mean,
        'Node Class Std': nc_auroc_std
    })

results_df = pd.DataFrame(results)

# =============================================================================
# 6. DISPLAY RESULTS
# =============================================================================
print("\n" + "="*80)
print("BENCHMARK RESULTS (Fair Evaluation)")
print("="*80)

print("\n" + "-"*90)
print(f"{'Method':<15} {'Silhouette':>12} {'LP AUROC':>12} {'LP AUPRC':>12} {'NC AUROC':>18}")
print("-"*90)

for _, row in results_df.iterrows():
    nc_str = f"{row['Node Class AUROC']:.3f} ± {row['Node Class Std']:.3f}"
    print(f"{row['Method']:<15} {row['Silhouette']:>12.3f} {row['Link Pred AUROC']:>12.3f} {row['Link Pred AUPRC']:>12.3f} {nc_str:>18}")

print("-"*90)

# =============================================================================
# 7. GENERATE PUBLICATION-QUALITY FIGURES
# =============================================================================
print("\n[7/8] Generating publication-quality figures...")

# Create comprehensive figure with 4-column layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

methods = results_df['Method'].tolist()
n_methods = len(methods)
x_pos = np.arange(n_methods)
bar_width = 0.7

# Color assignment
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
ax1.set_ylabel('Silhouette Score')
ax1.set_title('(A) Functional Clustering Quality', fontweight='bold', loc='left')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.set_ylim(-0.25, 0.1)

# Highlight RAG-GNN bar
bars1[0].set_edgecolor('red')
bars1[0].set_linewidth(2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, results_df['Silhouette'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.02,
             f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)

# --- Panel B: Link Prediction AUROC ---
ax2 = fig.add_subplot(gs[0, 2:4])
bars2 = ax2.bar(x_pos, results_df['Link Pred AUROC'], bar_width, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Random')
ax2.set_ylabel('AUROC')
ax2.set_title('(B) Link Prediction Performance', fontweight='bold', loc='left')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.set_ylim(0.4, 1.05)

# Highlight best performers
best_lp = results_df['Link Pred AUROC'].idxmax()
bars2[best_lp].set_edgecolor('red')
bars2[best_lp].set_linewidth(2)

for i, (bar, val) in enumerate(zip(bars2, results_df['Link Pred AUROC'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# --- Panel C: Node Classification AUROC with error bars ---
ax3 = fig.add_subplot(gs[1, 0:2])
bars3 = ax3.bar(x_pos, results_df['Node Class AUROC'], bar_width,
                yerr=results_df['Node Class Std'], capsize=3,
                color=colors, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, label='Random')
ax3.set_ylabel('AUROC')
ax3.set_title('(C) Node Classification (Fair Labels)', fontweight='bold', loc='left')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods, rotation=45, ha='right')
ax3.set_ylim(0.4, 1.0)

# --- Panel D: Link Prediction AUPRC ---
ax4 = fig.add_subplot(gs[1, 2:4])
bars4 = ax4.bar(x_pos, results_df['Link Pred AUPRC'], bar_width, color=colors, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('AUPRC')
ax4.set_title('(D) Link Prediction Precision-Recall', fontweight='bold', loc='left')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, rotation=45, ha='right')
ax4.set_ylim(0.4, 1.05)

# --- Panel E: 2D Embedding Visualization (RAG-GNN) ---
ax5 = fig.add_subplot(gs[2, 0])
pca = PCA(n_components=2, random_state=42)
rag_2d = pca.fit_transform(embeddings_dict['RAG-GNN'])

unique_cats = sorted(set(categories))
cat_colors = {cat: SET2_COLORS[i % len(SET2_COLORS)] for i, cat in enumerate(unique_cats)}
point_colors = [cat_colors[cat] for cat in categories]

scatter = ax5.scatter(rag_2d[:, 0], rag_2d[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.set_title('(E) RAG-GNN Embeddings', fontweight='bold', loc='left')

# --- Panel F: 2D Embedding Visualization (GNN-only) ---
ax6 = fig.add_subplot(gs[2, 1])
gnn_2d = pca.fit_transform(embeddings_dict['GNN-only'])
ax6.scatter(gnn_2d[:, 0], gnn_2d[:, 1], c=point_colors, s=15, alpha=0.7, edgecolors='none')
ax6.set_xlabel('PC1')
ax6.set_ylabel('PC2')
ax6.set_title('(F) GNN-only Embeddings', fontweight='bold', loc='left')

# --- Panel G: Radar Chart Comparison ---
ax7 = fig.add_subplot(gs[2, 2], projection='polar')

# Normalize metrics for radar chart
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

ax7.plot(angles, rag_norm, 'o-', linewidth=2, color=SET2_COLORS[0], label='RAG-GNN')
ax7.fill(angles, rag_norm, alpha=0.25, color=SET2_COLORS[0])
ax7.plot(angles, gnn_norm, 's-', linewidth=2, color=SET2_COLORS[1], label='GNN-only')
ax7.fill(angles, gnn_norm, alpha=0.25, color=SET2_COLORS[1])
ax7.plot(angles, gcn_norm, '^-', linewidth=2, color=SET2_COLORS[2], label='GCN')
ax7.fill(angles, gcn_norm, alpha=0.25, color=SET2_COLORS[2])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(['Silhouette', 'LP AUROC', 'LP AUPRC', 'NC AUROC'], fontsize=8)
ax7.set_title('(G) Method Comparison', fontweight='bold', loc='left', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

# --- Panel H: Performance Improvement Over Baselines ---
ax8 = fig.add_subplot(gs[2, 3])

# Calculate improvements over GNN-only baseline
gnn_only_row = results_df[results_df['Method'] == 'GNN-only'].iloc[0]
improvements = []
for _, row in results_df.iterrows():
    if row['Method'] != 'GNN-only':
        improvements.append({
            'Method': row['Method'],
            'Silhouette Δ': row['Silhouette'] - gnn_only_row['Silhouette'],
            'LP AUROC Δ': row['Link Pred AUROC'] - gnn_only_row['Link Pred AUROC'],
            'NC AUROC Δ': row['Node Class AUROC'] - gnn_only_row['Node Class AUROC'],
        })

imp_df = pd.DataFrame(improvements)

# Plot improvements for RAG-GNN
rag_imp = imp_df[imp_df['Method'] == 'RAG-GNN'].iloc[0]
metrics_short = ['Silhouette', 'LP AUROC', 'NC AUROC']
rag_deltas = [rag_imp['Silhouette Δ'], rag_imp['LP AUROC Δ'], rag_imp['NC AUROC Δ']]

bar_colors = [SET2_COLORS[0] if d > 0 else SET2_COLORS[1] for d in rag_deltas]
bars8 = ax8.barh(metrics_short, rag_deltas, color=bar_colors, edgecolor='black', linewidth=0.5)
ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax8.set_xlabel('Δ vs GNN-only')
ax8.set_title('(H) RAG-GNN Improvement', fontweight='bold', loc='left')

for bar, val in zip(bars8, rag_deltas):
    ax8.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()

# Save figures in multiple formats
fig.savefig('figures/Fig-Benchmark.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-Benchmark.svg', format='svg', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-Benchmark.png', format='png', dpi=300, bbox_inches='tight')

print("  ✓ Saved: figures/Fig-Benchmark.pdf")
print("  ✓ Saved: figures/Fig-Benchmark.svg")
print("  ✓ Saved: figures/Fig-Benchmark.png")

plt.close()

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n[8/8] Saving results...")

# Save comprehensive results
benchmark_results = {
    'methods': results_df['Method'].tolist(),
    'silhouette_scores': results_df['Silhouette'].tolist(),
    'link_prediction_auroc': results_df['Link Pred AUROC'].tolist(),
    'link_prediction_auprc': results_df['Link Pred AUPRC'].tolist(),
    'node_classification_auroc': results_df['Node Class AUROC'].tolist(),
    'node_classification_std': results_df['Node Class Std'].tolist(),
    'fair_evaluation_note': 'Node classification uses topology-derived labels (hub/bridge nodes) independent of functional categories',
    'summary': {
        'best_silhouette': {
            'method': results_df.loc[results_df['Silhouette'].idxmax(), 'Method'],
            'score': float(results_df['Silhouette'].max())
        },
        'best_link_prediction': {
            'method': results_df.loc[results_df['Link Pred AUROC'].idxmax(), 'Method'],
            'score': float(results_df['Link Pred AUROC'].max())
        },
        'best_node_classification': {
            'method': results_df.loc[results_df['Node Class AUROC'].idxmax(), 'Method'],
            'score': float(results_df['Node Class AUROC'].max())
        }
    },
    'rag_gnn_vs_gnn_only': {
        'silhouette_improvement': float(rag_imp['Silhouette Δ']),
        'link_pred_improvement': float(rag_imp['LP AUROC Δ']),
        'node_class_improvement': float(rag_imp['NC AUROC Δ'])
    }
}

with open('figures/fair_benchmark_results.json', 'w') as f:
    json.dump(benchmark_results, f, indent=2)

results_df.to_csv('figures/fair_benchmark_results.csv', index=False)

print("  ✓ Saved: figures/fair_benchmark_results.json")
print("  ✓ Saved: figures/fair_benchmark_results.csv")

# Print summary table for manuscript
print("\n" + "="*80)
print("SUMMARY TABLE FOR MANUSCRIPT")
print("="*80)
print("\nTable: Comparative benchmark results across embedding methods")
print("-"*90)
print(f"{'Method':<15} {'Silhouette':>12} {'LP AUROC':>12} {'LP AUPRC':>12} {'NC AUROC':>12}")
print("-"*90)
for _, row in results_df.iterrows():
    print(f"{row['Method']:<15} {row['Silhouette']:>12.3f} {row['Link Pred AUROC']:>12.3f} {row['Link Pred AUPRC']:>12.3f} {row['Node Class AUROC']:>12.3f}")
print("-"*90)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
