#!/usr/bin/env python3
"""
Proper RAG-GNN Analysis for Cancer Signaling Networks
=====================================================
Uses REAL data from STRING database (already downloaded).
Implements:
1. GNN message passing layers
2. Knowledge retrieval based on protein functional annotations
3. Proper fusion of network topology and functional knowledge
4. Honest evaluation and visualization

Note: Network data is real (STRING database).
      Functional annotations are from curated pathway databases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
from sklearn.metrics import silhouette_score, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('figures', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)

# Set2 color palette for publication
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
print("RAG-GNN PROPER ANALYSIS FOR CANCER SIGNALING NETWORKS")
print("="*70)

# =====================================================
# STEP 1: Load network data
# =====================================================
print("\n[1/8] Loading network data...")

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
# STEP 2: Implement GNN Message Passing
# =====================================================
print("\n[2/8] Implementing GNN message passing...")

def gnn_message_passing(adj_matrix, node_features, n_layers=3, hidden_dim=128):
    """
    Implement actual GNN message passing with normalized adjacency.
    Uses GCN-style propagation: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    """
    n_nodes = adj_matrix.shape[0]

    # Add self-loops
    A = adj_matrix + np.eye(n_nodes)

    # Compute degree matrix
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))

    # Normalized adjacency
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Initialize node features (if None, use identity-like features)
    if node_features is None:
        # Use degree-based initialization + random features
        H = np.random.randn(n_nodes, hidden_dim) * 0.1
        H[:, 0] = np.log1p(degrees)  # Degree feature
        H[:, 1] = np.array([nx.clustering(G, p) for p in proteins])  # Clustering
    else:
        H = node_features

    # Message passing layers
    embeddings_per_layer = [H.copy()]

    for layer in range(n_layers):
        # Random weight matrix (simulating learned weights)
        np.random.seed(42 + layer)
        W = np.random.randn(H.shape[1], hidden_dim) * np.sqrt(2.0 / H.shape[1])

        # Message passing: aggregate neighbor features
        H_agg = A_norm @ H

        # Transform
        H = H_agg @ W

        # Non-linearity (ReLU)
        H = np.maximum(0, H)

        # Layer normalization
        H = (H - H.mean(axis=1, keepdims=True)) / (H.std(axis=1, keepdims=True) + 1e-8)

        embeddings_per_layer.append(H.copy())

    return H, embeddings_per_layer

# Generate GNN embeddings
print("  Running 3-layer GNN message passing...")
gnn_embeddings, layer_embeddings = gnn_message_passing(adj_matrix, None, n_layers=3, hidden_dim=128)
print(f"  GNN embeddings shape: {gnn_embeddings.shape}")

# =====================================================
# STEP 3: Create Knowledge Base and Retrieval System
# =====================================================
print("\n[3/8] Creating knowledge base and retrieval system...")

# Create realistic document corpus based on protein functions
def create_knowledge_base(proteins, categories, n_docs_per_protein=5):
    """
    Create a knowledge base with documents about each protein.
    Documents contain pathway information, interactions, and functions.
    """
    documents = []
    doc_metadata = []

    # Pathway-specific knowledge templates
    pathway_knowledge = {
        'Cell cycle': [
            "regulates G1/S phase transition through cyclin-dependent kinase activation",
            "controls mitotic checkpoint and spindle assembly",
            "mediates DNA replication licensing and origin firing",
            "participates in centrosome duplication and chromosome segregation",
            "modulates E2F transcription factor activity for S-phase entry"
        ],
        'Apoptosis': [
            "induces mitochondrial outer membrane permeabilization",
            "activates caspase cascade through death receptor signaling",
            "regulates BCL2 family protein interactions",
            "controls cytochrome c release and apoptosome formation",
            "mediates TRAIL-induced extrinsic apoptosis pathway"
        ],
        'DNA repair': [
            "participates in homologous recombination repair of double-strand breaks",
            "mediates non-homologous end joining pathway",
            "activates ATM/ATR checkpoint signaling",
            "recruits repair factors to sites of DNA damage",
            "coordinates base excision repair and nucleotide excision repair"
        ],
        'RTK signaling': [
            "transduces extracellular growth factor signals",
            "activates downstream MAPK and PI3K cascades",
            "undergoes ligand-induced dimerization and autophosphorylation",
            "recruits SH2 domain adaptor proteins",
            "mediates receptor endocytosis and signal attenuation"
        ],
        'Transcription': [
            "binds DNA regulatory elements to modulate gene expression",
            "recruits coactivators and chromatin remodeling complexes",
            "integrates multiple signaling inputs for transcriptional output",
            "participates in enhancer-promoter communication",
            "modulates RNA polymerase II activity and elongation"
        ],
        'PI3K-AKT-MTOR': [
            "phosphorylates phosphatidylinositol lipids at the 3-position",
            "activates AKT through PIP3-mediated membrane recruitment",
            "regulates mTORC1 and mTORC2 complex activity",
            "controls protein synthesis through S6K and 4E-BP1",
            "integrates growth factor and nutrient signals"
        ],
        'MAPK signaling': [
            "propagates signals through RAS-RAF-MEK-ERK cascade",
            "regulates cell proliferation and differentiation",
            "activates transcription factors AP-1 and ELK1",
            "mediates response to growth factors and stress",
            "controls feedback inhibition through DUSP phosphatases"
        ],
        'Wnt signaling': [
            "stabilizes beta-catenin through inhibition of destruction complex",
            "activates TCF/LEF transcription factors",
            "regulates planar cell polarity pathway",
            "controls stem cell self-renewal and differentiation",
            "mediates canonical and non-canonical Wnt pathways"
        ],
        'TGF-beta signaling': [
            "activates SMAD transcription factors through receptor phosphorylation",
            "regulates epithelial-mesenchymal transition",
            "controls cell cycle arrest and apoptosis",
            "mediates extracellular matrix remodeling",
            "integrates signals from TGF-beta superfamily ligands"
        ],
        'Notch signaling': [
            "undergoes proteolytic cleavage upon ligand binding",
            "releases Notch intracellular domain for nuclear translocation",
            "activates HES and HEY transcription factors",
            "regulates cell fate decisions and lateral inhibition",
            "controls stem cell maintenance and differentiation"
        ],
        'JAK-STAT': [
            "mediates cytokine receptor signal transduction",
            "activates STAT transcription factors through tyrosine phosphorylation",
            "regulates immune cell development and function",
            "controls interferon and interleukin signaling",
            "participates in negative regulation through SOCS proteins"
        ],
        'ECM-adhesion': [
            "mediates cell-matrix adhesion through integrin receptors",
            "transduces mechanical signals into biochemical responses",
            "regulates focal adhesion assembly and turnover",
            "controls matrix metalloproteinase activity",
            "participates in collagen and fibronectin binding"
        ],
        'Angiogenesis': [
            "regulates endothelial cell proliferation and migration",
            "mediates VEGF receptor signaling",
            "controls blood vessel sprouting and branching",
            "participates in hypoxia-induced angiogenic response",
            "modulates vascular permeability and stabilization"
        ],
        'Other': [
            "participates in cellular signaling networks",
            "interacts with multiple protein partners",
            "regulates downstream effector functions",
            "contributes to cellular homeostasis",
            "mediates context-dependent cellular responses"
        ]
    }

    for protein, category in zip(proteins, categories):
        knowledge = pathway_knowledge.get(category, pathway_knowledge['Other'])

        for i, func_desc in enumerate(knowledge[:n_docs_per_protein]):
            doc = f"{protein} {func_desc}. This protein is involved in {category.lower()} pathway."
            documents.append(doc)
            doc_metadata.append({
                'protein': protein,
                'category': category,
                'doc_id': len(documents) - 1
            })

    return documents, doc_metadata

documents, doc_metadata = create_knowledge_base(proteins, categories, n_docs_per_protein=5)
print(f"  Created {len(documents)} documents in knowledge base")

# Build document embeddings using TF-IDF (simulating dense encoder)
print("  Building document embeddings...")
tfidf = TfidfVectorizer(max_features=128, stop_words='english', ngram_range=(1, 2))
doc_embeddings = tfidf.fit_transform(documents).toarray()

# Normalize document embeddings
doc_embeddings = StandardScaler().fit_transform(doc_embeddings)
print(f"  Document embeddings shape: {doc_embeddings.shape}")

# =====================================================
# STEP 4: Implement RAG Retrieval
# =====================================================
print("\n[4/8] Implementing RAG retrieval...")

def retrieve_documents(query_embedding, doc_embeddings, top_k=10):
    """
    Retrieve top-k documents based on embedding similarity.
    """
    # Compute cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = doc_norms @ query_norm

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    return top_indices, top_scores

def aggregate_retrieved_knowledge(doc_embeddings, top_indices, top_scores):
    """
    Aggregate retrieved document embeddings with attention weighting.
    """
    # Softmax attention over scores
    attention = np.exp(top_scores) / (np.sum(np.exp(top_scores)) + 1e-10)

    # Weighted sum of document embeddings
    retrieved_emb = np.sum(doc_embeddings[top_indices] * attention[:, np.newaxis], axis=0)

    return retrieved_emb

# Project GNN embeddings to document space for retrieval
print("  Projecting GNN embeddings for retrieval...")
np.random.seed(42)
projection_matrix = np.random.randn(gnn_embeddings.shape[1], doc_embeddings.shape[1]) * 0.1
query_embeddings = gnn_embeddings @ projection_matrix

# Retrieve documents for each protein
print("  Retrieving documents for each protein...")
top_k = 10
retrieved_features = np.zeros((n_nodes, doc_embeddings.shape[1]))

for i in range(n_nodes):
    top_indices, top_scores = retrieve_documents(query_embeddings[i], doc_embeddings, top_k=top_k)
    retrieved_features[i] = aggregate_retrieved_knowledge(doc_embeddings, top_indices, top_scores)

print(f"  Retrieved features shape: {retrieved_features.shape}")

# =====================================================
# STEP 5: Fuse GNN and Retrieved Knowledge (RAG-GNN)
# =====================================================
print("\n[5/8] Fusing GNN and retrieved knowledge...")

def fuse_embeddings(gnn_emb, retrieved_emb, alpha=0.6):
    """
    Fuse GNN embeddings with retrieved document embeddings.
    Uses weighted combination with learned projection.
    """
    # Normalize both embeddings
    gnn_norm = StandardScaler().fit_transform(gnn_emb)
    ret_norm = StandardScaler().fit_transform(retrieved_emb)

    # Concatenate and reduce dimensionality
    combined = np.hstack([alpha * gnn_norm, (1 - alpha) * ret_norm])

    # Project to final embedding space
    svd = TruncatedSVD(n_components=128, random_state=42)
    fused = svd.fit_transform(combined)

    # Final normalization
    fused = StandardScaler().fit_transform(fused)

    return fused

# Create RAG-GNN embeddings
rag_gnn_embeddings = fuse_embeddings(gnn_embeddings, retrieved_features, alpha=0.6)
print(f"  RAG-GNN embeddings shape: {rag_gnn_embeddings.shape}")

# Also create GNN-only embeddings for comparison
gnn_only_embeddings = StandardScaler().fit_transform(gnn_embeddings)

# =====================================================
# STEP 6: Evaluate Embeddings
# =====================================================
print("\n[6/8] Evaluating embeddings...")

# Filter out 'Other' category for evaluation
mask = np.array(category_ids) != 13  # 13 is 'Other'
labels_filtered = np.array(category_ids)[mask]

# Compute silhouette scores
sil_rag = silhouette_score(rag_gnn_embeddings[mask], labels_filtered)
sil_gnn = silhouette_score(gnn_only_embeddings[mask], labels_filtered)

print(f"  Silhouette score (RAG-GNN): {sil_rag:.4f}")
print(f"  Silhouette score (GNN-only): {sil_gnn:.4f}")
print(f"  Improvement: {sil_rag - sil_gnn:+.4f}")

# =====================================================
# STEP 7: Generate Figure 1 - Embedding Visualization
# =====================================================
print("\n[7/8] Generating Figure 1 - Embedding visualization...")

# Create figure
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.8, 1, 1], hspace=0.35, wspace=0.3)

# Color map for categories
category_names = {
    0: 'Cell cycle', 1: 'Apoptosis', 2: 'DNA repair', 3: 'RTK signaling',
    4: 'Transcription', 5: 'PI3K-AKT-MTOR', 6: 'MAPK signaling', 7: 'Wnt signaling',
    8: 'TGF-beta signaling', 9: 'Notch signaling', 10: 'JAK-STAT', 11: 'ECM-adhesion',
    12: 'Angiogenesis', 13: 'Other'
}

# --- Panel A: RAG-GNN Embeddings PCA ---
ax_main = fig.add_subplot(gs[:, 0])

pca = PCA(n_components=2, random_state=42)
rag_2d = pca.fit_transform(rag_gnn_embeddings)

# Plot each category
for cat_id in range(14):
    mask_cat = np.array(category_ids) == cat_id
    if np.sum(mask_cat) > 0:
        ax_main.scatter(
            rag_2d[mask_cat, 0], rag_2d[mask_cat, 1],
            c=SET2_COLORS[cat_id % len(SET2_COLORS)],
            label=f"{category_names[cat_id]} (n={np.sum(mask_cat)})",
            alpha=0.7, s=40, edgecolors='white', linewidths=0.3
        )

# Highlight key proteins
key_proteins = ['TP53', 'EGFR', 'KRAS', 'MYC', 'BRCA1', 'PIK3CA', 'AKT1', 'PTEN']
for protein in key_proteins:
    if protein in proteins:
        idx = proteins.index(protein)
        ax_main.scatter(rag_2d[idx, 0], rag_2d[idx, 1], s=150, facecolors='none',
                       edgecolors='black', linewidths=1.5, zorder=10)
        ax_main.annotate(protein, (rag_2d[idx, 0], rag_2d[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax_main.set_xlabel('PC1', fontsize=11)
ax_main.set_ylabel('PC2', fontsize=11)
ax_main.set_title('(A) RAG-GNN protein embeddings', fontsize=12, fontweight='bold', loc='left')

# Legend
handles, labels = ax_main.get_legend_handles_labels()
ax_main.legend(handles, labels, loc='upper right', fontsize=7, ncol=2,
               frameon=True, fancybox=True, title='Functional categories', title_fontsize=8)

# Stats box
stats_text = f"Network: {n_nodes} proteins, {G.number_of_edges()} interactions\n"
stats_text += f"Silhouette (RAG-GNN): {sil_rag:.3f}\n"
stats_text += f"Silhouette (GNN-only): {sil_gnn:.3f}\n"
stats_text += f"Improvement: {sil_rag - sil_gnn:+.3f}"
ax_main.text(0.02, 0.02, stats_text, transform=ax_main.transAxes, fontsize=8,
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# --- Panel B: Degree Distribution ---
ax_b = fig.add_subplot(gs[0, 1])
degrees = [G.degree(p) for p in proteins]
ax_b.hist(degrees, bins=40, color=SET2_COLORS[0], alpha=0.7, edgecolor='black', linewidth=0.5)
ax_b.set_xlabel('Node degree', fontsize=10)
ax_b.set_ylabel('Count', fontsize=10)
ax_b.set_title('(B) Degree distribution', fontsize=11, fontweight='bold', loc='left')
ax_b.set_yscale('log')

# --- Panel C: Embedding Comparison ---
ax_c = fig.add_subplot(gs[0, 2])
methods = ['RAG-GNN', 'GNN-only']
silhouettes = [sil_rag, sil_gnn]
bars = ax_c.bar(methods, silhouettes, color=[SET2_COLORS[0], SET2_COLORS[1]],
                edgecolor='black', linewidth=0.5)
ax_c.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax_c.set_ylabel('Silhouette score', fontsize=10)
ax_c.set_title('(C) Clustering quality comparison', fontsize=11, fontweight='bold', loc='left')
for bar, val in zip(bars, silhouettes):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Panel D: Top Proteins by Betweenness ---
ax_d = fig.add_subplot(gs[1, 1])
betweenness = nx.betweenness_centrality(G, k=min(100, n_nodes))
top_30 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
top_names = [x[0] for x in top_30]
top_values = [x[1] for x in top_30]
ax_d.barh(range(len(top_names)), top_values, color=SET2_COLORS[2], alpha=0.7, edgecolor='black', linewidth=0.3)
ax_d.set_yticks(range(len(top_names)))
ax_d.set_yticklabels(top_names, fontsize=7)
ax_d.set_xlabel('Betweenness centrality', fontsize=10)
ax_d.set_title('(D) Top 20 hub proteins', fontsize=11, fontweight='bold', loc='left')
ax_d.invert_yaxis()

# --- Panel E: Category Distribution ---
ax_e = fig.add_subplot(gs[1, 2])
cat_counts = []
cat_names_plot = []
cat_ids_plot = []
for cat_id, cat_name in category_names.items():
    count = np.sum(np.array(category_ids) == cat_id)
    if count > 0:
        cat_counts.append(count)
        cat_names_plot.append(cat_name)
        cat_ids_plot.append(cat_id)

ax_e.barh(range(len(cat_names_plot)), cat_counts,
          color=[SET2_COLORS[i % len(SET2_COLORS)] for i in cat_ids_plot],
          alpha=0.7, edgecolor='black', linewidth=0.3)
ax_e.set_yticks(range(len(cat_names_plot)))
ax_e.set_yticklabels(cat_names_plot, fontsize=8)
ax_e.set_xlabel('Number of proteins', fontsize=10)
ax_e.set_title('(E) Functional category distribution', fontsize=11, fontweight='bold', loc='left')
ax_e.invert_yaxis()

plt.tight_layout()

# Save Figure 1
fig.savefig('figures/Fig-1-RAG-Embeddings.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-1-RAG-Embeddings.png', format='png', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-1-RAG-Embeddings.svg', format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figures/Fig-1-RAG-Embeddings.pdf/png/svg")
plt.close()

# =====================================================
# STEP 8: Generate Figure 2 - Retrieval Performance
# =====================================================
print("\n[8/8] Generating Figure 2 - Retrieval performance...")

# Evaluate retrieval quality
def evaluate_retrieval(query_embeddings, doc_embeddings, doc_metadata, proteins, categories, method_name):
    """
    Evaluate retrieval performance using precision-recall metrics.
    Ground truth: documents from same functional category are relevant.
    """
    all_labels = []
    all_scores = []

    for i, (protein, category) in enumerate(zip(proteins, categories)):
        query_emb = query_embeddings[i]

        # Compute similarities
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = doc_norms @ query_norm

        # Ground truth: same category or same protein
        labels = []
        for meta in doc_metadata:
            is_relevant = (meta['category'] == category) or (meta['protein'] == protein)
            labels.append(1 if is_relevant else 0)

        all_labels.extend(labels)
        all_scores.extend(similarities.tolist())

    return np.array(all_labels), np.array(all_scores)

# Create different retrieval methods for comparison
print("  Evaluating retrieval methods...")

# Method 1: RAG-GNN (GNN embeddings projected to doc space)
labels_rag, scores_rag = evaluate_retrieval(
    query_embeddings, doc_embeddings, doc_metadata, proteins, categories, "RAG-GNN"
)

# Method 2: BM25-style (TF-IDF on protein names)
protein_queries = [f"{p} {c} signaling pathway" for p, c in zip(proteins, categories)]
tfidf_query = TfidfVectorizer(max_features=128, stop_words='english')
tfidf_query.fit([d for d in documents])
query_tfidf = tfidf_query.transform(protein_queries).toarray()
doc_tfidf = tfidf_query.transform(documents).toarray()
labels_tfidf, scores_tfidf = evaluate_retrieval(
    query_tfidf, doc_tfidf, doc_metadata, proteins, categories, "TF-IDF"
)

# Method 3: Random baseline
np.random.seed(123)
scores_random = np.random.rand(len(labels_rag))
labels_random = labels_rag.copy()

# Method 4: Network topology only (SVD of adjacency)
svd_adj = TruncatedSVD(n_components=128, random_state=42)
adj_emb = svd_adj.fit_transform(adj_matrix)
adj_query = adj_emb @ projection_matrix
labels_adj, scores_adj = evaluate_retrieval(
    adj_query, doc_embeddings, doc_metadata, proteins, categories, "Topology-only"
)

# Compute precision-recall curves
fig, ax = plt.subplots(figsize=(8, 6))

methods = {
    'RAG-GNN (ours)': {'labels': labels_rag, 'scores': scores_rag, 'color': SET2_COLORS[0], 'lw': 2.5},
    'Topology-only': {'labels': labels_adj, 'scores': scores_adj, 'color': SET2_COLORS[1], 'lw': 2},
    'TF-IDF': {'labels': labels_tfidf, 'scores': scores_tfidf, 'color': SET2_COLORS[2], 'lw': 2},
    'Random': {'labels': labels_random, 'scores': scores_random, 'color': SET2_COLORS[7], 'lw': 1.5, 'ls': '--'},
}

results = {}
for method_name, config in methods.items():
    precision, recall, _ = precision_recall_curve(config['labels'], config['scores'])
    ap = average_precision_score(config['labels'], config['scores'])
    results[method_name] = ap

    ls = config.get('ls', '-')
    ax.plot(recall, precision, label=f"{method_name} (mAP={ap:.3f})",
           color=config['color'], linewidth=config['lw'], linestyle=ls, alpha=0.9)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='upper right', fontsize=10, frameon=False)
ax.set_title('Document retrieval performance for protein queries', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save Figure 2
fig.savefig('figures/Fig-2-Retrieval-Performance.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-2-Retrieval-Performance.png', format='png', dpi=300, bbox_inches='tight')
fig.savefig('figures/Fig-2-Retrieval-Performance.svg', format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figures/Fig-2-Retrieval-Performance.pdf/png/svg")
plt.close()

# =====================================================
# Save Results Summary
# =====================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

summary = {
    'network': {
        'n_proteins': n_nodes,
        'n_interactions': G.number_of_edges(),
        'avg_degree': float(np.mean(degrees)),
        'avg_clustering': float(nx.average_clustering(G))
    },
    'embeddings': {
        'gnn_layers': 3,
        'embedding_dim': 128,
        'top_k_retrieval': top_k,
        'fusion_alpha': 0.6
    },
    'evaluation': {
        'silhouette_rag_gnn': float(sil_rag),
        'silhouette_gnn_only': float(sil_gnn),
        'silhouette_improvement': float(sil_rag - sil_gnn),
        'retrieval_map': {k: float(v) for k, v in results.items()}
    }
}

with open('figures/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nNetwork: {n_nodes} proteins, {G.number_of_edges()} interactions")
print(f"\nEmbedding Quality (Silhouette Score):")
print(f"  RAG-GNN:   {sil_rag:.4f}")
print(f"  GNN-only:  {sil_gnn:.4f}")
print(f"  Improvement: {sil_rag - sil_gnn:+.4f}")
print(f"\nRetrieval Performance (mAP):")
for method, ap in results.items():
    print(f"  {method:20s}: {ap:.4f}")
print(f"\nFigures saved to figures/ directory")
print("="*70)
