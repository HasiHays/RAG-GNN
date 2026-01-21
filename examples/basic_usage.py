#!/usr/bin/env python3
"""
Basic usage example for RAG-GNN.

This example demonstrates how to:
1. Create a simple network
2. Generate RAG-GNN embeddings
3. Evaluate embedding quality
"""

import numpy as np
import networkx as nx
from rag_gnn import RAGGNN, GNNEncoder
from rag_gnn.utils import compute_silhouette, evaluate_link_prediction, create_train_test_edges

# Set random seed for reproducibility
np.random.seed(42)

# Create a synthetic network
print("Creating synthetic network...")
n_nodes = 100
n_communities = 4

# Generate community structure
community_labels = np.repeat(np.arange(n_communities), n_nodes // n_communities)
np.random.shuffle(community_labels)

# Create adjacency matrix with community structure
adj_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        # Higher probability of connection within community
        if community_labels[i] == community_labels[j]:
            prob = 0.3
        else:
            prob = 0.05
        if np.random.random() < prob:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

print(f"  Nodes: {n_nodes}")
print(f"  Edges: {int(adj_matrix.sum() / 2)}")
print(f"  Communities: {n_communities}")

# Create simple document corpus
print("\nCreating document corpus...")
community_names = ['Cell Cycle', 'Apoptosis', 'DNA Repair', 'Signaling']
documents = []
for i in range(n_nodes):
    comm = community_labels[i]
    doc = f"Protein {i} is involved in {community_names[comm]} pathway. "
    doc += f"It interacts with other {community_names[comm]} proteins. "
    doc += f"Function relates to cellular {community_names[comm].lower()} processes."
    documents.append(doc)

print(f"  Documents: {len(documents)}")

# Train RAG-GNN
print("\nTraining RAG-GNN...")
model = RAGGNN(
    n_layers=3,
    hidden_dim=64,
    retrieval_k=5,
    fusion_alpha=0.6,
    random_state=42
)

embeddings = model.fit_transform(adj_matrix, documents=documents)
print(f"  Embedding shape: {embeddings.shape}")

# Get GNN-only embeddings for comparison
gnn_embeddings = model.get_gnn_only_embeddings()
print(f"  GNN-only shape: {gnn_embeddings.shape}")

# Evaluate clustering quality
print("\nEvaluating clustering quality...")
rag_silhouette = compute_silhouette(embeddings, community_labels)
gnn_silhouette = compute_silhouette(gnn_embeddings, community_labels)

print(f"  RAG-GNN silhouette: {rag_silhouette:.4f}")
print(f"  GNN-only silhouette: {gnn_silhouette:.4f}")
print(f"  Improvement: {rag_silhouette - gnn_silhouette:+.4f}")

# Evaluate link prediction
print("\nEvaluating link prediction...")
train_edges, test_pos, test_neg, train_adj = create_train_test_edges(
    adj_matrix, test_ratio=0.2, random_state=42
)

# Retrain on training edges
model_lp = RAGGNN(n_layers=3, hidden_dim=64, retrieval_k=5, random_state=42)
emb_lp = model_lp.fit_transform(train_adj, documents=documents)

lp_metrics = evaluate_link_prediction(emb_lp, test_pos, test_neg)
print(f"  AUROC: {lp_metrics['auroc']:.4f}")
print(f"  AUPRC: {lp_metrics['auprc']:.4f}")

# Find similar proteins
print("\nFinding similar proteins to node 0...")
similar = model.get_most_similar(0, top_k=5)
print(f"  Node 0 community: {community_names[community_labels[0]]}")
print("  Most similar nodes:")
for node_idx, sim in similar:
    print(f"    Node {node_idx}: similarity={sim:.4f}, community={community_names[community_labels[node_idx]]}")

print("\nDone!")
