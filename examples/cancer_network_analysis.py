#!/usr/bin/env python3
"""
Cancer signaling network analysis with RAG-GNN.

This example demonstrates the full workflow used in the paper:
1. Load protein interaction network
2. Create functional knowledge base
3. Train RAG-GNN and baseline methods
4. Benchmark comparison
5. Case study: DDR1 analysis
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from rag_gnn import RAGGNN, GNNEncoder
from rag_gnn.utils import (
    normalize_adjacency,
    compute_silhouette,
    create_train_test_edges,
    evaluate_link_prediction
)

np.random.seed(42)


def create_knowledge_base(proteins, categories, n_docs_per_protein=5):
    """Create functional knowledge base from protein annotations."""
    documents = []

    pathway_knowledge = {
        'Cell cycle': [
            "regulates G1/S phase transition through cyclin-dependent kinase activation",
            "controls mitotic checkpoint and spindle assembly",
            "mediates DNA replication licensing and origin firing",
        ],
        'Apoptosis': [
            "induces mitochondrial outer membrane permeabilization",
            "activates caspase cascade through death receptor signaling",
            "regulates BCL2 family protein interactions",
        ],
        'DNA repair': [
            "participates in homologous recombination repair",
            "mediates non-homologous end joining pathway",
            "activates ATM/ATR checkpoint signaling",
        ],
        'RTK signaling': [
            "transduces extracellular growth factor signals",
            "activates downstream MAPK and PI3K cascades",
            "undergoes ligand-induced dimerization",
        ],
    }

    for protein, category in zip(proteins, categories):
        base_cat = category.split('_')[0] if '_' in category else category
        templates = pathway_knowledge.get(base_cat, [
            f"participates in {category} biological processes",
            f"functions in {category} cellular pathway",
        ])

        for template in templates[:n_docs_per_protein]:
            doc = f"{protein} {template}"
            documents.append(doc)

    return documents


def generate_baseline_embeddings(adj_matrix, embedding_dim=128):
    """Generate embeddings using baseline methods."""
    n_nodes = adj_matrix.shape[0]
    embeddings = {}

    # Normalized adjacency
    degrees = np.sum(adj_matrix, axis=1) + 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    A_norm = D_inv_sqrt @ (adj_matrix + np.eye(n_nodes)) @ D_inv_sqrt

    # Transition matrix
    D_inv = np.diag(1.0 / (np.sum(adj_matrix, axis=1) + 1e-10))
    P = D_inv @ adj_matrix

    # Spectral
    adj_sparse = csr_matrix(adj_matrix)
    n_comp = min(embedding_dim, n_nodes - 2)
    U, S, Vt = svds(adj_sparse.astype(float), k=n_comp)
    embeddings['Spectral'] = StandardScaler().fit_transform(U * S)

    # DeepWalk
    walk_matrix = P + P @ P + P @ P @ P
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    embeddings['DeepWalk'] = StandardScaler().fit_transform(svd.fit_transform(walk_matrix))

    # Node2Vec
    P2 = P @ P
    node2vec_matrix = 0.5 * P + 0.3 * P2 + 0.2 * (P2 @ P)
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    embeddings['Node2Vec'] = StandardScaler().fit_transform(svd.fit_transform(node2vec_matrix))

    # LINE
    line_matrix = 0.5 * adj_matrix + 0.5 * (adj_matrix @ adj_matrix)
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    embeddings['LINE'] = StandardScaler().fit_transform(svd.fit_transform(line_matrix))

    # GCN
    np.random.seed(42)
    H = np.random.randn(n_nodes, embedding_dim)
    for _ in range(3):
        H = np.tanh(A_norm @ H)
    embeddings['GCN'] = StandardScaler().fit_transform(H)

    return embeddings


def run_benchmark(embeddings_dict, labels, adj_matrix):
    """Run benchmark evaluation."""
    results = []

    # Prepare link prediction data
    train_edges, test_pos, test_neg, train_adj = create_train_test_edges(
        adj_matrix, test_ratio=0.2, random_state=42
    )

    for name, emb in embeddings_dict.items():
        # Silhouette score
        sil = compute_silhouette(emb, labels)

        # Link prediction
        lp = evaluate_link_prediction(emb, test_pos, test_neg)

        results.append({
            'Method': name,
            'Silhouette': sil,
            'Link Pred AUROC': lp['auroc'],
            'Link Pred AUPRC': lp['auprc'],
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("CANCER SIGNALING NETWORK ANALYSIS WITH RAG-GNN")
    print("=" * 70)

    # Create synthetic cancer network (similar to paper)
    print("\n[1/5] Creating cancer signaling network...")
    n_proteins = 379
    n_categories = 13

    # Assign proteins to categories
    category_names = [
        'Cell cycle', 'Apoptosis', 'DNA repair', 'RTK signaling',
        'PI3K-AKT-MTOR', 'MAPK signaling', 'Wnt signaling', 'TGF-beta',
        'Notch signaling', 'Hedgehog', 'NF-kB', 'JAK-STAT', 'Metabolism'
    ]
    category_labels = np.random.randint(0, n_categories, n_proteins)
    categories = [category_names[i] for i in category_labels]
    proteins = [f"PROT{i:03d}" for i in range(n_proteins)]

    # Create network with pathway structure
    adj_matrix = np.zeros((n_proteins, n_proteins))
    for i in range(n_proteins):
        for j in range(i + 1, n_proteins):
            if category_labels[i] == category_labels[j]:
                prob = 0.15  # Intra-pathway
            else:
                prob = 0.02  # Inter-pathway
            if np.random.random() < prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    n_edges = int(adj_matrix.sum() / 2)
    print(f"  Proteins: {n_proteins}")
    print(f"  Interactions: {n_edges}")
    print(f"  Categories: {n_categories}")

    # Create knowledge base
    print("\n[2/5] Creating knowledge base...")
    documents = create_knowledge_base(proteins, categories)
    print(f"  Documents: {len(documents)}")

    # Train RAG-GNN
    print("\n[3/5] Training RAG-GNN...")
    model = RAGGNN(
        n_layers=3,
        hidden_dim=128,
        retrieval_k=10,
        fusion_alpha=0.6,
        random_state=42
    )
    rag_emb = model.fit_transform(adj_matrix, documents=documents)
    gnn_emb = model.get_gnn_only_embeddings()

    print(f"  RAG-GNN embedding shape: {rag_emb.shape}")

    # Generate baseline embeddings
    print("\n[4/5] Generating baseline embeddings...")
    baseline_emb = generate_baseline_embeddings(adj_matrix, embedding_dim=128)

    # Combine all embeddings
    all_embeddings = {
        'RAG-GNN': rag_emb,
        'GNN-only': gnn_emb,
        **baseline_emb
    }

    # Run benchmark
    print("\n[5/5] Running benchmark evaluation...")
    results = run_benchmark(all_embeddings, category_labels, adj_matrix)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    rag_sil = results[results['Method'] == 'RAG-GNN']['Silhouette'].values[0]
    gnn_sil = results[results['Method'] == 'GNN-only']['Silhouette'].values[0]

    print(f"\n1. Functional clustering (Silhouette score):")
    print(f"   RAG-GNN: {rag_sil:.4f}")
    print(f"   GNN-only: {gnn_sil:.4f}")
    print(f"   Improvement: {rag_sil - gnn_sil:+.4f}")

    best_lp = results.loc[results['Link Pred AUROC'].idxmax()]
    print(f"\n2. Link prediction (best method):")
    print(f"   {best_lp['Method']}: AUROC = {best_lp['Link Pred AUROC']:.4f}")

    # DDR1-like case study
    print("\n" + "=" * 70)
    print("CASE STUDY: TARGET IDENTIFICATION")
    print("=" * 70)

    # Simulate DDR1-like protein
    ddr1_idx = 0
    print(f"\nQuery protein: {proteins[ddr1_idx]} ({categories[ddr1_idx]})")

    similar = model.get_most_similar(ddr1_idx, top_k=5)
    print("\nTop 5 most similar proteins by RAG-GNN embedding:")
    for idx, sim in similar:
        print(f"  {proteins[idx]} ({categories[idx]}): similarity = {sim:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
