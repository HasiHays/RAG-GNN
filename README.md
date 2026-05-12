---
noteId: "4108c8004e2f11f1a2570378ad05fce2"
tags: []

---

# RAG-GNN: Integrating Retrieved Knowledge with Graph Neural Networks for Precision Medicine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for integrating graph neural network representations with retrieval-augmented generation for biological network modeling and precision medicine applications.

## Overview

RAG-GNN combines network topology encoding via graph neural networks with dynamically retrieved literature-derived knowledge to create embeddings that capture both structural and functional relationships in biological networks.

**Key features:**
- GNN-based network topology encoding with message passing
- Knowledge retrieval from biomedical corpora
- Weighted fusion of structural and semantic information
- End-to-end learnable model with gated fusion and curriculum training (PyTorch)
- Comprehensive evaluation metrics for biological networks
- Application to cancer signaling pathway analysis

## Installation

### From source

```bash
git clone https://github.com/hasihays/RAG-GNN.git
cd RAG-GNN
pip install -e .
```

### With learnable (PyTorch) support

```bash
pip install -e ".[learnable]"
```

### Development install

```bash
pip install -e ".[dev]"
```

## Quick start

### Analytical pipeline

```python
import numpy as np
import networkx as nx
from rag_gnn import RAGGNN, GNNEncoder, KnowledgeRetriever

# Load your network
G = nx.read_edgelist('protein_network.edgelist')
adj_matrix = nx.to_numpy_array(G)

# Initialize model
model = RAGGNN(
    n_layers=3,
    hidden_dim=128,
    retrieval_k=10,
    fusion_alpha=0.6
)

# Generate embeddings
embeddings = model.fit_transform(adj_matrix, documents=knowledge_base)

# Access GNN-only embeddings
gnn_embeddings = model.gnn_embeddings_

# Access retrieved features
retrieved_features = model.retrieved_features_
```

### Learnable pipeline (PyTorch)

```python
import torch
from rag_gnn import LearnableRAGGNN

# Initialize end-to-end trainable model
model = LearnableRAGGNN(
    hidden_dim=128,
    doc_dim=64,
    n_layers=3,
    k=10
)

# Forward pass
# A: normalized adjacency (n_nodes, n_nodes)
# X: node features (n_nodes, hidden_dim)
# doc_emb: document embeddings (n_docs, doc_dim)
fused = model(A, X, doc_emb)

# With intermediate outputs
fused, gnn_emb, ctx, scores, topk_idx, gate = model(A, X, doc_emb, return_all=True)
# gate values indicate fusion balance (>0.5 = topology-dominant)
```

The learnable pipeline uses three-phase curriculum training:
1. **Phase 1** (GNN warm-up): Train the GCN encoder with link prediction
2. **Phase 2** (Retrieval alignment): Train retrieval projection with contrastive loss
3. **Phase 3** (Joint fine-tuning): End-to-end optimization of all components

See `examples/learnable_cancer_network.py` for a complete training script.

## Architecture

The RAG-GNN framework consists of three main components:

### 1. GNN encoder
Implements spectral graph convolutions through neighborhood aggregation:

```python
from rag_gnn import GNNEncoder

encoder = GNNEncoder(
    n_layers=3,
    hidden_dim=128,
    activation='relu',
    normalize=True
)
node_embeddings = encoder.fit_transform(adj_matrix)
```

### 2. Knowledge retriever
Retrieves relevant documents for each node based on semantic similarity:

```python
from rag_gnn import KnowledgeRetriever

retriever = KnowledgeRetriever(
    embedding_method='tfidf',
    top_k=10
)
retrieved_docs = retriever.retrieve(node_embeddings, document_corpus)
```

### 3. Fusion module
Combines GNN embeddings with retrieved knowledge:

```python
from rag_gnn import FusionModule

fusion = FusionModule(
    method='weighted_concat',
    alpha=0.6,  # Weight for topology features
    output_dim=128
)
fused_embeddings = fusion.fuse(gnn_embeddings, retrieved_features)
```

In the learnable pipeline, fusion uses a learned gating mechanism that dynamically balances topology and retrieval contributions (mean gate ~0.59, indicating 59% topology / 41% retrieval).

## Results

Evaluation on a cancer signaling network (379 proteins, 3,498 interactions, 14 functional categories, 1,895 documents) with 10-seed cross-validation:

| Method | Silhouette | NMI | ARI | Link Pred AUROC |
|--------|------------|-----|-----|-----------------|
| **RAG-GNN** | **-0.144 +/- 0.066** | **0.244 +/- 0.032** | **0.083 +/- 0.029** | 0.822 +/- 0.063 |
| GNN-only | -0.237 +/- 0.065 | 0.242 +/- 0.032 | 0.061 +/- 0.017 | 0.774 +/- 0.095 |
| GCN | -0.094 +/- 0.009 | 0.278 +/- 0.010 | 0.066 +/- 0.008 | **0.962 +/- 0.006** |
| Spectral | -0.085 +/- 0.000 | 0.275 +/- 0.011 | 0.056 +/- 0.007 | 0.977 +/- 0.002 |
| DeepWalk | -0.066 +/- 0.000 | 0.273 +/- 0.009 | 0.060 +/- 0.005 | 0.949 +/- 0.002 |
| Node2Vec | -0.062 +/- 0.000 | 0.265 +/- 0.018 | 0.054 +/- 0.011 | 0.950 +/- 0.003 |
| GAT | -0.063 +/- 0.006 | 0.196 +/- 0.020 | 0.036 +/- 0.009 | 0.806 +/- 0.013 |
| GraphSAGE | -0.019 +/- 0.002 | 0.105 +/- 0.008 | 0.003 +/- 0.002 | 0.556 +/- 0.020 |

**Key finding:** RAG-GNN achieves the highest ARI for functional clustering and the best link prediction AUROC among learnable methods, demonstrating that knowledge retrieval provides complementary signal to network topology.

## Citation

If you use RAG-GNN in your research, please cite:

```bibtex
@article{hays2025rag,
  title={RAG-GNN: Integrating Retrieved Knowledge with Graph Neural Networks for Precision Medicine},
  author={Hays, Hasi and Richardson, William J.},
  journal={Frontiers in Artificial Intelligence},
  year={2025}
}
```

## Funding

This study was supported by:
- National Institutes of Health (NIGMS R01GM157589)
- Department of Defense (DEPSCoR FA9550-22-1-0379)

## Authors

- **Hasi Hays** - [ORCID](https://orcid.org/0000-0003-0843-050X) - Conceptualization, model development, methodology, coding, simulations, analysis and writing
- **William J. Richardson** - [ORCID](https://orcid.org/0000-0001-8678-9716) - Review, editing, funding acquisition, resources, and supervision

Department of Chemical Engineering, University of Arkansas, Fayetteville, AR 72701, USA

**Correspondence:** hasih@uark.edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
