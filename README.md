# RAG-GNN: Integrating Retrieved Knowledge with Graph Neural Networks for Precision Medicine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/)

A framework for integrating graph neural network representations with retrieval-augmented generation for biological network modeling and precision medicine applications.

## Overview

RAG-GNN combines network topology encoding via graph neural networks with dynamically retrieved literature-derived knowledge to create embeddings that capture both structural and functional relationships in biological networks.

**Key features:**
- GNN-based network topology encoding with message passing
- Knowledge retrieval from biomedical corpora
- Weighted fusion of structural and semantic information
- Comprehensive evaluation metrics for biological networks
- Application to cancer signaling pathway analysis

## Installation

### Using pip

```bash
pip install rag-gnn
```

### Using conda

```bash
conda create -n rag-gnn python=3.9
conda activate rag-gnn
pip install rag-gnn
```

### From source

```bash
git clone https://github.com/hasihays/RAG-GNN.git
cd RAG-GNN
pip install -e .
```

## Quick start

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

## Benchmarking

Compare RAG-GNN against baseline methods:

```python
from rag_gnn.benchmarks import run_benchmark

results = run_benchmark(
    adj_matrix=adj_matrix,
    labels=functional_categories,
    methods=['RAG-GNN', 'GNN-only', 'GCN', 'GAT', 'GraphSAGE',
             'DeepWalk', 'Node2Vec', 'LINE', 'Spectral'],
    tasks=['silhouette', 'link_prediction', 'node_classification']
)

print(results.to_dataframe())
```

## Example: Cancer signaling network analysis

```python
from rag_gnn import RAGGNN
from rag_gnn.data import load_cancer_network
from rag_gnn.evaluation import evaluate_embeddings

# Load cancer signaling network (379 proteins, 3,498 interactions)
adj_matrix, proteins, categories = load_cancer_network()

# Create knowledge base from functional annotations
knowledge_base = create_pathway_documents(proteins, categories)

# Train RAG-GNN
model = RAGGNN(n_layers=3, hidden_dim=128, retrieval_k=10)
embeddings = model.fit_transform(adj_matrix, documents=knowledge_base)

# Evaluate
metrics = evaluate_embeddings(
    embeddings,
    categories,
    metrics=['silhouette', 'link_auroc', 'node_classification']
)
print(f"Silhouette score: {metrics['silhouette']:.3f}")
print(f"Link prediction AUROC: {metrics['link_auroc']:.3f}")
```

## Results

Benchmarking on cancer signaling networks reveals task-specific strengths:

| Method | Silhouette | Link Pred AUROC | Node Class AUROC |
|--------|------------|-----------------|------------------|
| **RAG-GNN** | **0.001** | 0.804 | 0.520 |
| GNN-only | -0.049 | **0.975** | 0.542 |
| GCN | -0.031 | 0.983 | 0.527 |
| DeepWalk | -0.040 | 0.785 | 0.474 |
| Node2Vec | -0.043 | 0.743 | 0.477 |

**Key finding:** RAG-GNN is the only method achieving positive silhouette scores for functional clustering, while topology-focused methods excel at link prediction.

## Citation

If you use RAG-GNN in your research, please cite:

```bibtex
@article{hays2025rag,
  title={RAG-GNN: Integrating Retrieved Knowledge with Graph Neural Networks for Precision Medicine},
  author={Hays, Hasi and Richardson, William J.},
  journal={},
  year={2025},
  doi={}
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

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.
