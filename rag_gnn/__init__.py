"""
RAG-GNN: Retrieval-Augmented Graph Neural Networks for Biological Network Analysis

A framework for integrating graph neural network representations with
retrieval-augmented generation for biological network modeling.
"""

from .model import RAGGNN
from .gnn import GNNEncoder
from .retrieval import KnowledgeRetriever
from .fusion import FusionModule
from .utils import normalize_adjacency, compute_silhouette

try:
    from .learnable import LearnableRAGGNN
except ImportError:
    LearnableRAGGNN = None

__version__ = "0.2.0"
__author__ = "Hasi Hays"
__email__ = "hasih@uark.edu"

__all__ = [
    "RAGGNN",
    "LearnableRAGGNN",
    "GNNEncoder",
    "KnowledgeRetriever",
    "FusionModule",
    "normalize_adjacency",
    "compute_silhouette",
]
