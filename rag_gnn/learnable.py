"""
Learnable RAG-GNN model with end-to-end training in PyTorch.

Implements the trainable RAG-GNN framework with:
- Learnable GCN encoder with gradient-optimized weight matrices
- Learnable retrieval projection (two-layer MLP)
- Gated fusion mechanism
- Three-phase curriculum training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LearnableRAGGNN(nn.Module):
    """
    End-to-end trainable RAG-GNN model.

    Parameters
    ----------
    hidden_dim : int, default=128
        Dimension of GNN hidden representations.
    doc_dim : int, default=64
        Dimension of document embeddings.
    n_layers : int, default=3
        Number of GCN message passing layers.
    k : int, default=10
        Number of documents to retrieve per node.
    """

    def __init__(self, hidden_dim=128, doc_dim=64, n_layers=3, k=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.doc_dim = doc_dim
        self.k = k

        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, doc_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + doc_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.transform_ctx = nn.Linear(doc_dim, hidden_dim)

        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def gnn_forward(self, A, X):
        """GCN message passing with 3 layers."""
        H = F.gelu(self.ln1(self.W1(A @ X)))
        H = F.gelu(self.ln2(self.W2(A @ H)))
        H = F.gelu(self.ln3(self.W3(A @ H)))
        return H

    def forward(self, A, X, doc_emb, return_all=False):
        """
        Forward pass.

        Parameters
        ----------
        A : Tensor of shape (n_nodes, n_nodes)
            Normalized adjacency matrix.
        X : Tensor of shape (n_nodes, hidden_dim)
            Node features.
        doc_emb : Tensor of shape (n_docs, doc_dim)
            Document embeddings.
        return_all : bool
            If True, returns intermediate representations.

        Returns
        -------
        fused : Tensor of shape (n_nodes, hidden_dim)
            Fused node embeddings.
        """
        gnn = self.gnn_forward(A, X)

        q = self.proj(gnn)
        scores = q @ doc_emb.T / np.sqrt(self.doc_dim)
        topk_v, topk_i = torch.topk(scores, self.k, dim=1)
        attn = F.softmax(topk_v / 0.5, dim=1)
        ctx = (attn.unsqueeze(-1) * doc_emb[topk_i]).sum(1)

        gate_input = torch.cat([gnn, ctx], dim=1)
        g = self.gate(gate_input)
        ctx_transformed = self.transform_ctx(ctx)
        fused = g * gnn + (1 - g) * ctx_transformed

        if return_all:
            return fused, gnn, ctx, scores, topk_i, g
        return fused
