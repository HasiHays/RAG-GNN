"""
Fusion module for combining GNN embeddings with retrieved knowledge.
"""

import numpy as np
from typing import Optional, Literal
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class FusionModule:
    """
    Module for fusing GNN embeddings with retrieved document features.

    Supports multiple fusion strategies including weighted concatenation,
    gated fusion, and attention-based fusion.

    Parameters
    ----------
    method : str, default='weighted_concat'
        Fusion method ('weighted_concat', 'gated', 'attention', 'add').
    alpha : float, default=0.6
        Weight for topology features (1-alpha for retrieved features).
    output_dim : int, optional
        Output dimension after fusion. If None, keeps concatenated dimension.
    normalize : bool, default=True
        Whether to normalize output embeddings.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    fused_embeddings_ : ndarray
        Fused embeddings after calling fuse().

    Examples
    --------
    >>> from rag_gnn import FusionModule
    >>> import numpy as np
    >>> gnn_emb = np.random.randn(100, 128)
    >>> retrieved_emb = np.random.randn(100, 64)
    >>> fusion = FusionModule(alpha=0.6, output_dim=128)
    >>> fused = fusion.fuse(gnn_emb, retrieved_emb)
    >>> fused.shape
    (100, 128)
    """

    def __init__(
        self,
        method: Literal['weighted_concat', 'gated', 'attention', 'add'] = 'weighted_concat',
        alpha: float = 0.6,
        output_dim: Optional[int] = None,
        normalize: bool = True,
        random_state: Optional[int] = None
    ):
        self.method = method
        self.alpha = alpha
        self.output_dim = output_dim
        self.normalize = normalize
        self.random_state = random_state

        self.fused_embeddings_ = None
        self._svd = None
        self._scaler = None

    def fuse(
        self,
        gnn_embeddings: np.ndarray,
        retrieved_features: np.ndarray
    ) -> np.ndarray:
        """
        Fuse GNN embeddings with retrieved document features.

        Parameters
        ----------
        gnn_embeddings : ndarray of shape (n_nodes, gnn_dim)
            Node embeddings from GNN encoder.
        retrieved_features : ndarray of shape (n_nodes, retrieved_dim)
            Aggregated features from retrieved documents.

        Returns
        -------
        fused_embeddings : ndarray of shape (n_nodes, output_dim)
            Fused node representations.
        """
        n_nodes = gnn_embeddings.shape[0]

        # Normalize inputs
        gnn_norm = StandardScaler().fit_transform(gnn_embeddings)
        ret_norm = StandardScaler().fit_transform(retrieved_features)

        if self.method == 'weighted_concat':
            # Weighted concatenation
            fused = np.hstack([
                self.alpha * gnn_norm,
                (1 - self.alpha) * ret_norm
            ])

        elif self.method == 'add':
            # Additive fusion (requires same dimensions)
            if gnn_embeddings.shape[1] != retrieved_features.shape[1]:
                # Project to same dimension
                min_dim = min(gnn_embeddings.shape[1], retrieved_features.shape[1])
                gnn_proj = gnn_norm[:, :min_dim]
                ret_proj = ret_norm[:, :min_dim]
            else:
                gnn_proj = gnn_norm
                ret_proj = ret_norm

            fused = self.alpha * gnn_proj + (1 - self.alpha) * ret_proj

        elif self.method == 'gated':
            # Gated fusion with learned gates
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # Simple gate based on feature norms
            gnn_norms = np.linalg.norm(gnn_norm, axis=1, keepdims=True)
            ret_norms = np.linalg.norm(ret_norm, axis=1, keepdims=True)
            gates = gnn_norms / (gnn_norms + ret_norms + 1e-8)

            # Pad to same dimension
            max_dim = max(gnn_embeddings.shape[1], retrieved_features.shape[1])
            gnn_padded = np.zeros((n_nodes, max_dim))
            ret_padded = np.zeros((n_nodes, max_dim))
            gnn_padded[:, :gnn_norm.shape[1]] = gnn_norm
            ret_padded[:, :ret_norm.shape[1]] = ret_norm

            fused = gates * gnn_padded + (1 - gates) * ret_padded

        elif self.method == 'attention':
            # Attention-based fusion
            # Compute attention scores based on similarity
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # Project to common space
            common_dim = min(gnn_embeddings.shape[1], retrieved_features.shape[1])
            W_gnn = np.random.randn(gnn_embeddings.shape[1], common_dim) * 0.1
            W_ret = np.random.randn(retrieved_features.shape[1], common_dim) * 0.1

            gnn_proj = gnn_norm @ W_gnn
            ret_proj = ret_norm @ W_ret

            # Compute attention weights
            attn_scores = np.sum(gnn_proj * ret_proj, axis=1, keepdims=True)
            attn_weights = 1 / (1 + np.exp(-attn_scores))  # Sigmoid

            fused = attn_weights * gnn_proj + (1 - attn_weights) * ret_proj

        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        # Dimensionality reduction if output_dim specified
        if self.output_dim is not None and fused.shape[1] != self.output_dim:
            n_components = min(self.output_dim, fused.shape[1] - 1, n_nodes - 1)
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            fused = self._svd.fit_transform(fused)

            # Pad if needed
            if fused.shape[1] < self.output_dim:
                padded = np.zeros((n_nodes, self.output_dim))
                padded[:, :fused.shape[1]] = fused
                fused = padded

        # Final normalization
        if self.normalize:
            self._scaler = StandardScaler()
            fused = self._scaler.fit_transform(fused)

        self.fused_embeddings_ = fused
        return fused

    def transform(
        self,
        gnn_embeddings: np.ndarray,
        retrieved_features: np.ndarray
    ) -> np.ndarray:
        """
        Transform new data using fitted fusion parameters.

        Parameters
        ----------
        gnn_embeddings : ndarray of shape (n_nodes, gnn_dim)
            Node embeddings from GNN encoder.
        retrieved_features : ndarray of shape (n_nodes, retrieved_dim)
            Aggregated features from retrieved documents.

        Returns
        -------
        fused_embeddings : ndarray of shape (n_nodes, output_dim)
            Fused node representations.
        """
        # For simple methods, just call fuse again
        # For methods with learned parameters, would use stored weights
        return self.fuse(gnn_embeddings, retrieved_features)
