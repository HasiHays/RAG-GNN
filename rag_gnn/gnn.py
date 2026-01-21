"""
Graph Neural Network encoder module.

Implements GCN-style message passing for network topology encoding.
"""

import numpy as np
from typing import Optional, List, Literal
from .utils import normalize_adjacency


class GNNEncoder:
    """
    Graph Neural Network encoder using spectral convolutions.

    Implements GCN-style message passing:
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    Parameters
    ----------
    n_layers : int, default=3
        Number of message passing layers.
    hidden_dim : int, default=128
        Dimension of hidden representations.
    activation : str, default='relu'
        Activation function ('relu', 'tanh', 'elu').
    normalize : bool, default=True
        Whether to apply layer normalization.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    embeddings_ : ndarray of shape (n_nodes, hidden_dim)
        Node embeddings after fitting.
    layer_embeddings_ : list of ndarray
        Embeddings at each layer.

    Examples
    --------
    >>> from rag_gnn import GNNEncoder
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> encoder = GNNEncoder(n_layers=2, hidden_dim=64)
    >>> embeddings = encoder.fit_transform(adj)
    >>> embeddings.shape
    (3, 64)
    """

    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 128,
        activation: Literal['relu', 'tanh', 'elu'] = 'relu',
        normalize: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.normalize = normalize
        self.random_state = random_state

        self.embeddings_ = None
        self.layer_embeddings_ = None
        self._weights = None

    def _get_activation(self):
        """Get activation function."""
        if self.activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh
        elif self.activation == 'elu':
            return lambda x: np.where(x > 0, x, np.exp(x) - 1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _layer_norm(self, H: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = H.mean(axis=1, keepdims=True)
        std = H.std(axis=1, keepdims=True) + 1e-8
        return (H - mean) / std

    def fit(self, adj_matrix: np.ndarray, node_features: Optional[np.ndarray] = None):
        """
        Fit GNN encoder to adjacency matrix.

        Parameters
        ----------
        adj_matrix : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
        node_features : ndarray of shape (n_nodes, n_features), optional
            Initial node features. If None, uses random initialization.

        Returns
        -------
        self : GNNEncoder
            Fitted encoder.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_nodes = adj_matrix.shape[0]

        # Normalize adjacency matrix
        A_norm = normalize_adjacency(adj_matrix)

        # Initialize node features
        if node_features is None:
            H = np.random.randn(n_nodes, self.hidden_dim) * 0.1
        else:
            # Project to hidden dimension if needed
            if node_features.shape[1] != self.hidden_dim:
                W_init = np.random.randn(node_features.shape[1], self.hidden_dim)
                W_init *= np.sqrt(2.0 / node_features.shape[1])
                H = node_features @ W_init
            else:
                H = node_features.copy()

        # Store layer embeddings
        self.layer_embeddings_ = [H.copy()]
        self._weights = []

        # Get activation function
        activation_fn = self._get_activation()

        # Message passing layers
        for layer in range(self.n_layers):
            # Weight matrix (Xavier initialization)
            if self.random_state is not None:
                np.random.seed(self.random_state + layer)
            W = np.random.randn(H.shape[1], self.hidden_dim)
            W *= np.sqrt(2.0 / H.shape[1])
            self._weights.append(W)

            # Aggregate neighbor features
            H_agg = A_norm @ H

            # Transform
            H = H_agg @ W

            # Activation
            H = activation_fn(H)

            # Layer normalization
            if self.normalize:
                H = self._layer_norm(H)

            self.layer_embeddings_.append(H.copy())

        self.embeddings_ = H
        return self

    def fit_transform(self, adj_matrix: np.ndarray, node_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit GNN encoder and return node embeddings.

        Parameters
        ----------
        adj_matrix : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
        node_features : ndarray of shape (n_nodes, n_features), optional
            Initial node features.

        Returns
        -------
        embeddings : ndarray of shape (n_nodes, hidden_dim)
            Node embeddings.
        """
        self.fit(adj_matrix, node_features)
        return self.embeddings_

    def transform(self, adj_matrix: np.ndarray, node_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform using fitted weights.

        Parameters
        ----------
        adj_matrix : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
        node_features : ndarray of shape (n_nodes, n_features), optional
            Initial node features.

        Returns
        -------
        embeddings : ndarray of shape (n_nodes, hidden_dim)
            Node embeddings.
        """
        if self._weights is None:
            raise ValueError("Encoder not fitted. Call fit() first.")

        n_nodes = adj_matrix.shape[0]
        A_norm = normalize_adjacency(adj_matrix)

        # Initialize
        if node_features is None:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            H = np.random.randn(n_nodes, self.hidden_dim) * 0.1
        else:
            if node_features.shape[1] != self.hidden_dim:
                W_init = np.random.randn(node_features.shape[1], self.hidden_dim)
                W_init *= np.sqrt(2.0 / node_features.shape[1])
                H = node_features @ W_init
            else:
                H = node_features.copy()

        activation_fn = self._get_activation()

        # Apply stored weights
        for W in self._weights:
            H_agg = A_norm @ H
            H = H_agg @ W
            H = activation_fn(H)
            if self.normalize:
                H = self._layer_norm(H)

        return H
