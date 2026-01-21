"""
Utility functions for RAG-GNN.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import silhouette_score as sklearn_silhouette


def normalize_adjacency(adj_matrix: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Compute normalized adjacency matrix for GCN-style convolution.

    Computes: D^(-1/2) * A * D^(-1/2) where A includes self-loops.

    Parameters
    ----------
    adj_matrix : ndarray of shape (n_nodes, n_nodes)
        Input adjacency matrix.
    add_self_loops : bool, default=True
        Whether to add self-loops before normalization.

    Returns
    -------
    A_norm : ndarray of shape (n_nodes, n_nodes)
        Normalized adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> A_norm = normalize_adjacency(adj)
    >>> A_norm.shape
    (3, 3)
    """
    n_nodes = adj_matrix.shape[0]

    # Add self-loops
    if add_self_loops:
        A = adj_matrix + np.eye(n_nodes)
    else:
        A = adj_matrix.copy()

    # Compute degree matrix
    degrees = np.sum(A, axis=1)

    # Avoid division by zero
    degrees_inv_sqrt = np.power(degrees + 1e-10, -0.5)

    # D^(-1/2)
    D_inv_sqrt = np.diag(degrees_inv_sqrt)

    # Normalized adjacency
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute silhouette score for embeddings.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, n_features)
        Sample embeddings.
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    metric : str, default='euclidean'
        Distance metric to use.

    Returns
    -------
    score : float
        Silhouette score (range: -1 to 1).

    Examples
    --------
    >>> import numpy as np
    >>> emb = np.random.randn(100, 64)
    >>> labels = np.random.randint(0, 5, 100)
    >>> score = compute_silhouette(emb, labels)
    """
    # Ensure at least 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    return sklearn_silhouette(embeddings, labels, metric=metric)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input matrix.

    Returns
    -------
    sim_matrix : ndarray of shape (n_samples, n_samples)
        Cosine similarity matrix.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    X_norm = X / norms
    return X_norm @ X_norm.T


def create_train_test_edges(
    adj_matrix: np.ndarray,
    test_ratio: float = 0.2,
    negative_ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split edges into train/test sets for link prediction.

    Parameters
    ----------
    adj_matrix : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix.
    test_ratio : float, default=0.2
        Fraction of edges for testing.
    negative_ratio : float, default=1.0
        Ratio of negative to positive samples.
    random_state : int, optional
        Random seed.

    Returns
    -------
    train_edges : ndarray of shape (n_train, 2)
        Training edge indices.
    test_edges_pos : ndarray of shape (n_test, 2)
        Positive test edges.
    test_edges_neg : ndarray of shape (n_neg, 2)
        Negative test edges.
    train_adj : ndarray of shape (n_nodes, n_nodes)
        Training adjacency matrix with test edges removed.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get all edges (upper triangle to avoid duplicates)
    edges = np.array(np.triu(adj_matrix, k=1).nonzero()).T
    n_edges = len(edges)

    # Shuffle and split
    perm = np.random.permutation(n_edges)
    n_test = int(n_edges * test_ratio)

    test_indices = perm[:n_test]
    train_indices = perm[n_test:]

    test_edges_pos = edges[test_indices]
    train_edges = edges[train_indices]

    # Create training adjacency (remove test edges)
    train_adj = adj_matrix.copy()
    for i, j in test_edges_pos:
        train_adj[i, j] = 0
        train_adj[j, i] = 0

    # Sample negative edges
    n_neg = int(len(test_edges_pos) * negative_ratio)
    n_nodes = adj_matrix.shape[0]

    neg_edges = []
    while len(neg_edges) < n_neg:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j and adj_matrix[i, j] == 0:
            neg_edges.append([i, j])

    test_edges_neg = np.array(neg_edges)

    return train_edges, test_edges_pos, test_edges_neg, train_adj


def edge_prediction_scores(
    embeddings: np.ndarray,
    edges: np.ndarray,
    method: str = 'dot'
) -> np.ndarray:
    """
    Compute edge prediction scores.

    Parameters
    ----------
    embeddings : ndarray of shape (n_nodes, n_features)
        Node embeddings.
    edges : ndarray of shape (n_edges, 2)
        Edge indices.
    method : str, default='dot'
        Scoring method ('dot', 'cosine', 'l2').

    Returns
    -------
    scores : ndarray of shape (n_edges,)
        Edge prediction scores.
    """
    src_emb = embeddings[edges[:, 0]]
    dst_emb = embeddings[edges[:, 1]]

    if method == 'dot':
        scores = np.sum(src_emb * dst_emb, axis=1)
    elif method == 'cosine':
        src_norm = np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-8
        dst_norm = np.linalg.norm(dst_emb, axis=1, keepdims=True) + 1e-8
        scores = np.sum((src_emb / src_norm) * (dst_emb / dst_norm), axis=1)
    elif method == 'l2':
        # Negative L2 distance (higher is more similar)
        scores = -np.linalg.norm(src_emb - dst_emb, axis=1)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return scores


def evaluate_link_prediction(
    embeddings: np.ndarray,
    test_edges_pos: np.ndarray,
    test_edges_neg: np.ndarray,
    method: str = 'dot'
) -> dict:
    """
    Evaluate link prediction performance.

    Parameters
    ----------
    embeddings : ndarray of shape (n_nodes, n_features)
        Node embeddings.
    test_edges_pos : ndarray of shape (n_pos, 2)
        Positive test edges.
    test_edges_neg : ndarray of shape (n_neg, 2)
        Negative test edges.
    method : str, default='dot'
        Scoring method.

    Returns
    -------
    metrics : dict
        Dictionary with AUROC and AUPRC scores.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Get scores
    pos_scores = edge_prediction_scores(embeddings, test_edges_pos, method)
    neg_scores = edge_prediction_scores(embeddings, test_edges_neg, method)

    # Combine
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # Compute metrics
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return {'auroc': auroc, 'auprc': auprc}
