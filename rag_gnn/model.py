"""
Main RAG-GNN model combining GNN encoding, retrieval, and fusion.
"""

import numpy as np
from typing import Optional, List, Union, Literal
from .gnn import GNNEncoder
from .retrieval import KnowledgeRetriever
from .fusion import FusionModule


class RAGGNN:
    """
    Retrieval-Augmented Graph Neural Network for biological networks.

    Combines GNN-based topology encoding with knowledge retrieval to create
    embeddings that capture both structural and functional relationships.

    Parameters
    ----------
    n_layers : int, default=3
        Number of GNN message passing layers.
    hidden_dim : int, default=128
        Dimension of hidden representations.
    retrieval_k : int, default=10
        Number of documents to retrieve per node.
    fusion_alpha : float, default=0.6
        Weight for topology features in fusion (1-alpha for retrieved).
    fusion_method : str, default='weighted_concat'
        Fusion strategy ('weighted_concat', 'gated', 'attention', 'add').
    activation : str, default='relu'
        GNN activation function.
    normalize : bool, default=True
        Whether to normalize embeddings.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    embeddings_ : ndarray of shape (n_nodes, hidden_dim)
        Final RAG-GNN embeddings after fitting.
    gnn_embeddings_ : ndarray of shape (n_nodes, hidden_dim)
        GNN-only embeddings (before fusion).
    retrieved_features_ : ndarray of shape (n_nodes, doc_dim)
        Aggregated retrieved document features.
    gnn_encoder_ : GNNEncoder
        Fitted GNN encoder.
    retriever_ : KnowledgeRetriever
        Fitted knowledge retriever.
    fusion_module_ : FusionModule
        Fusion module.

    Examples
    --------
    >>> from rag_gnn import RAGGNN
    >>> import numpy as np
    >>> adj = np.random.rand(100, 100) > 0.9
    >>> adj = (adj | adj.T).astype(float)
    >>> documents = [f"Document {i}" for i in range(50)]
    >>> model = RAGGNN(n_layers=3, hidden_dim=128)
    >>> embeddings = model.fit_transform(adj, documents=documents)
    >>> embeddings.shape
    (100, 128)
    """

    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 128,
        retrieval_k: int = 10,
        fusion_alpha: float = 0.6,
        fusion_method: Literal['weighted_concat', 'gated', 'attention', 'add'] = 'weighted_concat',
        activation: Literal['relu', 'tanh', 'elu'] = 'relu',
        normalize: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.retrieval_k = retrieval_k
        self.fusion_alpha = fusion_alpha
        self.fusion_method = fusion_method
        self.activation = activation
        self.normalize = normalize
        self.random_state = random_state

        # Components
        self.gnn_encoder_ = None
        self.retriever_ = None
        self.fusion_module_ = None

        # Outputs
        self.embeddings_ = None
        self.gnn_embeddings_ = None
        self.retrieved_features_ = None

    def fit(
        self,
        adj_matrix: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        documents: Optional[Union[List[str], np.ndarray]] = None,
        document_embeddings: Optional[np.ndarray] = None
    ):
        """
        Fit RAG-GNN model to network and document corpus.

        Parameters
        ----------
        adj_matrix : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
        node_features : ndarray of shape (n_nodes, n_features), optional
            Initial node features for GNN.
        documents : list of str or ndarray, optional
            Document corpus for retrieval. Required if document_embeddings
            is not provided.
        document_embeddings : ndarray, optional
            Precomputed document embeddings.

        Returns
        -------
        self : RAGGNN
            Fitted model.
        """
        n_nodes = adj_matrix.shape[0]

        # Step 1: GNN encoding
        self.gnn_encoder_ = GNNEncoder(
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            normalize=self.normalize,
            random_state=self.random_state
        )
        self.gnn_embeddings_ = self.gnn_encoder_.fit_transform(adj_matrix, node_features)

        # Step 2: Knowledge retrieval (if documents provided)
        if documents is not None or document_embeddings is not None:
            self.retriever_ = KnowledgeRetriever(
                embedding_method='tfidf' if documents is not None and isinstance(documents[0], str) else 'precomputed',
                top_k=self.retrieval_k,
                use_neighborhood=True,
                random_state=self.random_state
            )

            if documents is not None:
                self.retriever_.fit(documents, document_embeddings)
            else:
                self.retriever_.fit(document_embeddings, document_embeddings)

            self.retrieved_features_ = self.retriever_.get_retrieved_features(
                self.gnn_embeddings_,
                adj_matrix
            )

            # Step 3: Fusion
            self.fusion_module_ = FusionModule(
                method=self.fusion_method,
                alpha=self.fusion_alpha,
                output_dim=self.hidden_dim,
                normalize=self.normalize,
                random_state=self.random_state
            )
            self.embeddings_ = self.fusion_module_.fuse(
                self.gnn_embeddings_,
                self.retrieved_features_
            )
        else:
            # No retrieval, just use GNN embeddings
            self.embeddings_ = self.gnn_embeddings_
            self.retrieved_features_ = None

        return self

    def fit_transform(
        self,
        adj_matrix: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        documents: Optional[Union[List[str], np.ndarray]] = None,
        document_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit model and return embeddings.

        Parameters
        ----------
        adj_matrix : ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.
        node_features : ndarray of shape (n_nodes, n_features), optional
            Initial node features for GNN.
        documents : list of str or ndarray, optional
            Document corpus for retrieval.
        document_embeddings : ndarray, optional
            Precomputed document embeddings.

        Returns
        -------
        embeddings : ndarray of shape (n_nodes, hidden_dim)
            RAG-GNN node embeddings.
        """
        self.fit(adj_matrix, node_features, documents, document_embeddings)
        return self.embeddings_

    def get_gnn_only_embeddings(self) -> np.ndarray:
        """
        Get GNN-only embeddings (without retrieval fusion).

        Returns
        -------
        gnn_embeddings : ndarray of shape (n_nodes, hidden_dim)
            Node embeddings from GNN encoder only.
        """
        if self.gnn_embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.gnn_embeddings_

    def get_retrieved_features(self) -> Optional[np.ndarray]:
        """
        Get aggregated retrieved document features.

        Returns
        -------
        retrieved_features : ndarray or None
            Retrieved features if documents were provided during fit.
        """
        return self.retrieved_features_

    def retrieve_documents(self, node_indices: List[int]) -> List[List[int]]:
        """
        Get retrieved document indices for specific nodes.

        Parameters
        ----------
        node_indices : list of int
            Indices of nodes to retrieve documents for.

        Returns
        -------
        doc_indices : list of list of int
            Retrieved document indices for each node.
        """
        if self.retriever_ is None:
            raise ValueError("No retriever available. Fit with documents first.")

        all_retrieved = self.retriever_.retrieve(self.gnn_embeddings_)
        return [all_retrieved[i] for i in node_indices]

    def compute_similarity(self, node_i: int, node_j: int) -> float:
        """
        Compute embedding similarity between two nodes.

        Parameters
        ----------
        node_i : int
            Index of first node.
        node_j : int
            Index of second node.

        Returns
        -------
        similarity : float
            Cosine similarity between node embeddings.
        """
        if self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        emb_i = self.embeddings_[node_i]
        emb_j = self.embeddings_[node_j]

        norm_i = np.linalg.norm(emb_i)
        norm_j = np.linalg.norm(emb_j)

        if norm_i == 0 or norm_j == 0:
            return 0.0

        return np.dot(emb_i, emb_j) / (norm_i * norm_j)

    def get_most_similar(self, node_idx: int, top_k: int = 5) -> List[tuple]:
        """
        Get most similar nodes to a given node.

        Parameters
        ----------
        node_idx : int
            Index of query node.
        top_k : int, default=5
            Number of similar nodes to return.

        Returns
        -------
        similar_nodes : list of (int, float)
            List of (node_index, similarity) tuples.
        """
        if self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        query = self.embeddings_[node_idx]
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            return []

        # Compute similarities to all nodes
        norms = np.linalg.norm(self.embeddings_, axis=1)
        similarities = np.dot(self.embeddings_, query) / (norms * query_norm + 1e-8)
        similarities[node_idx] = -1  # Exclude self

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
