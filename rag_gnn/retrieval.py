"""
Knowledge retrieval module for RAG-GNN.

Implements document retrieval based on node embeddings and semantic similarity.
"""

import numpy as np
from typing import Optional, List, Dict, Union, Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class KnowledgeRetriever:
    """
    Knowledge retrieval system for biological networks.

    Retrieves relevant documents for each node based on semantic similarity
    between node embeddings and document embeddings.

    Parameters
    ----------
    embedding_method : str, default='tfidf'
        Method for document embedding ('tfidf', 'precomputed').
    top_k : int, default=10
        Number of documents to retrieve per node.
    use_neighborhood : bool, default=True
        Whether to use neighborhood-aware retrieval scoring.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    document_embeddings_ : ndarray
        Embeddings of documents in the corpus.
    vectorizer_ : TfidfVectorizer
        Fitted TF-IDF vectorizer (if using tfidf method).

    Examples
    --------
    >>> from rag_gnn import KnowledgeRetriever
    >>> documents = ["protein A regulates cell cycle", "protein B induces apoptosis"]
    >>> retriever = KnowledgeRetriever(top_k=1)
    >>> retriever.fit(documents)
    >>> node_emb = np.random.randn(3, 64)
    >>> retrieved = retriever.retrieve(node_emb)
    """

    def __init__(
        self,
        embedding_method: Literal['tfidf', 'precomputed'] = 'tfidf',
        top_k: int = 10,
        use_neighborhood: bool = True,
        random_state: Optional[int] = None
    ):
        self.embedding_method = embedding_method
        self.top_k = top_k
        self.use_neighborhood = use_neighborhood
        self.random_state = random_state

        self.document_embeddings_ = None
        self.vectorizer_ = None
        self._documents = None

    def fit(
        self,
        documents: Union[List[str], np.ndarray],
        precomputed_embeddings: Optional[np.ndarray] = None
    ):
        """
        Fit retriever to document corpus.

        Parameters
        ----------
        documents : list of str or ndarray
            Document corpus. If embedding_method='precomputed',
            this should be document embeddings directly.
        precomputed_embeddings : ndarray, optional
            Precomputed document embeddings.

        Returns
        -------
        self : KnowledgeRetriever
            Fitted retriever.
        """
        if self.embedding_method == 'tfidf':
            if not isinstance(documents[0], str):
                raise ValueError("For 'tfidf' method, documents must be strings.")

            self._documents = documents
            self.vectorizer_ = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.document_embeddings_ = self.vectorizer_.fit_transform(documents).toarray()

        elif self.embedding_method == 'precomputed':
            if precomputed_embeddings is not None:
                self.document_embeddings_ = precomputed_embeddings
            else:
                self.document_embeddings_ = np.array(documents)
            self._documents = None

        return self

    def retrieve(
        self,
        node_embeddings: np.ndarray,
        adj_matrix: Optional[np.ndarray] = None,
        return_scores: bool = False
    ) -> Union[List[List[int]], tuple]:
        """
        Retrieve top-k documents for each node.

        Parameters
        ----------
        node_embeddings : ndarray of shape (n_nodes, embedding_dim)
            Node embeddings to use as queries.
        adj_matrix : ndarray of shape (n_nodes, n_nodes), optional
            Adjacency matrix for neighborhood-aware retrieval.
        return_scores : bool, default=False
            Whether to return retrieval scores.

        Returns
        -------
        retrieved_indices : list of list of int
            Indices of retrieved documents for each node.
        scores : ndarray, optional
            Retrieval scores if return_scores=True.
        """
        if self.document_embeddings_ is None:
            raise ValueError("Retriever not fitted. Call fit() first.")

        n_nodes = node_embeddings.shape[0]
        n_docs = self.document_embeddings_.shape[0]

        # Compute base relevance scores
        # Project node embeddings to document space if dimensions differ
        if node_embeddings.shape[1] != self.document_embeddings_.shape[1]:
            # Use random projection for dimension matching
            if self.random_state is not None:
                np.random.seed(self.random_state)
            proj = np.random.randn(node_embeddings.shape[1], self.document_embeddings_.shape[1])
            proj /= np.sqrt(node_embeddings.shape[1])
            query_emb = node_embeddings @ proj
        else:
            query_emb = node_embeddings

        # Compute similarity scores
        scores = query_emb @ self.document_embeddings_.T

        # Apply neighborhood-aware scoring if adjacency provided
        if self.use_neighborhood and adj_matrix is not None:
            # Normalize adjacency
            degrees = adj_matrix.sum(axis=1) + 1
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            A_norm = D_inv_sqrt @ (adj_matrix + np.eye(n_nodes)) @ D_inv_sqrt
            # Smooth scores with 2-hop neighborhood
            scores = A_norm @ A_norm @ scores

        # Get top-k documents for each node
        k = min(self.top_k, n_docs)
        retrieved_indices = []
        for i in range(n_nodes):
            top_indices = np.argsort(scores[i])[-k:][::-1]
            retrieved_indices.append(top_indices.tolist())

        if return_scores:
            return retrieved_indices, scores
        return retrieved_indices

    def get_retrieved_features(
        self,
        node_embeddings: np.ndarray,
        adj_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get aggregated features from retrieved documents.

        Parameters
        ----------
        node_embeddings : ndarray of shape (n_nodes, embedding_dim)
            Node embeddings to use as queries.
        adj_matrix : ndarray of shape (n_nodes, n_nodes), optional
            Adjacency matrix for neighborhood-aware retrieval.

        Returns
        -------
        retrieved_features : ndarray of shape (n_nodes, doc_embedding_dim)
            Mean of retrieved document embeddings for each node.
        """
        retrieved_indices = self.retrieve(node_embeddings, adj_matrix)
        n_nodes = len(retrieved_indices)
        doc_dim = self.document_embeddings_.shape[1]

        retrieved_features = np.zeros((n_nodes, doc_dim))
        for i, indices in enumerate(retrieved_indices):
            if len(indices) > 0:
                retrieved_features[i] = self.document_embeddings_[indices].mean(axis=0)

        return retrieved_features

    def get_documents(self, indices: List[int]) -> List[str]:
        """
        Get document texts by indices.

        Parameters
        ----------
        indices : list of int
            Document indices.

        Returns
        -------
        documents : list of str
            Document texts.
        """
        if self._documents is None:
            raise ValueError("Documents not available (using precomputed embeddings).")
        return [self._documents[i] for i in indices]
