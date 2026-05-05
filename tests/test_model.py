"""Tests for RAG-GNN model."""

import numpy as np
import pytest
from rag_gnn import RAGGNN, GNNEncoder, KnowledgeRetriever, FusionModule
from rag_gnn.utils import normalize_adjacency, compute_silhouette


class TestGNNEncoder:
    """Tests for GNN encoder."""

    def test_init(self):
        encoder = GNNEncoder(n_layers=3, hidden_dim=64)
        assert encoder.n_layers == 3
        assert encoder.hidden_dim == 64

    def test_fit_transform(self):
        np.random.seed(42)
        adj = np.random.rand(50, 50) > 0.8
        adj = (adj | adj.T).astype(float)

        encoder = GNNEncoder(n_layers=2, hidden_dim=32, random_state=42)
        embeddings = encoder.fit_transform(adj)

        assert embeddings.shape == (50, 32)
        assert encoder.embeddings_ is not None
        assert len(encoder.layer_embeddings_) == 3  # input + 2 layers

    def test_reproducibility(self):
        np.random.seed(42)
        adj = np.random.rand(30, 30) > 0.7
        adj = (adj | adj.T).astype(float)

        encoder1 = GNNEncoder(n_layers=2, hidden_dim=16, random_state=123)
        emb1 = encoder1.fit_transform(adj)

        encoder2 = GNNEncoder(n_layers=2, hidden_dim=16, random_state=123)
        emb2 = encoder2.fit_transform(adj)

        np.testing.assert_array_almost_equal(emb1, emb2)


class TestKnowledgeRetriever:
    """Tests for knowledge retriever."""

    def test_fit_tfidf(self):
        documents = [
            "protein A regulates cell cycle",
            "protein B induces apoptosis",
            "protein C repairs DNA damage",
        ]
        retriever = KnowledgeRetriever(top_k=2)
        retriever.fit(documents)

        assert retriever.document_embeddings_ is not None
        assert retriever.document_embeddings_.shape[0] == 3

    def test_retrieve(self):
        documents = [f"document {i} about topic" for i in range(10)]
        retriever = KnowledgeRetriever(top_k=3, random_state=42)
        retriever.fit(documents)

        query_emb = np.random.randn(5, 64)
        retrieved = retriever.retrieve(query_emb)

        assert len(retrieved) == 5
        assert all(len(r) == 3 for r in retrieved)

    def test_get_retrieved_features(self):
        documents = [f"document {i}" for i in range(10)]
        retriever = KnowledgeRetriever(top_k=3, random_state=42)
        retriever.fit(documents)

        query_emb = np.random.randn(5, 64)
        features = retriever.get_retrieved_features(query_emb)

        assert features.shape[0] == 5
        assert features.shape[1] == retriever.document_embeddings_.shape[1]


class TestFusionModule:
    """Tests for fusion module."""

    def test_weighted_concat(self):
        gnn_emb = np.random.randn(20, 64)
        ret_emb = np.random.randn(20, 32)

        fusion = FusionModule(method='weighted_concat', alpha=0.6, output_dim=64)
        fused = fusion.fuse(gnn_emb, ret_emb)

        assert fused.shape == (20, 64)

    def test_add_fusion(self):
        gnn_emb = np.random.randn(20, 64)
        ret_emb = np.random.randn(20, 64)

        fusion = FusionModule(method='add', alpha=0.5)
        fused = fusion.fuse(gnn_emb, ret_emb)

        assert fused.shape[0] == 20


class TestRAGGNN:
    """Tests for main RAG-GNN model."""

    def test_init(self):
        model = RAGGNN(n_layers=3, hidden_dim=128)
        assert model.n_layers == 3
        assert model.hidden_dim == 128

    def test_fit_without_documents(self):
        np.random.seed(42)
        adj = np.random.rand(30, 30) > 0.7
        adj = (adj | adj.T).astype(float)

        model = RAGGNN(n_layers=2, hidden_dim=32, random_state=42)
        embeddings = model.fit_transform(adj)

        assert embeddings.shape == (30, 32)
        assert model.retrieved_features_ is None

    def test_fit_with_documents(self):
        np.random.seed(42)
        adj = np.random.rand(30, 30) > 0.7
        adj = (adj | adj.T).astype(float)
        documents = [f"document {i} about proteins" for i in range(20)]

        model = RAGGNN(n_layers=2, hidden_dim=32, retrieval_k=3, random_state=42)
        embeddings = model.fit_transform(adj, documents=documents)

        assert embeddings.shape == (30, 32)
        assert model.retrieved_features_ is not None
        assert model.gnn_embeddings_ is not None

    def test_get_most_similar(self):
        np.random.seed(42)
        adj = np.random.rand(30, 30) > 0.7
        adj = (adj | adj.T).astype(float)

        model = RAGGNN(n_layers=2, hidden_dim=32, random_state=42)
        model.fit(adj)

        similar = model.get_most_similar(0, top_k=5)

        assert len(similar) == 5
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
        assert 0 not in [s[0] for s in similar]  # Exclude self

    def test_compute_similarity(self):
        np.random.seed(42)
        adj = np.random.rand(30, 30) > 0.7
        adj = (adj | adj.T).astype(float)

        model = RAGGNN(n_layers=2, hidden_dim=32, random_state=42)
        model.fit(adj)

        sim = model.compute_similarity(0, 1)

        assert -1 <= sim <= 1


class TestUtils:
    """Tests for utility functions."""

    def test_normalize_adjacency(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        A_norm = normalize_adjacency(adj)

        assert A_norm.shape == (3, 3)
        # Check symmetry
        np.testing.assert_array_almost_equal(A_norm, A_norm.T)

    def test_compute_silhouette(self):
        np.random.seed(42)
        # Create clearly separated clusters
        emb1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        emb2 = np.random.randn(20, 10) + np.array([-5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([emb1, emb2])
        labels = np.array([0] * 20 + [1] * 20)

        score = compute_silhouette(embeddings, labels)

        assert -1 <= score <= 1
        assert score > 0  # Well-separated clusters should have positive score


class TestLearnableRAGGNN:
    """Tests for learnable (PyTorch) RAG-GNN model."""

    @pytest.fixture(autouse=True)
    def check_torch(self):
        pytest.importorskip("torch")

    def test_init(self):
        from rag_gnn import LearnableRAGGNN
        model = LearnableRAGGNN(hidden_dim=64, doc_dim=32, n_layers=3, k=5)
        assert model.hidden_dim == 64
        assert model.doc_dim == 32
        assert model.k == 5

    def test_forward_shape(self):
        import torch
        from rag_gnn import LearnableRAGGNN

        n_nodes, hidden_dim, doc_dim, n_docs, k = 20, 64, 32, 50, 5
        model = LearnableRAGGNN(hidden_dim=hidden_dim, doc_dim=doc_dim, n_layers=3, k=k)

        A = torch.randn(n_nodes, n_nodes)
        X = torch.randn(n_nodes, hidden_dim)
        doc_emb = torch.randn(n_docs, doc_dim)

        fused = model(A, X, doc_emb)
        assert fused.shape == (n_nodes, hidden_dim)

    def test_forward_return_all(self):
        import torch
        from rag_gnn import LearnableRAGGNN

        n_nodes, hidden_dim, doc_dim, n_docs, k = 20, 64, 32, 50, 5
        model = LearnableRAGGNN(hidden_dim=hidden_dim, doc_dim=doc_dim, n_layers=3, k=k)

        A = torch.randn(n_nodes, n_nodes)
        X = torch.randn(n_nodes, hidden_dim)
        doc_emb = torch.randn(n_docs, doc_dim)

        fused, gnn, ctx, scores, topk_idx, gate = model(A, X, doc_emb, return_all=True)
        assert fused.shape == (n_nodes, hidden_dim)
        assert gnn.shape == (n_nodes, hidden_dim)
        assert ctx.shape == (n_nodes, doc_dim)
        assert scores.shape == (n_nodes, n_docs)
        assert topk_idx.shape == (n_nodes, k)
        assert gate.shape == (n_nodes, hidden_dim)

    def test_gate_range(self):
        import torch
        from rag_gnn import LearnableRAGGNN

        model = LearnableRAGGNN(hidden_dim=32, doc_dim=16, k=3)
        A = torch.randn(10, 10)
        X = torch.randn(10, 32)
        doc_emb = torch.randn(20, 16)

        _, _, _, _, _, gate = model(A, X, doc_emb, return_all=True)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_gradient_flow(self):
        import torch
        from rag_gnn import LearnableRAGGNN

        model = LearnableRAGGNN(hidden_dim=32, doc_dim=16, k=3)
        A = torch.randn(10, 10)
        X = torch.randn(10, 32)
        doc_emb = torch.randn(20, 16)

        fused = model(A, X, doc_emb)
        loss = fused.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
