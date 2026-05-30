# =============================================================================
# Tests for RAG - Similar Item Retrieval
# =============================================================================
"""
Unit tests for the SimilarItemRetriever RAG module.
"""

import numpy as np
import pandas as pd
import pytest

from src.rag import SimilarItem, SimilarItemRetriever


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_embeddings_df():
    """Create a small DataFrame of fake item embeddings."""
    rng = np.random.default_rng(42)
    items = [
        {
            "item_id": 1,
            "auction_id": 100,
            "embedding": rng.random(50).astype(np.float32).tolist(),
            "item_title": "Vintage Oak Table",
            "item_description": "Beautiful antique table from the 1920s.",
            "winning_price": 120.0,
        },
        {
            "item_id": 2,
            "auction_id": 100,
            "embedding": rng.random(50).astype(np.float32).tolist(),
            "item_title": "Antique Chair Set",
            "item_description": "Set of 4 Victorian chairs.",
            "winning_price": 75.0,
        },
        {
            "item_id": 3,
            "auction_id": 101,
            "embedding": rng.random(50).astype(np.float32).tolist(),
            "item_title": "Porcelain Vase",
            "item_description": "Blue and white porcelain vase.",
            "winning_price": 45.0,
        },
        {
            "item_id": 4,
            "auction_id": 101,
            "embedding": rng.random(50).astype(np.float32).tolist(),
            "item_title": "Brass Clock",
            "item_description": "Antique brass pendulum clock.",
            "winning_price": 200.0,
        },
        {
            "item_id": 5,
            "auction_id": 102,
            "embedding": rng.random(50).astype(np.float32).tolist(),
            "item_title": "Wooden Dresser",
            "item_description": "Solid wood dresser with 4 drawers.",
            "winning_price": 90.0,
        },
    ]
    return pd.DataFrame(items)


@pytest.fixture
def retriever(sample_embeddings_df):
    """Build and return a SimilarItemRetriever."""
    return SimilarItemRetriever.from_dataframe(sample_embeddings_df)


@pytest.fixture
def query_embedding(sample_embeddings_df):
    """Return the embedding of the first item as a query vector."""
    return np.array(sample_embeddings_df.iloc[0]["embedding"], dtype=np.float32)


# =============================================================================
# SimilarItem
# =============================================================================


class TestSimilarItem:
    """Tests for the SimilarItem data class."""

    def test_to_dict(self):
        item = SimilarItem(
            item_id=1,
            auction_id=100,
            similarity=0.95,
            item_title="Vintage Table",
            item_description="Oak table",
            winning_price=120.0,
        )
        d = item.to_dict()
        assert d["item_id"] == 1
        assert d["auction_id"] == 100
        assert d["similarity"] == 0.95
        assert d["item_title"] == "Vintage Table"
        assert d["winning_price"] == 120.0

    def test_similarity_rounded(self):
        item = SimilarItem(item_id=1, auction_id=1, similarity=0.123456789)
        d = item.to_dict()
        assert d["similarity"] == round(0.123456789, 4)

    def test_repr(self):
        item = SimilarItem(item_id=7, auction_id=3, similarity=0.8, winning_price=50.0)
        assert "7" in repr(item)
        assert "0.8000" in repr(item)


# =============================================================================
# SimilarItemRetriever - index construction
# =============================================================================


class TestBuildIndex:
    """Tests for SimilarItemRetriever.build_index."""

    def test_index_built(self, retriever, sample_embeddings_df):
        assert retriever.index_size == len(sample_embeddings_df)

    def test_embedding_dim(self, retriever):
        assert retriever.embedding_dim == 50

    def test_repr(self, retriever):
        r = repr(retriever)
        assert "5" in r
        assert "50" in r

    def test_raises_on_empty_df(self):
        r = SimilarItemRetriever()
        with pytest.raises(ValueError, match="must not be empty"):
            r.build_index(pd.DataFrame())

    def test_raises_on_missing_columns(self):
        r = SimilarItemRetriever()
        bad_df = pd.DataFrame({"item_id": [1], "embedding": [[0.1] * 10]})
        with pytest.raises(ValueError, match="missing columns"):
            r.build_index(bad_df)

    def test_unit_normalised(self, retriever):
        """All rows of index_matrix should be unit vectors."""
        norms = np.linalg.norm(retriever.index_matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_zero_embedding_handled(self):
        """Zero-vector embeddings should not cause NaN in the index."""
        df = pd.DataFrame(
            {
                "item_id": [1, 2],
                "auction_id": [10, 10],
                "embedding": [
                    [0.0] * 10,  # all-zero vector
                    [1.0] + [0.0] * 9,
                ],
            }
        )
        r = SimilarItemRetriever.from_dataframe(df)
        assert not np.any(np.isnan(r.index_matrix))


# =============================================================================
# SimilarItemRetriever - factory constructors
# =============================================================================


class TestFromDataframe:
    """Tests for SimilarItemRetriever.from_dataframe."""

    def test_creates_retriever(self, sample_embeddings_df):
        r = SimilarItemRetriever.from_dataframe(sample_embeddings_df)
        assert r.index_size == len(sample_embeddings_df)

    def test_metadata_preserved(self, sample_embeddings_df):
        r = SimilarItemRetriever.from_dataframe(sample_embeddings_df)
        assert "item_title" in r.metadata.columns
        assert "winning_price" in r.metadata.columns


class TestFromParquet:
    """Tests for SimilarItemRetriever.from_parquet."""

    def test_loads_from_file(self, sample_embeddings_df, tmp_path):
        parquet_path = tmp_path / "embeddings.parquet"
        sample_embeddings_df.to_parquet(parquet_path, index=False)

        r = SimilarItemRetriever.from_parquet(parquet_path)
        assert r.index_size == len(sample_embeddings_df)

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimilarItemRetriever.from_parquet(tmp_path / "nonexistent.parquet")

    def test_merges_optional_metadata(self, sample_embeddings_df, tmp_path):
        """Extra metadata parquet is merged on item_id."""
        # Main embeddings (without winning_price)
        emb_df = sample_embeddings_df[["item_id", "auction_id", "embedding"]].copy()
        emb_path = tmp_path / "embeddings.parquet"
        emb_df.to_parquet(emb_path, index=False)

        # Separate metadata
        meta_df = sample_embeddings_df[["item_id", "winning_price"]].copy()
        meta_path = tmp_path / "metadata.parquet"
        meta_df.to_parquet(meta_path, index=False)

        r = SimilarItemRetriever.from_parquet(emb_path, metadata_path=meta_path)
        assert "winning_price" in r.metadata.columns


# =============================================================================
# SimilarItemRetriever - retrieval
# =============================================================================


class TestRetrieve:
    """Tests for SimilarItemRetriever.retrieve."""

    def test_returns_k_results(self, retriever, query_embedding):
        results = retriever.retrieve(query_embedding, k=3)
        assert len(results) == 3

    def test_all_results_are_similar_items(self, retriever, query_embedding):
        results = retriever.retrieve(query_embedding, k=2)
        assert all(isinstance(r, SimilarItem) for r in results)

    def test_results_ordered_by_similarity(self, retriever, query_embedding):
        results = retriever.retrieve(query_embedding, k=5)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_similarity_range(self, retriever, query_embedding):
        results = retriever.retrieve(query_embedding, k=5)
        for r in results:
            assert -1.0 <= r.similarity <= 1.0 + 1e-5

    def test_exact_match_has_highest_similarity(self, retriever, sample_embeddings_df):
        """Querying with an item's own embedding should return it first."""
        first_emb = np.array(
            sample_embeddings_df.iloc[0]["embedding"], dtype=np.float32
        )
        results = retriever.retrieve(first_emb, k=1)
        assert results[0].item_id == int(sample_embeddings_df.iloc[0]["item_id"])

    def test_exclude_item_ids(self, retriever, query_embedding):
        results_all = retriever.retrieve(query_embedding, k=5)
        top_id = results_all[0].item_id

        results_excl = retriever.retrieve(
            query_embedding, k=5, exclude_item_ids=[top_id]
        )
        returned_ids = [r.item_id for r in results_excl]
        assert top_id not in returned_ids

    def test_k_capped_at_50(self, retriever, query_embedding):
        """k > 50 should be silently capped."""
        results = retriever.retrieve(query_embedding, k=100)
        # Only 5 items in fixture; all are returned
        assert len(results) <= 50

    def test_raises_if_no_index(self, query_embedding):
        r = SimilarItemRetriever()
        with pytest.raises(RuntimeError, match="Index has not been built"):
            r.retrieve(query_embedding)

    def test_zero_vector_query(self, retriever):
        """Zero-vector query should not raise."""
        zero = np.zeros(50, dtype=np.float32)
        results = retriever.retrieve(zero, k=3)
        assert isinstance(results, list)

    def test_metadata_in_results(self, retriever, query_embedding):
        """Results should carry metadata columns (title, price)."""
        results = retriever.retrieve(query_embedding, k=1)
        r = results[0]
        # Fixture includes titles and prices
        assert r.item_title is not None
        assert r.winning_price is not None

    def test_k_larger_than_index(self, retriever, query_embedding):
        """Requesting more items than in the index returns all of them."""
        results = retriever.retrieve(query_embedding, k=50)
        assert len(results) == retriever.index_size


# =============================================================================
# SimilarItemRetriever - properties
# =============================================================================


class TestProperties:
    """Tests for SimilarItemRetriever properties before/after index build."""

    def test_index_size_zero_before_build(self):
        r = SimilarItemRetriever()
        assert r.index_size == 0

    def test_embedding_dim_zero_before_build(self):
        r = SimilarItemRetriever()
        assert r.embedding_dim == 0
