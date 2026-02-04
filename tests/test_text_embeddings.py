# =============================================================================
# Tests for Text Embeddings Feature Engineering Pipeline
# =============================================================================
"""
Unit tests for the text embeddings pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.text_embeddings import (
    combine_title_description,
    generate_item_embeddings,
    get_document_embedding,
    get_stopwords,
    prepare_training_corpus,
    preprocess_text,
    train_fasttext_model,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def stopwords_set():
    """Get stopwords set for testing."""
    return get_stopwords()


@pytest.fixture
def sample_corpus():
    """Create sample corpus for testing."""
    return [
        ["vintage", "oak", "table", "beautiful", "antique"],
        ["wooden", "chair", "hand", "carved", "victorian"],
        ["collectible", "porcelain", "vase", "blue", "white"],
        ["furniture", "set", "dining", "table", "chairs"],
        ["antique", "clock", "brass", "pendulum", "working"],
    ]


@pytest.fixture
def sample_item_df():
    """Create sample item DataFrame for testing."""
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "auction_id": [100, 100, 101, 101],
            "item_title": [
                "Vintage Oak Table",
                "Antique Chair Set",
                None,
                "Beautiful Vase",
            ],
            "item_description": [
                "Beautiful wooden dining table from 1920s.",
                "Set of 4 Victorian chairs in excellent condition.",
                "Blue and white porcelain vase.",
                None,
            ],
        }
    )


@pytest.fixture
def trained_model(sample_corpus):
    """Train a small model for testing."""
    return train_fasttext_model(
        sample_corpus,
        vector_size=50,  # Small for fast testing
        window=3,
        min_count=1,
        epochs=5,
    )


# =============================================================================
# Test Stopwords Loading
# =============================================================================


class TestStopwords:
    """Test stopwords loading."""

    def test_stopwords_loaded(self, stopwords_set):
        """Test that stopwords are loaded."""
        assert len(stopwords_set) > 0
        assert "the" in stopwords_set
        assert "and" in stopwords_set
        assert "is" in stopwords_set

    def test_stopwords_is_set(self, stopwords_set):
        """Test that stopwords is a set for O(1) lookup."""
        assert isinstance(stopwords_set, set)


# =============================================================================
# Test Text Preprocessing
# =============================================================================


class TestPreprocessText:
    """Test text preprocessing function."""

    def test_basic_preprocessing(self, stopwords_set):
        """Test basic text preprocessing."""
        text = "Beautiful Antique Oak Table"
        tokens = preprocess_text(text, stopwords_set)

        assert "beautiful" in tokens
        assert "antique" in tokens
        assert "oak" in tokens
        assert "table" in tokens

    def test_lowercase(self, stopwords_set):
        """Test that text is lowercased."""
        text = "VINTAGE TABLE"
        tokens = preprocess_text(text, stopwords_set)

        assert all(t.islower() for t in tokens)

    def test_stopwords_removed(self, stopwords_set):
        """Test that stopwords are removed."""
        text = "This is a beautiful table"
        tokens = preprocess_text(text, stopwords_set)

        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "beautiful" in tokens
        assert "table" in tokens

    def test_punctuation_removed(self, stopwords_set):
        """Test that punctuation is removed."""
        text = "Beautiful, antique table! For $500."
        tokens = preprocess_text(text, stopwords_set)

        assert all("," not in t and "!" not in t and "$" not in t for t in tokens)

    def test_numbers_removed(self, stopwords_set):
        """Test that pure numbers are removed by default."""
        text = "Table costs 500 dollars"
        tokens = preprocess_text(text, stopwords_set, remove_numbers=True)

        assert "500" not in tokens
        assert "table" in tokens
        assert "dollars" in tokens

    def test_numbers_kept_when_disabled(self, stopwords_set):
        """Test that numbers are kept when remove_numbers=False."""
        text = "Item from 1950"
        tokens = preprocess_text(text, stopwords_set, remove_numbers=False)

        assert "1950" in tokens

    def test_alphanumeric_kept(self, stopwords_set):
        """Test that alphanumeric tokens like '1920s' are kept."""
        text = "From the 1920s era"
        tokens = preprocess_text(text, stopwords_set)

        assert "1920s" in tokens

    def test_none_input(self, stopwords_set):
        """Test handling of None input."""
        tokens = preprocess_text(None, stopwords_set)
        assert tokens == []

    def test_empty_input(self, stopwords_set):
        """Test handling of empty string."""
        tokens = preprocess_text("", stopwords_set)
        assert tokens == []

    def test_whitespace_only(self, stopwords_set):
        """Test handling of whitespace-only string."""
        tokens = preprocess_text("   ", stopwords_set)
        assert tokens == []

    def test_html_tags_removed(self, stopwords_set):
        """Test that HTML tags are removed."""
        text = "<p>Beautiful <b>antique</b> table</p>"
        tokens = preprocess_text(text, stopwords_set)

        assert "p" not in tokens
        assert "b" not in tokens
        assert "beautiful" in tokens
        assert "antique" in tokens

    def test_short_tokens_removed(self, stopwords_set):
        """Test that single-character tokens are removed."""
        text = "A beautiful B table C"
        tokens = preprocess_text(text, stopwords_set)

        assert all(len(t) > 1 for t in tokens)


# =============================================================================
# Test Combined Text Processing
# =============================================================================


class TestCombineTitleDescription:
    """Test combined title and description processing."""

    def test_combines_title_and_description(self, stopwords_set):
        """Test that title and description are combined."""
        title = "Vintage Table"
        description = "Beautiful wooden piece"
        tokens = combine_title_description(title, description, stopwords_set)

        assert "vintage" in tokens
        assert "table" in tokens
        assert "beautiful" in tokens
        assert "wooden" in tokens
        assert "piece" in tokens

    def test_title_first(self, stopwords_set):
        """Test that title tokens come before description tokens."""
        title = "Vintage"
        description = "Beautiful"
        tokens = combine_title_description(title, description, stopwords_set)

        # Vintage should come before Beautiful
        assert tokens.index("vintage") < tokens.index("beautiful")

    def test_handles_none_title(self, stopwords_set):
        """Test handling of None title."""
        tokens = combine_title_description(None, "Beautiful table", stopwords_set)

        assert "beautiful" in tokens
        assert "table" in tokens

    def test_handles_none_description(self, stopwords_set):
        """Test handling of None description."""
        tokens = combine_title_description("Vintage table", None, stopwords_set)

        assert "vintage" in tokens
        assert "table" in tokens

    def test_handles_both_none(self, stopwords_set):
        """Test handling of both None."""
        tokens = combine_title_description(None, None, stopwords_set)
        assert tokens == []


# =============================================================================
# Test Corpus Preparation
# =============================================================================


class TestPrepareTrainingCorpus:
    """Test corpus preparation."""

    def test_corpus_structure(self, sample_item_df):
        """Test that corpus has correct structure."""
        corpus = prepare_training_corpus(sample_item_df)

        assert isinstance(corpus, list)
        assert all(isinstance(doc, list) for doc in corpus)
        assert all(isinstance(token, str) for doc in corpus for token in doc)

    def test_handles_missing_text(self, sample_item_df):
        """Test that items with missing text are handled."""
        corpus = prepare_training_corpus(sample_item_df)

        # Should have entries for items with at least some text
        # Item 3 has no title but has description, Item 4 has title but no description
        assert len(corpus) >= 2


# =============================================================================
# Test Model Training
# =============================================================================


class TestTrainFastTextModel:
    """Test FastText model training."""

    def test_model_trains(self, sample_corpus):
        """Test that model trains successfully."""
        model = train_fasttext_model(
            sample_corpus, vector_size=50, window=3, min_count=1, epochs=5
        )

        assert model is not None
        assert hasattr(model, "wv")

    def test_model_has_vocabulary(self, trained_model):
        """Test that trained model has vocabulary."""
        assert len(trained_model.wv) > 0

    def test_model_generates_vectors(self, trained_model):
        """Test that model generates vectors for words."""
        vector = trained_model.wv["table"]

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (50,)  # Our test model has vector_size=50

    def test_model_handles_oov(self, trained_model):
        """Test that FastText handles out-of-vocabulary words."""
        # FastText should generate vectors for OOV words using subwords
        vector = trained_model.wv["unknownword123"]

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (50,)


# =============================================================================
# Test Document Embedding
# =============================================================================


class TestGetDocumentEmbedding:
    """Test document embedding generation."""

    def test_embedding_shape(self, trained_model):
        """Test that embedding has correct shape."""
        tokens = ["vintage", "oak", "table"]
        embedding = get_document_embedding(tokens, trained_model)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (50,)  # Our test model has vector_size=50

    def test_embedding_dtype(self, trained_model):
        """Test that embedding has correct dtype."""
        tokens = ["vintage", "oak", "table"]
        embedding = get_document_embedding(tokens, trained_model)

        assert embedding.dtype == np.float32

    def test_empty_tokens_returns_zeros(self, trained_model):
        """Test that empty tokens return zero vector."""
        embedding = get_document_embedding([], trained_model)

        assert isinstance(embedding, np.ndarray)
        assert np.allclose(embedding, 0)

    def test_consistent_embeddings(self, trained_model):
        """Test that same tokens produce same embedding."""
        tokens = ["vintage", "oak", "table"]
        embedding1 = get_document_embedding(tokens, trained_model)
        embedding2 = get_document_embedding(tokens, trained_model)

        assert np.allclose(embedding1, embedding2)


# =============================================================================
# Test Item Embeddings Generation
# =============================================================================


class TestGenerateItemEmbeddings:
    """Test item embeddings generation."""

    def test_output_structure(self, sample_item_df, trained_model):
        """Test that output has correct structure."""
        embeddings_df = generate_item_embeddings(sample_item_df, trained_model)

        assert "auction_id" in embeddings_df.columns
        assert "item_id" in embeddings_df.columns
        assert "embedding" in embeddings_df.columns

    def test_output_count(self, sample_item_df, trained_model):
        """Test that output has correct number of rows."""
        embeddings_df = generate_item_embeddings(sample_item_df, trained_model)

        assert len(embeddings_df) == len(sample_item_df)

    def test_embedding_format(self, sample_item_df, trained_model):
        """Test that embeddings are stored as lists."""
        embeddings_df = generate_item_embeddings(sample_item_df, trained_model)

        # Embeddings should be lists (for Parquet compatibility)
        assert isinstance(embeddings_df.iloc[0]["embedding"], list)
        assert len(embeddings_df.iloc[0]["embedding"]) == 50  # Our model's vector size

    def test_ids_preserved(self, sample_item_df, trained_model):
        """Test that item and auction IDs are preserved."""
        embeddings_df = generate_item_embeddings(sample_item_df, trained_model)

        assert set(embeddings_df["item_id"]) == set(sample_item_df["item_id"])
        assert set(embeddings_df["auction_id"]) == set(sample_item_df["auction_id"])
