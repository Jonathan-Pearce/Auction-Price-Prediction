"""
Tests for text feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import (
    preprocess_text,
    extract_handcrafted_text_features,
    extract_text_features,
)


class TestPreprocessText:
    """Test text preprocessing."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        text = "  Hello WORLD!  Multiple   spaces  "
        result = preprocess_text(text)
        assert result == "hello world! multiple spaces"
    
    def test_empty_text(self):
        """Test handling of empty text."""
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text(pd.NA) == ""


class TestHandcraftedFeatures:
    """Test hand-crafted text feature extraction."""
    
    def test_basic_features(self):
        """Test extraction of basic text features."""
        text = "Vintage leather bag in excellent condition."
        features = extract_handcrafted_text_features(text)
        
        # Check feature presence
        assert 'text_char_count' in features
        assert 'text_word_count' in features
        assert 'text_sentence_count' in features
        assert 'text_avg_word_length' in features
        
        # Check values make sense
        assert features['text_char_count'] > 0
        assert features['text_word_count'] == 7
        assert features['text_sentence_count'] >= 1
    
    def test_keyword_detection(self):
        """Test keyword detection in text."""
        text = "Rare vintage antique collectible item in mint condition"
        features = extract_handcrafted_text_features(text)
        
        # Should detect quality keywords
        assert features['text_has_quality_keywords'] == 1
        assert features['text_quality_keyword_count'] >= 2  # vintage, antique, rare, collectible
        
        # Should detect condition keywords
        assert features['text_has_condition_keywords'] == 1
        assert features['text_condition_keyword_count'] >= 1  # mint
    
    def test_empty_text_features(self):
        """Test features for empty text."""
        features = extract_handcrafted_text_features("")
        
        assert features['text_char_count'] == 0
        assert features['text_word_count'] == 0
        assert features['text_has_quality_keywords'] == 0


class TestTextFeatureExtraction:
    """Test various text feature extraction methods."""
    
    def test_handcrafted_method(self):
        """Test handcrafted method."""
        texts = [
            "Vintage leather bag",
            "Modern glass vase",
            "Antique wooden chair in excellent condition",
        ]
        
        features, vectorizer = extract_text_features(texts, method="handcrafted")
        
        # Should return DataFrame
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(texts)
        assert vectorizer is None  # No vectorizer for handcrafted
        
        # Check some columns exist
        assert 'text_word_count' in features.columns
        assert 'text_has_quality_keywords' in features.columns
    
    def test_tfidf_method(self):
        """Test TF-IDF extraction."""
        texts = [
            "vintage leather bag excellent condition",
            "modern glass vase beautiful design",
            "antique wooden chair rare collectible",
        ]
        
        features, vectorizer = extract_text_features(
            texts, 
            method="tfidf",
            max_features=50
        )
        
        # Check output shape
        assert features.shape[0] == len(texts)
        assert features.shape[1] <= 50
        assert vectorizer is not None
        
        # Should be sparse
        import scipy.sparse
        assert scipy.sparse.issparse(features)
    
    def test_bow_method(self):
        """Test Bag of Words extraction."""
        texts = [
            "vintage leather bag",
            "modern glass vase",
            "antique wooden chair",
        ]
        
        features, vectorizer = extract_text_features(
            texts,
            method="bow",
            max_features=50
        )
        
        # Check output
        assert features.shape[0] == len(texts)
        assert vectorizer is not None
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        texts = ["test text"]
        
        with pytest.raises(ValueError, match="Unknown method"):
            extract_text_features(texts, method="invalid_method")
    
    def test_tfidf_with_fitted_vectorizer(self):
        """Test using pre-fitted vectorizer."""
        texts_train = ["vintage bag", "modern vase", "antique chair"]
        texts_test = ["vintage vase", "modern bag"]
        
        # Fit on train
        features_train, vectorizer = extract_text_features(
            texts_train,
            method="tfidf",
            max_features=50
        )
        
        # Transform test with same vectorizer
        features_test, _ = extract_text_features(
            texts_test,
            method="tfidf",
            vectorizer=vectorizer
        )
        
        # Should have same number of features
        assert features_train.shape[1] == features_test.shape[1]


@pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers"),
    reason="sentence-transformers not installed"
)
class TestEmbeddings:
    """Test sentence embeddings (requires sentence-transformers)."""
    
    def test_embeddings_method(self):
        """Test sentence embeddings extraction."""
        texts = [
            "Vintage leather bag in excellent condition",
            "Modern glass vase with beautiful design",
        ]
        
        features, model = extract_text_features(texts, method="embeddings")
        
        # Check output shape
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(texts)
        assert features.shape[1] == 384  # all-MiniLM-L6-v2 dimension
        assert model is not None
    
    def test_embeddings_consistency(self):
        """Test that same text produces same embedding."""
        text = ["Test text for consistency"]
        
        features1, model = extract_text_features(text, method="embeddings")
        features2, _ = extract_text_features(text, method="embeddings", vectorizer=model)
        
        # Should be very similar (allowing for numerical precision)
        np.testing.assert_allclose(features1, features2, rtol=1e-5)
