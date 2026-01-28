# =============================================================================
# Auction Price Prediction - Feature Engineering
# =============================================================================
"""
Feature engineering pipeline for auction price prediction.

Transforms raw auction, item, and bid data into features for ML models:
- Tabular features: Numerical and categorical from auction/item metadata
- Text features: Processed item descriptions and titles
- Image features: Extracted from item photos (via pretrained models)
- Sequential features: Bid history time series
"""

import re
from typing import Any

import pandas as pd
from loguru import logger

from src.config import settings


# =============================================================================
# Tabular Features
# =============================================================================


def engineer_auction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from auction-level data.

    Features to consider:
    - Auction type (estate sale, moving sale, reseller, etc.)
    - Location (city, state/province, region)
    - Day of week / time of day auction ends
    - Season / month
    - Total number of items in auction
    - Auction duration

    Args:
        df: Raw auction DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering auction features...")
    features = df.copy()

    # TODO: Implement auction-level feature engineering
    # Example features:
    # - features['auction_item_count'] = ...
    # - features['auction_day_of_week'] = ...
    # - features['auction_is_weekend'] = ...

    return features


def engineer_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from item-level data.

    Features to consider:
    - Category (hierarchical)
    - Starting bid / reserve price
    - Number of images
    - Description length
    - Keywords in title/description
    - Item position in auction (lot number)
    - Estimated value (if available)

    Args:
        df: Raw item DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering item features...")
    features = df.copy()

    # TODO: Implement item-level feature engineering
    # Example features:
    # - features['description_length'] = features['description'].str.len()
    # - features['num_images'] = ...
    # - features['has_reserve'] = ...

    return features


def engineer_bid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from bid history.

    Features to consider:
    - Number of bids
    - Number of unique bidders
    - Bid velocity (bids per hour)
    - Max bid increment
    - Average bid increment
    - Time since last bid
    - Soft-close extensions count
    - Early vs late bidding ratio

    Args:
        df: Raw bid DataFrame (for a single item or aggregated)

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering bid features...")
    features = df.copy()

    # TODO: Implement bid-level feature engineering
    # Example features:
    # - features['num_bids'] = ...
    # - features['num_unique_bidders'] = ...
    # - features['bid_velocity'] = ...

    return features


# =============================================================================
# Text Features
# =============================================================================


def preprocess_text(text: str) -> str:
    """
    Preprocess text for NLP models.

    Steps:
    - Lowercase
    - Remove special characters
    - Normalize whitespace
    - (Optional) Remove stopwords
    - (Optional) Lemmatization

    Args:
        text: Raw text string

    Returns:
        Preprocessed text
    """
    if not text or pd.isna(text):
        return ""

    # Basic preprocessing
    text = str(text).lower().strip()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # TODO: Add more sophisticated preprocessing
    # - Remove HTML tags
    # - Normalize unicode
    # - Handle auction-specific abbreviations

    return text


def extract_handcrafted_text_features(text: str) -> dict[str, Any]:
    """
    Extract hand-crafted features from item description text.
    
    These features are fast to compute and interpretable,
    making them ideal for tabular models and quick inference.
    
    Args:
        text: Item description text (raw, unprocessed)
        
    Returns:
        Dictionary of numeric features
    """
    if not text or pd.isna(text):
        text = ""
    
    features = {}
    
    # Basic length features
    features['text_char_count'] = len(text)
    words = text.split()
    features['text_word_count'] = len(words)
    
    # Sentence count (filter empty strings)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    features['text_sentence_count'] = len(sentences)
    
    # Average word length (excluding spaces)
    features['text_avg_word_length'] = (
        sum(len(word) for word in words) / max(len(words), 1)
    )
    
    # Lexical features
    features['text_uppercase_ratio'] = (
        sum(c.isupper() for c in text) / max(len(text), 1)
    )
    features['text_digit_count'] = sum(c.isdigit() for c in text)
    features['text_punctuation_count'] = sum(c in '.,!?;:' for c in text)
    
    # Auction-specific keyword presence (use word boundaries)
    text_lower = text.lower()
    
    # Brand/authenticity keywords
    brand_keywords = ['authentic', 'original', 'genuine', 'branded', 'signed']
    features['text_brand_keyword_count'] = sum(
        bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
        for word in brand_keywords
    )
    features['text_has_brand_keywords'] = int(features['text_brand_keyword_count'] > 0)
    
    # Condition keywords
    condition_keywords = ['mint', 'excellent', 'good', 'fair', 'poor', 'damaged', 'worn', 'new']
    features['text_condition_keyword_count'] = sum(
        bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
        for word in condition_keywords
    )
    features['text_has_condition_keywords'] = int(features['text_condition_keyword_count'] > 0)
    
    # Quality/collectibility keywords
    quality_keywords = ['rare', 'vintage', 'antique', 'collectible', 'limited', 'unique', 'classic']
    features['text_quality_keyword_count'] = sum(
        bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
        for word in quality_keywords
    )
    features['text_has_quality_keywords'] = int(features['text_quality_keyword_count'] > 0)
    
    # Material keywords
    material_keywords = ['wood', 'metal', 'glass', 'ceramic', 'plastic', 'leather', 'silver', 'gold']
    features['text_material_keyword_count'] = sum(
        bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
        for word in material_keywords
    )
    features['text_has_material_keywords'] = int(features['text_material_keyword_count'] > 0)
    
    return features


def extract_text_features(
    texts: list[str],
    method: str = "tfidf",
    vectorizer: Any = None,
    max_features: int = 300,
) -> tuple[Any, Any]:
    """
    Extract features from text data.

    Methods:
    - 'tfidf': TF-IDF vectorization (sparse, fast)
    - 'embeddings': Pretrained sentence embeddings (dense, semantic)
    - 'bow': Bag of words (sparse, fast)
    - 'handcrafted': Hand-crafted features (fast, interpretable)

    Args:
        texts: List of text strings
        method: Feature extraction method
        vectorizer: Pre-fitted vectorizer (for tfidf/bow), will fit if None
        max_features: Maximum features for tfidf/bow

    Returns:
        Tuple of (features, vectorizer/model) where:
        - features: Feature matrix or embeddings
        - vectorizer/model: Fitted vectorizer or model for future use
    """
    logger.info(f"Extracting text features using {method}...")
    
    # For handcrafted method, don't preprocess (need raw text for features like uppercase)
    if method != "handcrafted":
        # Preprocess texts for ML methods
        texts = [preprocess_text(text) for text in texts]

    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if vectorizer is None:
            # Adjust min_df based on dataset size
            min_doc_freq = max(1, min(5, len(texts) // 10))
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_doc_freq,  # Adjust based on corpus size
                max_df=0.9,  # More lenient for small datasets
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words='english' if len(texts) > 10 else None,  # Skip stopwords for tiny datasets
                lowercase=True,
                strip_accents='unicode',
            )
            features = vectorizer.fit_transform(texts)
        else:
            features = vectorizer.transform(texts)
        
        return features, vectorizer
    
    elif method == "bow":
        from sklearn.feature_extraction.text import CountVectorizer
        
        if vectorizer is None:
            # Adjust min_df based on dataset size
            min_doc_freq = max(1, min(5, len(texts) // 10))
            
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_doc_freq,  # Adjust based on corpus size
                max_df=0.9,  # More lenient for small datasets
                ngram_range=(1, 2),
                stop_words='english' if len(texts) > 10 else None,  # Skip stopwords for tiny datasets
                lowercase=True,
                binary=False,  # Count occurrences
            )
            features = vectorizer.fit_transform(texts)
        else:
            features = vectorizer.transform(texts)
        
        return features, vectorizer
    
    elif method == "embeddings":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        
        import numpy as np
        
        # Use pre-trained model if not provided
        if vectorizer is None:
            # Use fast, high-quality model (384 dimensions)
            model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            model = vectorizer
        
        # Generate embeddings
        features = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        return features, model
    
    elif method == "handcrafted":
        # Extract hand-crafted features from raw text
        features_list = [extract_handcrafted_text_features(text) for text in texts]
        features = pd.DataFrame(features_list)
        
        return features, None
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: 'tfidf', 'bow', 'embeddings', 'handcrafted'"
        )


# =============================================================================
# Image Features
# =============================================================================


def extract_image_features(
    image_paths: list[str],
    model_name: str = "resnet50",
) -> Any:
    """
    Extract features from item images using pretrained CNN.

    Args:
        image_paths: List of paths to images
        model_name: Pretrained model to use for feature extraction

    Returns:
        Feature matrix (N x D) where D is embedding dimension
    """
    logger.info(f"Extracting image features using {model_name}...")

    # TODO: Implement image feature extraction
    # - Load pretrained model (ResNet, EfficientNet, etc.)
    # - Remove classification head
    # - Extract embeddings from penultimate layer

    raise NotImplementedError("Image feature extraction not yet implemented")


# =============================================================================
# Sequential Features
# =============================================================================


def prepare_bid_sequences(
    bids_df: pd.DataFrame,
    max_seq_length: int = 100,
    padding: str = "post",
) -> Any:
    """
    Prepare bid sequences for sequential models (LSTM, Transformer).

    Features per timestep:
    - Bid amount
    - Time since auction start
    - Time since last bid
    - Bid increment
    - Bidder ID (encoded)

    Args:
        bids_df: DataFrame with bid history
        max_seq_length: Maximum sequence length (pad/truncate)
        padding: 'pre' or 'post' padding

    Returns:
        Padded sequences array (N x T x F)
    """
    logger.info("Preparing bid sequences...")

    # TODO: Implement bid sequence preparation
    # - Group bids by item
    # - Create feature vectors per bid
    # - Pad/truncate to max_seq_length

    raise NotImplementedError("Bid sequence preparation not yet implemented")


# =============================================================================
# Feature Pipeline
# =============================================================================


def build_feature_pipeline(
    auctions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    bids_df: pd.DataFrame,
    include_text: bool = True,
    include_images: bool = True,
    include_sequences: bool = True,
) -> dict[str, Any]:
    """
    Build complete feature set for all models.

    Args:
        auctions_df: Auction-level data
        items_df: Item-level data
        bids_df: Bid history data
        include_text: Whether to extract text features
        include_images: Whether to extract image features
        include_sequences: Whether to prepare bid sequences

    Returns:
        Dictionary with feature sets for each model type
    """
    logger.info("Building feature pipeline...")

    features = {}

    # Tabular features (always included)
    auction_features = engineer_auction_features(auctions_df)
    item_features = engineer_item_features(items_df)
    bid_features = engineer_bid_features(bids_df)

    features["tabular"] = {
        "auctions": auction_features,
        "items": item_features,
        "bids": bid_features,
    }

    # Text features
    if include_text:
        # TODO: Extract text features from item descriptions
        features["text"] = None

    # Image features
    if include_images:
        # TODO: Extract image features from item photos
        features["images"] = None

    # Sequential features
    if include_sequences:
        # TODO: Prepare bid sequences
        features["sequences"] = None

    logger.info("Feature pipeline complete")
    return features


# =============================================================================
# Target Variable
# =============================================================================


def prepare_target(
    items_df: pd.DataFrame,
    target_col: str = "winning_price",
    handle_zero_bids: str = "include",
) -> pd.Series:
    """
    Prepare target variable (winning price).

    Args:
        items_df: Item DataFrame with winning prices
        target_col: Name of target column
        handle_zero_bids: How to handle items with no bids
            - 'include': Keep as 0
            - 'exclude': Remove from dataset
            - 'separate': Flag for separate model

    Returns:
        Target series
    """
    target = items_df[target_col].copy()

    if handle_zero_bids == "exclude":
        target = target[target > 0]
        logger.info(f"Excluded {(items_df[target_col] == 0).sum()} zero-bid items")
    elif handle_zero_bids == "include":
        logger.info(f"Including {(target == 0).sum()} zero-bid items")

    return target
