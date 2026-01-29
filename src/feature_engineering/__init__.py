# =============================================================================
# Auction Price Prediction - Feature Engineering Module
# =============================================================================
"""
Feature engineering module for tabular model training.

This module provides utilities for:
- Loading bid data from Hugging Face datasets
- Aggregating bids per item into summary statistics
- Extracting time-based and distribution features
- Preprocessing and normalization

All configuration is managed via a central YAML config file.
"""

from src.feature_engineering.bid_features import (
    BidFeatureExtractor,
    aggregate_bids_per_item,
)
from src.feature_engineering.feature_config import (
    FeatureConfig,
    load_feature_config,
)
from src.feature_engineering.hf_bid_loader import (
    HFBidDataLoader,
    load_bid_data,
)
from src.feature_engineering.preprocessing import (
    handle_missing_values,
    normalize_features,
    prepare_tabular_dataset,
)

__all__ = [
    # Config
    "FeatureConfig",
    "load_feature_config",
    # Data Loading
    "HFBidDataLoader",
    "load_bid_data",
    # Feature Extraction
    "BidFeatureExtractor",
    "aggregate_bids_per_item",
    # Preprocessing
    "handle_missing_values",
    "normalize_features",
    "prepare_tabular_dataset",
]
