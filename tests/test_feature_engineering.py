# =============================================================================
# Tests for Feature Engineering Module
# =============================================================================
"""
Tests for the feature engineering module for tabular ML models.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.bid_features import (
    BidFeatureExtractor,
    aggregate_bids_per_item,
)
from src.feature_engineering.feature_config import (
    FeatureConfig,
    load_feature_config,
    get_config_value,
)
from src.feature_engineering.preprocessing import (
    handle_missing_values,
    normalize_features,
    select_features,
    get_feature_target_split,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_bid_data():
    """Sample bid data for testing."""
    return pd.DataFrame(
        {
            "auction_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "item_id": [101, 101, 101, 101, 101, 201, 201, 201],
            "bid_time": [
                "2024-01-01 10:00:00",
                "2024-01-01 10:05:00",
                "2024-01-01 10:10:00",
                "2024-01-01 10:15:00",
                "2024-01-01 10:20:00",
                "2024-01-02 14:00:00",
                "2024-01-02 14:30:00",
                "2024-01-02 15:00:00",
            ],
            "bid_amount": [10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 75.0, 100.0],
            "bid_is_proxy": [False, True, False, True, False, False, True, False],
            "bid_id": [1, 2, 3, 4, 5, 1, 2, 3],
            "bid_count": [5, 5, 5, 5, 5, 3, 3, 3],
        }
    )


@pytest.fixture
def single_bid_item_data():
    """Data with an item that has only one bid."""
    return pd.DataFrame(
        {
            "auction_id": [1],
            "item_id": [101],
            "bid_time": ["2024-01-01 10:00:00"],
            "bid_amount": [25.0],
            "bid_is_proxy": [False],
            "bid_id": [1],
            "bid_count": [1],
        }
    )


@pytest.fixture
def features_with_missing():
    """Feature DataFrame with missing values."""
    return pd.DataFrame(
        {
            "auction_id": [1, 2, 3, 4],
            "item_id": [101, 102, 103, 104],
            "winning_price": [100.0, 200.0, np.nan, 150.0],
            "total_bids": [5, 10, np.nan, 8],
            "bid_amount_mean": [50.0, np.nan, 75.0, 60.0],
        }
    )


# =============================================================================
# Configuration Tests
# =============================================================================


def test_load_feature_config():
    """Test loading feature configuration."""
    config = load_feature_config()

    assert config is not None
    assert isinstance(config, FeatureConfig)
    assert config.data_source.hf_dataset_repo == "jpearce610/bid_data"


def test_get_config_value():
    """Test getting configuration values by nested keys."""
    repo = get_config_value("data_source", "hf_dataset_repo")
    assert repo == "jpearce610/bid_data"

    # Test default value for non-existent key
    missing = get_config_value("nonexistent", "key", default="default_value")
    assert missing == "default_value"


def test_config_feature_groups():
    """Test feature group configuration."""
    config = load_feature_config()

    assert config.features.bid_amount.enabled is True
    assert config.features.bid_count.enabled is True
    assert config.features.time_features.enabled is True
    assert config.features.distribution_features.enabled is True
    assert config.features.proxy_features.enabled is True


def test_config_preprocessing():
    """Test preprocessing configuration."""
    config = load_feature_config()

    assert config.preprocessing.missing_values.numeric_strategy == "zero"
    assert config.preprocessing.missing_values.create_indicators is True


# =============================================================================
# Feature Extraction Tests
# =============================================================================


def test_bid_feature_extractor_init():
    """Test BidFeatureExtractor initialization."""
    extractor = BidFeatureExtractor()

    assert extractor.config is not None
    assert extractor.feature_config is not None


def test_extract_features_basic(sample_bid_data):
    """Test basic feature extraction."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    # Should have one row per unique item
    assert len(features_df) == 2

    # Should have auction_id and item_id columns
    assert "auction_id" in features_df.columns
    assert "item_id" in features_df.columns

    # Should have winning_price (target)
    assert "winning_price" in features_df.columns


def test_extract_bid_amount_features(sample_bid_data):
    """Test bid amount feature extraction."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    # Check item 101 (bids: 10, 15, 20, 25, 30)
    item_101 = features_df[features_df["item_id"] == 101].iloc[0]

    assert item_101["winning_price"] == 30.0
    assert item_101["bid_amount_max"] == 30.0
    assert item_101["bid_amount_min"] == 10.0
    assert item_101["bid_amount_mean"] == 20.0  # (10+15+20+25+30)/5
    assert item_101["bid_amount_median"] == 20.0
    assert item_101["bid_amount_range"] == 20.0  # 30 - 10


def test_extract_bid_count_features(sample_bid_data):
    """Test bid count feature extraction."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    item_101 = features_df[features_df["item_id"] == 101].iloc[0]
    item_201 = features_df[features_df["item_id"] == 201].iloc[0]

    assert item_101["total_bids"] == 5
    assert item_201["total_bids"] == 3


def test_extract_time_features(sample_bid_data):
    """Test time feature extraction."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    item_101 = features_df[features_df["item_id"] == 101].iloc[0]

    # Item 101: 20 minutes of bidding (10:00 to 10:20)
    assert item_101["bidding_duration_seconds"] == 1200.0  # 20 minutes

    # Mean time between bids: 5 minutes = 300 seconds
    assert item_101["mean_time_between_bids"] == 300.0


def test_extract_proxy_features(sample_bid_data):
    """Test proxy bid feature extraction."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    item_101 = features_df[features_df["item_id"] == 101].iloc[0]

    # Item 101 has 2 proxy bids out of 5
    assert item_101["proxy_bid_count"] == 2
    assert item_101["proxy_bid_ratio"] == 0.4


def test_extract_single_bid_item(single_bid_item_data):
    """Test feature extraction for item with single bid."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(single_bid_item_data)

    assert len(features_df) == 1

    item = features_df.iloc[0]
    assert item["winning_price"] == 25.0
    assert item["total_bids"] == 1
    assert item["bidding_duration_seconds"] == 0.0
    assert item["mean_time_between_bids"] == 0.0


def test_aggregate_bids_convenience_function(sample_bid_data):
    """Test the aggregate_bids_per_item convenience function."""
    features_df = aggregate_bids_per_item(sample_bid_data)

    assert len(features_df) == 2
    assert "winning_price" in features_df.columns


def test_get_feature_names(sample_bid_data):
    """Test getting feature names after extraction."""
    extractor = BidFeatureExtractor()
    extractor.extract_features(sample_bid_data)

    feature_names = extractor.get_feature_names()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "auction_id" not in feature_names
    assert "item_id" not in feature_names


def test_get_feature_metadata():
    """Test getting feature metadata."""
    extractor = BidFeatureExtractor()
    metadata = extractor.get_feature_metadata()

    assert "winning_price" in metadata
    assert metadata["winning_price"]["type"] == "target"
    assert "total_bids" in metadata


# =============================================================================
# Preprocessing Tests
# =============================================================================


def test_handle_missing_values_zero_strategy(features_with_missing):
    """Test missing value handling with zero strategy."""
    config = load_feature_config()
    config.preprocessing.missing_values.numeric_strategy = "zero"

    result = handle_missing_values(features_with_missing, config)

    # Check that missing values are filled with 0
    assert result["total_bids"].isnull().sum() == 0
    assert result["bid_amount_mean"].isnull().sum() == 0
    assert result.loc[2, "total_bids"] == 0.0


def test_handle_missing_values_with_indicators(features_with_missing):
    """Test missing value indicator creation."""
    config = load_feature_config()
    config.preprocessing.missing_values.create_indicators = True

    result = handle_missing_values(features_with_missing, config)

    # Check for indicator columns
    assert "total_bids_missing" in result.columns
    assert result.loc[2, "total_bids_missing"] == 1


def test_normalize_features_standard():
    """Test standard normalization."""
    df = pd.DataFrame(
        {
            "auction_id": [1, 2, 3, 4],
            "item_id": [101, 102, 103, 104],
            "winning_price": [100.0, 200.0, 150.0, 250.0],
            "feature1": [0.0, 10.0, 20.0, 30.0],
            "feature2": [100.0, 100.0, 100.0, 100.0],  # Zero variance
        }
    )

    config = load_feature_config()
    config.preprocessing.normalization.enabled = True
    config.preprocessing.normalization.method = "standard"

    result, params = normalize_features(df, config)

    # feature1 should be z-score normalized
    # Mean of [0, 10, 20, 30] = 15, std ≈ 12.91
    assert "mean" in params
    assert "std" in params

    # Check that normalization was applied
    assert result["feature1"].mean() < 1e-10  # Should be ~0


def test_normalize_features_disabled():
    """Test that normalization is skipped when disabled."""
    df = pd.DataFrame(
        {
            "auction_id": [1, 2],
            "item_id": [101, 102],
            "feature1": [10.0, 20.0],
        }
    )

    config = load_feature_config()
    config.preprocessing.normalization.enabled = False

    result, params = normalize_features(df, config)

    # Should return original values
    assert result["feature1"].iloc[0] == 10.0
    assert result["feature1"].iloc[1] == 20.0
    assert params == {}


def test_select_features_zero_variance():
    """Test removal of zero-variance features."""
    df = pd.DataFrame(
        {
            "auction_id": [1, 2, 3, 4],
            "item_id": [101, 102, 103, 104],
            "winning_price": [100.0, 200.0, 150.0, 250.0],
            "feature1": [10.0, 20.0, 30.0, 40.0],
            "feature2": [5.0, 5.0, 5.0, 5.0],  # Zero variance
        }
    )

    config = load_feature_config()
    config.preprocessing.feature_selection.remove_zero_variance = True

    result = select_features(df, config)

    assert "feature1" in result.columns
    assert "feature2" not in result.columns


def test_get_feature_target_split():
    """Test feature/target split function."""
    df = pd.DataFrame(
        {
            "auction_id": [1, 2, 3],
            "item_id": [101, 102, 103],
            "winning_price": [100.0, 200.0, 150.0],
            "feature1": [10.0, 20.0, 30.0],
            "feature2": [5.0, 15.0, 25.0],
        }
    )

    X, y = get_feature_target_split(df)

    # X should not contain identifiers or target
    assert "auction_id" not in X.columns
    assert "item_id" not in X.columns
    assert "winning_price" not in X.columns

    # X should contain features
    assert "feature1" in X.columns
    assert "feature2" in X.columns

    # y should be the target
    assert len(y) == 3
    assert y.iloc[0] == 100.0


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_pipeline(sample_bid_data):
    """Test the full feature engineering pipeline."""
    from src.feature_engineering.preprocessing import prepare_tabular_dataset

    features_df, metadata = prepare_tabular_dataset(sample_bid_data, save_to_disk=False)

    # Check output
    assert len(features_df) == 2
    assert "winning_price" in features_df.columns
    assert "total_bids" in features_df.columns

    # Check metadata
    assert "n_items" in metadata
    assert metadata["n_items"] == 2
    assert "feature_names" in metadata


def test_distribution_features_last_n_bids(sample_bid_data):
    """Test last N bids features."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    # Should have last_3_bids_mean_amount feature
    assert "last_3_bids_mean_amount" in features_df.columns

    # Check item 101: last 3 bids are 20, 25, 30 -> mean = 25
    item_101 = features_df[features_df["item_id"] == 101].iloc[0]
    assert item_101["last_3_bids_mean_amount"] == 25.0


def test_bid_concentration_features(sample_bid_data):
    """Test bid concentration features."""
    extractor = BidFeatureExtractor()
    features_df = extractor.extract_features(sample_bid_data)

    assert "bid_concentration_last_25pct" in features_df.columns
    assert "bid_concentration_last_10pct" in features_df.columns

    # All values should be between 0 and 1
    assert (features_df["bid_concentration_last_25pct"] >= 0).all()
    assert (features_df["bid_concentration_last_25pct"] <= 1).all()
