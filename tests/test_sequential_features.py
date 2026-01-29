# =============================================================================
# Tests for Sequential Feature Engineering
# =============================================================================
"""
Tests for the sequential model feature engineering modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.sequential_features import (
    NormalizationStats,
    create_padding_mask,
    extract_bid_amount,
    extract_bid_increment,
    extract_bid_position,
    extract_cumulative_bid_count,
    extract_is_proxy,
    extract_label,
    extract_per_bid_features,
    extract_relative_bid_amount,
    extract_sequence_level_features,
    extract_starting_bid,
    extract_time_since_previous_bid,
    extract_total_bids,
    pad_sequence,
    process_item_bids,
    truncate_sequence,
)
from src.data.sequential_loader import (
    filter_items_by_sequence_length,
    get_enabled_per_bid_features,
    get_hf_dataset_repo,
    get_max_sequence_length,
    group_bids_by_item,
    load_sequential_config,
    split_item_ids,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_bids_df():
    """Sample bid data for a single item."""
    return pd.DataFrame({
        "auction_id": [100] * 5,
        "item_id": [1001] * 5,
        "bid_amount": [10.0, 15.0, 20.0, 25.0, 30.0],
        "bid_time": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 10:05:00",
            "2024-01-01 10:10:00",
            "2024-01-01 10:15:00",
            "2024-01-01 10:20:00",
        ]),
        "bid_is_proxy": [False, True, False, True, False],
    })


@pytest.fixture
def sample_multi_item_bids_df():
    """Sample bid data for multiple items."""
    return pd.DataFrame({
        "auction_id": [100, 100, 100, 100, 100, 200, 200, 200],
        "item_id": [1001, 1001, 1001, 1002, 1002, 2001, 2001, 2001],
        "bid_amount": [10.0, 15.0, 20.0, 5.0, 10.0, 50.0, 60.0, 70.0],
        "bid_time": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 10:05:00",
            "2024-01-01 10:10:00",
            "2024-01-01 11:00:00",
            "2024-01-01 11:05:00",
            "2024-01-02 10:00:00",
            "2024-01-02 10:10:00",
            "2024-01-02 10:20:00",
        ]),
        "bid_is_proxy": [False, True, False, False, True, False, False, False],
    })


# =============================================================================
# Configuration Tests
# =============================================================================


def test_load_sequential_config():
    """Test that config loads successfully."""
    config = load_sequential_config()
    assert isinstance(config, dict)
    assert "sequence" in config
    assert "features" in config
    assert "data_source" in config


def test_get_hf_dataset_repo():
    """Test HF dataset repo configuration."""
    repo = get_hf_dataset_repo()
    assert repo == "jpearce610/bid_data"


def test_get_max_sequence_length():
    """Test max sequence length configuration."""
    max_len = get_max_sequence_length()
    assert isinstance(max_len, int)
    assert max_len > 0


def test_get_enabled_per_bid_features():
    """Test per-bid features configuration."""
    features = get_enabled_per_bid_features()
    assert isinstance(features, list)
    # Should have at least bid_amount enabled
    assert "bid_amount" in features


# =============================================================================
# Feature Extraction Tests
# =============================================================================


def test_extract_bid_amount(sample_bids_df):
    """Test bid amount extraction."""
    amounts = extract_bid_amount(sample_bids_df)
    assert isinstance(amounts, np.ndarray)
    assert len(amounts) == 5
    np.testing.assert_array_almost_equal(
        amounts, [10.0, 15.0, 20.0, 25.0, 30.0]
    )


def test_extract_time_since_previous_bid(sample_bids_df):
    """Test time since previous bid extraction."""
    time_diffs = extract_time_since_previous_bid(sample_bids_df)
    assert isinstance(time_diffs, np.ndarray)
    assert len(time_diffs) == 5
    assert time_diffs[0] == 0.0  # First bid has 0 time diff
    assert time_diffs[1] == 300.0  # 5 minutes = 300 seconds


def test_extract_time_since_previous_bid_with_clip(sample_bids_df):
    """Test time since previous bid with clipping."""
    time_diffs = extract_time_since_previous_bid(sample_bids_df, clip_max=100)
    assert np.all(time_diffs <= 100)


def test_extract_relative_bid_amount(sample_bids_df):
    """Test relative bid amount extraction."""
    relative = extract_relative_bid_amount(sample_bids_df)
    assert isinstance(relative, np.ndarray)
    assert len(relative) == 5
    assert relative[0] == 1.0  # First bid is baseline
    assert relative[1] == 1.5  # 15/10
    assert relative[4] == 3.0  # 30/10


def test_extract_bid_increment(sample_bids_df):
    """Test bid increment extraction."""
    increments = extract_bid_increment(sample_bids_df)
    assert isinstance(increments, np.ndarray)
    assert len(increments) == 5
    assert increments[0] == 0.0  # First bid has no increment
    np.testing.assert_array_almost_equal(
        increments[1:], [5.0, 5.0, 5.0, 5.0]
    )


def test_extract_bid_position(sample_bids_df):
    """Test bid position extraction."""
    positions = extract_bid_position(sample_bids_df)
    assert isinstance(positions, np.ndarray)
    assert len(positions) == 5
    assert positions[0] == 0.0
    assert positions[-1] == 1.0
    # Should be evenly spaced
    np.testing.assert_array_almost_equal(
        positions, [0.0, 0.25, 0.5, 0.75, 1.0]
    )


def test_extract_cumulative_bid_count(sample_bids_df):
    """Test cumulative bid count extraction."""
    counts = extract_cumulative_bid_count(sample_bids_df)
    assert isinstance(counts, np.ndarray)
    assert len(counts) == 5
    np.testing.assert_array_almost_equal(counts, [1, 2, 3, 4, 5])


def test_extract_is_proxy(sample_bids_df):
    """Test proxy bid flag extraction."""
    proxy = extract_is_proxy(sample_bids_df)
    assert isinstance(proxy, np.ndarray)
    assert len(proxy) == 5
    np.testing.assert_array_almost_equal(
        proxy, [0.0, 1.0, 0.0, 1.0, 0.0]
    )


def test_extract_total_bids(sample_bids_df):
    """Test total bids extraction."""
    total = extract_total_bids(sample_bids_df)
    assert total == 5.0


def test_extract_starting_bid(sample_bids_df):
    """Test starting bid extraction."""
    starting = extract_starting_bid(sample_bids_df)
    assert starting == 10.0


def test_extract_label(sample_bids_df):
    """Test label (final bid) extraction."""
    label = extract_label(sample_bids_df)
    assert label == 30.0


def test_extract_per_bid_features(sample_bids_df):
    """Test per-bid feature extraction pipeline."""
    features = extract_per_bid_features(sample_bids_df)
    assert isinstance(features, dict)
    assert "bid_amount" in features
    assert len(features["bid_amount"]) == 5


def test_extract_sequence_level_features(sample_bids_df):
    """Test sequence-level feature extraction."""
    features = extract_sequence_level_features(sample_bids_df)
    assert isinstance(features, dict)
    assert "total_bids" in features
    assert features["total_bids"] == 5.0


# =============================================================================
# Padding and Truncation Tests
# =============================================================================


def test_pad_sequence_post():
    """Test post-padding of sequences."""
    features = {
        "bid_amount": np.array([1.0, 2.0, 3.0]),
        "increment": np.array([0.0, 1.0, 1.0]),
    }
    padded = pad_sequence(features, max_length=5, padding_strategy="post")

    assert len(padded["bid_amount"]) == 5
    np.testing.assert_array_almost_equal(
        padded["bid_amount"], [1.0, 2.0, 3.0, 0.0, 0.0]
    )


def test_pad_sequence_pre():
    """Test pre-padding of sequences."""
    features = {
        "bid_amount": np.array([1.0, 2.0, 3.0]),
    }
    padded = pad_sequence(features, max_length=5, padding_strategy="pre")

    assert len(padded["bid_amount"]) == 5
    np.testing.assert_array_almost_equal(
        padded["bid_amount"], [0.0, 0.0, 1.0, 2.0, 3.0]
    )


def test_truncate_sequence_pre():
    """Test pre-truncation (keep last N)."""
    features = {
        "bid_amount": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    truncated = truncate_sequence(features, max_length=3, truncation_strategy="pre")

    assert len(truncated["bid_amount"]) == 3
    np.testing.assert_array_almost_equal(
        truncated["bid_amount"], [3.0, 4.0, 5.0]
    )


def test_truncate_sequence_post():
    """Test post-truncation (keep first N)."""
    features = {
        "bid_amount": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    truncated = truncate_sequence(features, max_length=3, truncation_strategy="post")

    assert len(truncated["bid_amount"]) == 3
    np.testing.assert_array_almost_equal(
        truncated["bid_amount"], [1.0, 2.0, 3.0]
    )


def test_create_padding_mask_post():
    """Test padding mask creation for post-padding."""
    mask = create_padding_mask(3, max_length=5, padding_strategy="post")
    np.testing.assert_array_almost_equal(
        mask, [1.0, 1.0, 1.0, 0.0, 0.0]
    )


def test_create_padding_mask_pre():
    """Test padding mask creation for pre-padding."""
    mask = create_padding_mask(3, max_length=5, padding_strategy="pre")
    np.testing.assert_array_almost_equal(
        mask, [0.0, 0.0, 1.0, 1.0, 1.0]
    )


# =============================================================================
# Normalization Tests
# =============================================================================


def test_normalization_stats_minmax():
    """Test min-max normalization statistics."""
    stats = NormalizationStats()
    values = np.array([0.0, 50.0, 100.0])
    stats.fit("test_feature", values, method="minmax")

    transformed = stats.transform("test_feature", values)
    np.testing.assert_array_almost_equal(transformed, [0.0, 0.5, 1.0])


def test_normalization_stats_standard():
    """Test standard (z-score) normalization."""
    stats = NormalizationStats()
    values = np.array([0.0, 10.0, 20.0])
    stats.fit("test_feature", values, method="standard")

    transformed = stats.transform("test_feature", values)
    # Mean should be 0
    assert abs(np.mean(transformed)) < 1e-6


def test_normalization_stats_log():
    """Test log normalization."""
    stats = NormalizationStats()
    values = np.array([1.0, 10.0, 100.0])
    stats.fit("test_feature", values, method="log")

    transformed = stats.transform("test_feature", values)
    assert transformed[0] < transformed[1] < transformed[2]


def test_normalization_stats_save_load(tmp_path):
    """Test saving and loading normalization stats."""
    stats = NormalizationStats()
    stats.fit("feature1", np.array([0.0, 100.0]), method="minmax")
    stats.fit("feature2", np.array([0.0, 10.0, 20.0]), method="standard")

    # Save
    save_path = tmp_path / "stats.json"
    stats.save(save_path)

    # Load
    loaded_stats = NormalizationStats()
    loaded_stats.load(save_path)

    assert "feature1" in loaded_stats.stats
    assert "feature2" in loaded_stats.stats


# =============================================================================
# Data Loading Tests
# =============================================================================


def test_group_bids_by_item(sample_multi_item_bids_df):
    """Test grouping bids by item."""
    grouped = group_bids_by_item(sample_multi_item_bids_df)

    assert isinstance(grouped, dict)
    assert (100, 1001) in grouped
    assert (100, 1002) in grouped
    assert (200, 2001) in grouped
    assert len(grouped[(100, 1001)]) == 3
    assert len(grouped[(100, 1002)]) == 2


def test_filter_items_by_sequence_length(sample_multi_item_bids_df):
    """Test filtering items by sequence length."""
    grouped = group_bids_by_item(sample_multi_item_bids_df)

    # Filter to min 3 bids
    filtered = filter_items_by_sequence_length(grouped, min_length=3)
    assert (100, 1001) in filtered  # Has 3 bids
    assert (200, 2001) in filtered  # Has 3 bids
    assert (100, 1002) not in filtered  # Has only 2 bids


def test_split_item_ids():
    """Test splitting item IDs into train/val/test."""
    item_ids = [(i, i * 10) for i in range(100)]

    train_ids, val_ids, test_ids = split_item_ids(
        item_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_ids) == 70
    assert len(val_ids) == 15
    assert len(test_ids) == 15

    # No overlap
    all_ids = set(train_ids + val_ids + test_ids)
    assert len(all_ids) == 100


# =============================================================================
# Full Pipeline Tests
# =============================================================================


def test_process_item_bids(sample_bids_df):
    """Test full bid processing pipeline."""
    sequence_features, padding_mask, label, static_features = process_item_bids(
        sample_bids_df, apply_normalization=False
    )

    assert isinstance(sequence_features, np.ndarray)
    assert isinstance(padding_mask, np.ndarray)
    assert isinstance(label, float)
    assert isinstance(static_features, dict)

    # Check dimensions
    max_len = get_max_sequence_length()
    assert sequence_features.shape[0] == max_len
    assert padding_mask.shape[0] == max_len
    assert label == 30.0


def test_process_item_bids_with_normalization(sample_bids_df):
    """Test bid processing with normalization."""
    # Create and fit normalization stats
    stats = NormalizationStats()
    stats.fit("bid_amount", np.array([0.0, 100.0]), method="minmax")

    sequence_features, _, label, _ = process_item_bids(
        sample_bids_df, norm_stats=stats, apply_normalization=True
    )

    # Features should be normalized
    assert sequence_features.shape[0] == get_max_sequence_length()
