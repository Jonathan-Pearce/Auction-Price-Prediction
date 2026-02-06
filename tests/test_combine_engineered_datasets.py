# =============================================================================
# Tests for Combined Engineered Datasets Pipeline
# =============================================================================
"""
Tests for the pipeline that combines engineered auction, item, and bid datasets.
"""

import pandas as pd
import pytest

from src.combine_engineered_datasets import merge_datasets

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_auction_df():
    """Create sample auction feature data."""
    return pd.DataFrame(
        {
            "auction_id": [1, 2, 3],
            "auction_length_hours": [120, 144, 96],
            "auction_item_count": [50, 100, 25],
            "auction_total_winning_price": [2500.0, 10000.0, 1000.0],
        }
    )


@pytest.fixture
def sample_item_df():
    """Create sample item feature data."""
    return pd.DataFrame(
        {
            "auction_id": [1, 1, 2, 2, 3],
            "item_id": [101, 102, 201, 202, 301],
            "item_title_length": [50, 75, 60, 45, 80],
            "item_num_images": [5, 3, 8, 4, 6],
            "item_winning_price": [100.0, 150.0, 200.0, 250.0, 50.0],
        }
    )


@pytest.fixture
def sample_bid_df():
    """Create sample bid feature data."""
    return pd.DataFrame(
        {
            "auction_id": [1, 1, 1, 2, 2],
            "item_id": [101, 101, 102, 201, 201],
            "bid_number": [1, 2, 1, 1, 2],
            "bid_amount": [50.0, 100.0, 150.0, 150.0, 200.0],
            "bid_time_since_first": [0.0, 60.0, 0.0, 0.0, 120.0],
        }
    )


# =============================================================================
# Test Merging
# =============================================================================


def test_merge_datasets_basic(sample_auction_df, sample_item_df, sample_bid_df):
    """Test basic merging of all three datasets."""
    result = merge_datasets(sample_auction_df, sample_item_df, sample_bid_df)

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that we have rows (outer join should keep all records)
    assert len(result) > 0

    # Check that auction_id and item_id are the first two columns
    assert result.columns[0] == "auction_id"
    assert result.columns[1] == "item_id"


def test_merge_preserves_all_columns(sample_auction_df, sample_item_df, sample_bid_df):
    """Test that merging preserves all columns from all datasets."""
    result = merge_datasets(sample_auction_df, sample_item_df, sample_bid_df)

    # Check that we have columns from all three datasets
    auction_cols = set(sample_auction_df.columns)
    item_cols = set(sample_item_df.columns)
    bid_cols = set(sample_bid_df.columns)

    result_cols = set(result.columns)

    # Auction columns (except auction_id which is a merge key)
    for col in auction_cols - {"auction_id"}:
        assert (
            col in result_cols or f"{col}_auction" in result_cols
        ), f"Missing auction column: {col}"

    # Item columns (except merge keys)
    for col in item_cols - {"auction_id", "item_id"}:
        assert (
            col in result_cols or f"{col}_item" in result_cols
        ), f"Missing item column: {col}"

    # Bid columns (except merge keys)
    for col in bid_cols - {"auction_id", "item_id"}:
        assert (
            col in result_cols or f"{col}_bid" in result_cols
        ), f"Missing bid column: {col}"


def test_merge_with_missing_auction_id_in_items():
    """Test merging when item_df doesn't have auction_id column."""
    auction_df = pd.DataFrame(
        {
            "auction_id": [1, 2],
            "auction_feature": ["A", "B"],
        }
    )

    item_df = pd.DataFrame(
        {
            "item_id": [101, 102],
            "item_feature": ["X", "Y"],
        }
    )

    bid_df = pd.DataFrame(
        {
            "auction_id": [1, 1],
            "item_id": [101, 102],
            "bid_amount": [50.0, 75.0],
        }
    )

    # This should not raise an error
    result = merge_datasets(auction_df, item_df, bid_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_merge_column_order(sample_auction_df, sample_item_df, sample_bid_df):
    """Test that auction_id and item_id are always the first two columns."""
    result = merge_datasets(sample_auction_df, sample_item_df, sample_bid_df)

    # First column should be auction_id
    assert result.columns[0] == "auction_id"

    # Second column should be item_id
    assert result.columns[1] == "item_id"


def test_merge_outer_join_behavior(sample_auction_df, sample_item_df, sample_bid_df):
    """Test that outer join keeps all records from all datasets."""
    result = merge_datasets(sample_auction_df, sample_item_df, sample_bid_df)

    # With outer join, we should have at least as many rows as the largest dataset
    max_rows = max(len(sample_auction_df), len(sample_item_df), len(sample_bid_df))

    # The actual result may be larger due to multiple matches
    assert len(result) >= max_rows


def test_merge_handles_duplicates():
    """Test merging with duplicate keys (multiple bids per item)."""
    auction_df = pd.DataFrame(
        {
            "auction_id": [1],
            "auction_feature": ["A"],
        }
    )

    item_df = pd.DataFrame(
        {
            "auction_id": [1],
            "item_id": [101],
            "item_feature": ["X"],
        }
    )

    # Multiple bids for same item
    bid_df = pd.DataFrame(
        {
            "auction_id": [1, 1, 1],
            "item_id": [101, 101, 101],
            "bid_number": [1, 2, 3],
            "bid_amount": [50.0, 75.0, 100.0],
        }
    )

    result = merge_datasets(auction_df, item_df, bid_df)

    # Should have 3 rows (one per bid)
    assert len(result) == 3

    # All rows should have the same auction and item features
    assert (result["auction_feature"] == "A").all()
    assert (result["item_feature"] == "X").all()

    # But different bid numbers
    assert result["bid_number"].tolist() == [1, 2, 3]


def test_merge_with_empty_dataframe():
    """Test merging with one empty dataset."""
    auction_df = pd.DataFrame(
        {
            "auction_id": [1, 2],
            "auction_feature": ["A", "B"],
        }
    )

    item_df = pd.DataFrame(
        {
            "auction_id": [1],
            "item_id": [101],
            "item_feature": ["X"],
        }
    )

    # Empty bid dataframe
    bid_df = pd.DataFrame(
        {
            "auction_id": pd.Series([], dtype="int64"),
            "item_id": pd.Series([], dtype="int64"),
            "bid_amount": pd.Series([], dtype="float64"),
        }
    )

    result = merge_datasets(auction_df, item_df, bid_df)

    # Should still produce a result
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0  # Should have at least auction and item data


def test_merge_no_common_keys_raises_error():
    """Test that merging without common keys raises an error."""
    auction_df = pd.DataFrame(
        {
            "auction_key": [1, 2],
            "auction_feature": ["A", "B"],
        }
    )

    item_df = pd.DataFrame(
        {
            "item_key": [101, 102],
            "item_feature": ["X", "Y"],
        }
    )

    bid_df = pd.DataFrame(
        {
            "bid_key": [1001, 1002],
            "bid_amount": [50.0, 75.0],
        }
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Cannot merge datasets: no common keys"):
        merge_datasets(auction_df, item_df, bid_df)
