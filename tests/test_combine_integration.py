# =============================================================================
# Integration Test for Combined Engineered Datasets Pipeline
# =============================================================================
"""
Integration test that downloads actual datasets from HuggingFace
and verifies the merge pipeline works end-to-end.

Note: This test requires internet access and may take some time.
"""

import pytest

from src.combine_engineered_datasets import (
    load_auction_features,
    load_bid_features_batched,
    load_item_features,
    merge_datasets,
)


@pytest.mark.integration
def test_load_auction_features():
    """Test loading actual auction features from HuggingFace."""
    df = load_auction_features()

    # Check that we got data
    assert len(df) > 0

    # Check that required columns exist
    assert "auction_id" in df.columns

    # Check that we have multiple columns
    assert len(df.columns) > 10


@pytest.mark.integration
def test_load_item_features():
    """Test loading actual item features from HuggingFace."""
    df = load_item_features()

    # Check that we got data
    assert len(df) > 0

    # Check that required columns exist
    assert "auction_id" in df.columns
    assert "item_id" in df.columns

    # Check that we have multiple columns
    assert len(df.columns) > 10


@pytest.mark.integration
def test_load_bid_features():
    """Test loading actual bid features from HuggingFace."""
    df = load_bid_features_batched()

    # Check that we got data
    assert len(df) > 0

    # Check that required columns exist
    assert "auction_id" in df.columns
    assert "item_id" in df.columns

    # Check that we have multiple columns
    assert len(df.columns) > 10


@pytest.mark.integration
def test_full_pipeline_with_actual_data():
    """Test the full merge pipeline with actual data from HuggingFace."""
    # Load datasets
    auction_df = load_auction_features()
    item_df = load_item_features()
    bid_df = load_bid_features_batched()

    # Check that all datasets have data
    assert len(auction_df) > 0
    assert len(item_df) > 0
    assert len(bid_df) > 0

    # Merge datasets
    merged_df = merge_datasets(auction_df, item_df, bid_df)

    # Check that merge produced results
    assert len(merged_df) > 0

    # Check that auction_id and item_id are first two columns
    assert merged_df.columns[0] == "auction_id"
    assert merged_df.columns[1] == "item_id"

    # Check that we have columns from all three datasets
    # Should have more columns than any individual dataset
    assert len(merged_df.columns) > len(auction_df.columns)
    assert len(merged_df.columns) > len(item_df.columns)
    assert len(merged_df.columns) > len(bid_df.columns)

    # Check that we have data in the merged dataset
    assert merged_df["auction_id"].notna().sum() > 0
    assert merged_df["item_id"].notna().sum() > 0

    print("\nMerge successful!")
    print(f"  Auction records: {len(auction_df):,}")
    print(f"  Item records: {len(item_df):,}")
    print(f"  Bid records: {len(bid_df):,}")
    print(f"  Merged records: {len(merged_df):,}")
    print(f"  Total columns: {len(merged_df.columns)}")
