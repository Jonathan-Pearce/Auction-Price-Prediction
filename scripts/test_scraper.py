#!/usr/bin/env python
"""
Test script for auction scraper functionality.

This script validates:
1. Configuration loading
2. Field transformations (renaming and prefixing)
3. Data aggregation logic
4. DataFrame creation

Does NOT test actual API calls (would require network access).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data import scraper_config as config
from src.data.auction_scraper import AuctionDataFetcher, transform_auction_data


def test_config():
    """Test configuration values."""
    print("Testing configuration...")

    # Check API configuration
    assert config.MAXSOLD_API_BASE_URL == "https://maxsold.maxsold.com/msapi"
    assert config.DEFAULT_ITEMS_LIMIT == 2500
    assert config.DEFAULT_RATE_LIMIT == 10

    # Check field mappings
    assert config.FIELD_RENAME_MAP["catalog_lots"] == "item_count"
    assert config.FIELD_RENAME_MAP["current_bid"] == "winning_price"
    assert config.COLUMN_PREFIX == "auction_"

    # Check required fields
    assert "id" in config.AUCTION_FIELDS
    assert "title" in config.AUCTION_FIELDS
    assert "catalog_lots" in config.AUCTION_FIELDS

    print("✓ Configuration tests passed")


def test_field_transformations():
    """Test field name transformations."""
    print("\nTesting field transformations...")

    # Test renaming
    assert config.get_renamed_field_name("catalog_lots") == "item_count"
    assert config.get_renamed_field_name("current_bid") == "winning_price"
    assert config.get_renamed_field_name("other_field") == "other_field"

    # Test prefixing
    assert config.get_prefixed_field_name("title") == "auction_title"
    assert config.get_prefixed_field_name("item_count") == "auction_item_count"

    # Test full transformation
    assert config.apply_field_transformations("catalog_lots") == "auction_item_count"
    assert config.apply_field_transformations("title") == "auction_title"

    print("✓ Field transformation tests passed")


def test_data_processing():
    """Test auction data processing logic."""
    print("\nTesting data processing...")

    # Create mock auction data
    mock_api_response = {
        "auction": {
            "id": 99941,
            "title": "Test Auction",
            "starts": "2024-01-01T00:00:00",
            "ends": "2024-01-02T00:00:00",
            "catalog_lots": 50,
            "extended_bidding": True,
        },
        "items": [
            {
                "item_id": 1,
                "viewed": 100,
                "current_bid": 25.50,
                "bid_count": 5,
                "images": ["img1.jpg", "img2.jpg"],
            },
            {
                "item_id": 2,
                "viewed": 50,
                "current_bid": 10.00,
                "bid_count": 2,
                "images": ["img3.jpg"],
            },
            {
                "item_id": 3,
                "viewed": 75,
                "current_bid": 0,
                "bid_count": 0,
                "images": [],
            },
        ],
    }

    # Process the data
    fetcher = AuctionDataFetcher()
    processed = fetcher.process_auction_data(99941, mock_api_response)

    # Verify aggregations
    assert processed["id"] == 99941
    assert processed["title"] == "Test Auction"
    assert processed["catalog_lots"] == 50
    assert processed["total_viewed"] == 225  # 100 + 50 + 75
    assert processed["total_winning_price"] == 35.50  # 25.50 + 10.00 + 0
    assert processed["total_bid_count"] == 7  # 5 + 2 + 0
    assert processed["total_images"] == 3  # 2 + 1 + 0

    print("✓ Data processing tests passed")


def test_dataframe_transformation():
    """Test DataFrame transformation (renaming and prefixing)."""
    print("\nTesting DataFrame transformation...")

    # Create mock auction data
    auctions = [
        {
            "id": 99941,
            "title": "Auction 1",
            "catalog_lots": 50,
            "total_viewed": 225,
            "total_winning_price": 35.50,
            "total_bid_count": 7,
            "total_images": 3,
        },
        {
            "id": 99942,
            "title": "Auction 2",
            "catalog_lots": 30,
            "total_viewed": 150,
            "total_winning_price": 50.00,
            "total_bid_count": 10,
            "total_images": 5,
        },
    ]

    # Transform
    df = transform_auction_data(auctions)

    # Verify column names
    expected_columns = [
        "auction_id",
        "auction_title",
        "auction_item_count",  # catalog_lots renamed and prefixed
        "auction_total_viewed",
        "auction_total_winning_price",  # current_bid -> winning_price
        "auction_total_bid_count",
        "auction_total_images",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in: {df.columns}"

    # Verify data integrity
    assert len(df) == 2
    assert df["auction_id"].tolist() == [99941, 99942]
    assert df["auction_item_count"].tolist() == [50, 30]
    assert df["auction_total_winning_price"].tolist() == [35.50, 50.00]

    print("✓ DataFrame transformation tests passed")


def test_auction_id_loading():
    """Test loading auction IDs from parquet file."""
    print("\nTesting auction ID loading...")

    # Check if file exists
    if not config.AUCTION_IDS_FILE.exists():
        print(f"⚠ Auction ID file not found: {config.AUCTION_IDS_FILE}")
        return

    # Load file
    df = pd.read_parquet(config.AUCTION_IDS_FILE)

    # Verify column exists
    assert (
        config.AUCTION_IDS_COLUMN in df.columns
    ), f"Column '{config.AUCTION_IDS_COLUMN}' not found"

    # Verify data
    auction_ids = df[config.AUCTION_IDS_COLUMN].tolist()
    assert len(auction_ids) > 0, "No auction IDs found"
    assert all(isinstance(aid, int) for aid in auction_ids), "Auction IDs must be integers"

    print(f"✓ Loaded {len(auction_ids)} auction IDs from file")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AUCTION SCRAPER VALIDATION TESTS")
    print("=" * 60)

    try:
        test_config()
        test_field_transformations()
        test_data_processing()
        test_dataframe_transformation()
        test_auction_id_loading()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
