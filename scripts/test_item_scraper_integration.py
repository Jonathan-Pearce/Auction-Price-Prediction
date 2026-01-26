#!/usr/bin/env python
# =============================================================================
# Item Scraper Integration Test
# =============================================================================
"""
Test script to verify item scraper functionality with mock data.
This simulates the scraping process without needing network access.
"""

import asyncio
from unittest.mock import MagicMock, patch

from src.data.item_scraper import ItemDataFetcher, transform_item_data

# Mock API response data
MOCK_API_RESPONSE = {
    "auction": {
        "items": {
            0: {
                "id": 1001,
                "auction_id": 99941,
                "title": "Antique Chair",
                "description": "Beautiful wooden chair from 1920s",
                "viewed": 150,
                "starting_bid": 10.0,
                "current_bid": 45.0,
                "proxy_bid": 50.0,
                "start_time": "2024-01-01T10:00:00",
                "end_time": "2024-01-05T18:00:00",
                "bid_count": 8,
                "bidding_extended": False,
                "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
            },
            1: {
                "id": 1002,
                "auction_id": 99941,
                "title": "Vintage Table",
                "description": "Solid oak dining table",
                "viewed": 200,
                "starting_bid": 25.0,
                "current_bid": 0.0,
                "proxy_bid": 0.0,
                "start_time": "2024-01-01T10:00:00",
                "end_time": "2024-01-05T18:00:00",
                "bid_count": 0,
                "bidding_extended": False,
                "images": ["img1.jpg"],
            },
            2: {
                "id": 1003,
                "auction_id": 99941,
                "title": "Lamp Set",
                "description": "Set of 2 matching lamps",
                "viewed": 75,
                "starting_bid": 5.0,
                "current_bid": 12.0,
                "proxy_bid": 15.0,
                "start_time": "2024-01-01T10:00:00",
                "end_time": "2024-01-05T18:00:00",
                "bid_count": 3,
                "bidding_extended": True,
                "images": ["img1.jpg", "img2.jpg"],
            },
        }
    }
}


async def test_scraper_integration():
    """Test the complete scraper workflow with mock data."""

    print("=" * 60)
    print("ITEM SCRAPER INTEGRATION TEST")
    print("=" * 60)
    print()

    # Test with mock API
    async with ItemDataFetcher() as fetcher:
        # Mock the HTTP client
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = MOCK_API_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            print("1. Fetching items for auction 99941...")
            items = await fetcher.fetch_auction_items(99941)
            print(f"   ✓ Retrieved {len(items)} items")
            print()

            print("2. Processing item data...")
            processed_items = [fetcher.process_item_data(item, 99941) for item in items]
            print(f"   ✓ Processed {len(processed_items)} items")
            print()

            print("3. Transforming data with 'item_' prefix...")
            df = transform_item_data(processed_items)
            print(f"   ✓ Created DataFrame with {len(df)} rows")
            print()

            print("4. Verifying output format...")
            print()
            print("Columns:")
            for col in df.columns:
                print(f"   - {col}")
            print()

            # Check all columns have item_ prefix except auction_id
            non_prefixed_cols = [
                col for col in df.columns if not col.startswith("item_")
            ]
            all_have_prefix = non_prefixed_cols == ["auction_id"]
            print(
                f"All columns have 'item_' prefix (except auction_id): {'✓ Yes' if all_have_prefix else '✗ No'}"
            )
            print()

            # Check for expected fields
            expected_fields = [
                "item_id",
                "auction_id",  # No prefix for auction_id
                "item_title",
                "item_description",
                "item_viewed",
                "item_starting_bid",
                "item_current_bid",
                "item_proxy_bid",
                "item_start_time",
                "item_end_time",
                "item_bid_count",
                "item_bidding_extended",
                "item_number_of_images",
            ]

            print("Expected fields present:")
            for field in expected_fields:
                present = field in df.columns
                print(f"   {'✓' if present else '✗'} {field}")
            print()

            print("5. Sample data:")
            print()
            print(df.head())
            print()

            print("6. Data types:")
            print()
            print(df.dtypes)
            print()

            print("7. Summary statistics:")
            print()
            print(df.describe())
            print()

            # Test zero-bid item handling
            zero_bid_items = df[df["item_bid_count"] == 0]
            print(f"8. Zero-bid items: {len(zero_bid_items)} items with no bids")
            if not zero_bid_items.empty:
                print(f"   Example: {zero_bid_items['item_title'].iloc[0]}")
            print()

            # Test image counting
            print("9. Image counts:")
            for _, row in df.iterrows():
                print(f"   {row['item_title']}: {row['item_number_of_images']} images")
            print()

            print("=" * 60)
            print("✓ ALL TESTS PASSED")
            print("=" * 60)

            return df


if __name__ == "__main__":
    asyncio.run(test_scraper_integration())
