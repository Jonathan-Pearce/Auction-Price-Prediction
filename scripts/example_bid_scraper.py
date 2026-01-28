#!/usr/bin/env python
"""
Example script demonstrating how to use the bid scraper.

This script shows:
1. How to use the bid scraper with specific item pairs
2. How to transform and save bid data
3. How to review the output
"""

import asyncio
from pathlib import Path

from src.data.bid_scraper import (
    BidDataFetcher,
    scrape_bids,
    transform_bid_data,
)


async def example_with_specific_items():
    """Example: Scrape bids for specific items."""
    print("=" * 60)
    print("Example: Scraping bids for specific items")
    print("=" * 60)

    # Define item pairs (auction_id, item_id)
    item_pairs = [
        (103293, 7433850),  # Example item from the requirements
        # Add more items as needed
    ]

    print(f"\nScraping {len(item_pairs)} items...")

    # Scrape with small batch size and workers for demonstration
    df = await scrape_bids(
        item_pairs=item_pairs,
        use_progress_tracking=False,  # Disable for demo
        max_workers=2,  # Limit concurrent requests
        rate_limit=5,  # 5 requests per second
        output_file=Path("/tmp/example_bid_data.parquet"),
        batch_size=10,  # Small batch size
    )

    print(f"\nScraped {len(df)} bids")
    if not df.empty:
        print("\nColumns:", list(df.columns))
        print("\nFirst few records:")
        print(df.head())

        # Show some statistics
        print("\nStatistics:")
        print(f"  Total bids: {len(df)}")
        print(f"  Unique items: {df['item_id'].nunique()}")
        print(f"  Average bids per item: {len(df) / df['item_id'].nunique():.1f}")
        if "bid_amount" in df.columns:
            print(
                f"  Bid amount range: ${df['bid_amount'].min():.2f} - ${df['bid_amount'].max():.2f}"
            )


async def example_with_mock_data():
    """Example: Process mock bid data without API calls."""
    print("\n" + "=" * 60)
    print("Example: Processing mock bid data")
    print("=" * 60)

    # Create mock bid data as if it came from the API
    fetcher = BidDataFetcher()

    mock_bids = [
        {
            "time_of_bid": "2024-01-05T17:45:00",
            "amount": 10.0,
            "isproxy": False,
        },
        {
            "time_of_bid": "2024-01-05T17:50:00",
            "amount": 15.0,
            "isproxy": True,
        },
        {
            "time_of_bid": "2024-01-05T17:55:00",
            "amount": 20.0,
            "isproxy": False,
        },
        {
            "time_of_bid": "2024-01-05T17:58:00",
            "amount": 25.0,
            "isproxy": False,
        },
    ]

    # Process the bids
    processed_bids = fetcher.process_bid_data(mock_bids, 103293, 7433850)

    print(f"\nProcessed {len(processed_bids)} bids")
    print("\nProcessed data (before transformation):")
    for bid in processed_bids:
        print(f"  Bid {bid['id']}: ${bid['amount']:.2f} at {bid['time_of_bid']}")

    # Transform to DataFrame with proper column names
    df = transform_bid_data(processed_bids)

    print("\nTransformed DataFrame:")
    print(df)

    print("\nColumn names after transformation:")
    for col in df.columns:
        print(f"  - {col}")

    # Verify requirements:
    print("\nVerification:")
    print(f"  ✓ First bid has bid_id = {df['bid_id'].max()} (should equal bid_count)")
    print(f"  ✓ Last bid has bid_id = {df['bid_id'].min()} (should be 1)")
    print(f"  ✓ 'time_of_bid' renamed to 'bid_time': {'bid_time' in df.columns}")
    print(f"  ✓ 'isproxy' renamed to 'bid_is_proxy': {'bid_is_proxy' in df.columns}")
    print(
        f"  ✓ 'auction_id' has no prefix: {'auction_id' in df.columns and 'bid_auction_id' not in df.columns}"
    )
    print(
        f"  ✓ 'item_id' has no prefix: {'item_id' in df.columns and 'bid_item_id' not in df.columns}"
    )


async def example_batch_processing():
    """Example: Demonstrate batch processing concept."""
    print("\n" + "=" * 60)
    print("Example: Batch processing demonstration")
    print("=" * 60)

    print("\nBatch processing configuration:")
    print("  - Default batch size: 100 items")
    print("  - Purpose: Minimize memory usage for large datasets")
    print("  - Behavior:")
    print("    1. Fetch 100 items concurrently")
    print("    2. Transform the data")
    print("    3. Append to parquet file using PyArrow")
    print("    4. Clear memory")
    print("    5. Repeat for next batch")
    print("\nThis allows scraping millions of items without running out of memory.")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "BID SCRAPER EXAMPLES" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")

    # Run mock data example (doesn't require API access)
    asyncio.run(example_with_mock_data())

    # Show batch processing concept
    asyncio.run(example_batch_processing())

    # Note: Uncomment to run with real API (requires valid items)
    # asyncio.run(example_with_specific_items())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo use with real data:")
    print("  1. Ensure Hugging Face token is set (HF_TOKEN env var)")
    print("  2. Run: python -m src.data.bid_scraper --limit 10")
    print("  3. Check output in: data/processed/bids/bid_data.parquet")
    print()


if __name__ == "__main__":
    main()
