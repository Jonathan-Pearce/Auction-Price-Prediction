#!/usr/bin/env python
"""
Example usage of the auction scraper.

This script demonstrates various ways to use the auction scraper
for different use cases.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import scraper_config
from src.data.auction_scraper import scrape_auctions, upload_to_huggingface


async def example_1_basic_usage():
    """Example 1: Basic usage with limit."""
    print("=" * 60)
    print("Example 1: Basic Usage (First 10 auctions)")
    print("=" * 60)

    df = await scrape_auctions(limit=10)

    print(f"\nScraped {len(df)} auctions")
    print(f"Columns: {', '.join(df.columns)}")
    if not df.empty:
        print("\nFirst few rows:")
        print(df.head())


async def example_2_specific_auctions():
    """Example 2: Scrape specific auction IDs."""
    print("\n" + "=" * 60)
    print("Example 2: Specific Auction IDs")
    print("=" * 60)

    # Replace these with actual auction IDs
    auction_ids = [99941, 99942, 99943]

    df = await scrape_auctions(
        auction_ids=auction_ids,
        use_progress_tracking=False,  # Disable for small runs
    )

    print(f"\nScraped {len(df)} auctions")
    if not df.empty:
        print("\nAuction summary:")
        print(df[["auction_id", "auction_title", "auction_item_count"]].to_string())


async def example_3_parallel_processing():
    """Example 3: Fast scraping with more workers."""
    print("\n" + "=" * 60)
    print("Example 3: Parallel Processing (10 workers)")
    print("=" * 60)

    df = await scrape_auctions(
        limit=50,
        max_workers=10,
        rate_limit=20,  # Increase rate limit
    )

    print(f"\nScraped {len(df)} auctions with parallel processing")


async def example_4_aggregated_metrics():
    """Example 4: Analyze aggregated metrics."""
    print("\n" + "=" * 60)
    print("Example 4: Analyzing Aggregated Metrics")
    print("=" * 60)

    df = await scrape_auctions(limit=20)

    if not df.empty:
        print("\nAuction Statistics:")
        print(f"Total auctions: {len(df)}")
        print(f"Total items across all auctions: {df['auction_item_count'].sum()}")
        print(
            f"Total winning price across all auctions: ${df['auction_total_winning_price'].sum():.2f}"
        )
        print(f"Total bids across all auctions: {df['auction_total_bid_count'].sum()}")
        print(f"Total images across all auctions: {df['auction_total_images'].sum()}")

        print("\nAverage metrics per auction:")
        print(f"  Items: {df['auction_item_count'].mean():.1f}")
        print(f"  Winning price: ${df['auction_total_winning_price'].mean():.2f}")
        print(f"  Bids: {df['auction_total_bid_count'].mean():.1f}")
        print(f"  Images: {df['auction_total_images'].mean():.1f}")


async def example_5_custom_output():
    """Example 5: Save to custom location."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Output Location")
    print("=" * 60)

    output_file = Path("/tmp/my_auction_data.parquet")

    _ = await scrape_auctions(limit=5, output_file=output_file)

    print(f"\nData saved to: {output_file}")
    print(f"File exists: {output_file.exists()}")


async def example_6_huggingface_upload():
    """Example 6: Upload to Hugging Face (requires token)."""
    print("\n" + "=" * 60)
    print("Example 6: Hugging Face Upload")
    print("=" * 60)
    print("NOTE: This requires HF_TOKEN to be set in environment")

    # First scrape some data
    _ = await scrape_auctions(limit=5)

    # Then upload (will fail if token not configured)
    try:
        await upload_to_huggingface(
            repo_id="your-username/test-dataset",  # Change this!
            private=True,
        )
        print("\n✓ Upload successful!")
    except Exception as e:
        print(f"\n⚠ Upload failed (expected if HF_TOKEN not set): {e}")


def example_7_configuration():
    """Example 7: View configuration."""
    print("\n" + "=" * 60)
    print("Example 7: Configuration Settings")
    print("=" * 60)

    print("\nAPI Configuration:")
    print(f"  Base URL: {scraper_config.MAXSOLD_API_BASE_URL}")
    print(f"  Items limit: {scraper_config.DEFAULT_ITEMS_LIMIT}")
    print(f"  Rate limit: {scraper_config.DEFAULT_RATE_LIMIT} req/sec")
    print(f"  Concurrent requests: {scraper_config.CONCURRENT_REQUESTS}")

    print("\nField Mappings:")
    for old, new in scraper_config.FIELD_RENAME_MAP.items():
        print(f"  {old} → {new}")

    print(f"\nColumn Prefix: '{scraper_config.COLUMN_PREFIX}'")

    print("\nData Files:")
    print(f"  Input: {scraper_config.AUCTION_IDS_FILE}")
    print(f"  Output: {scraper_config.PROCESSED_OUTPUT_DIR}")


async def main():
    """Run all examples."""
    print("\n")
    print("#" * 60)
    print("# AUCTION SCRAPER - USAGE EXAMPLES")
    print("#" * 60)

    # Configuration example (sync)
    example_7_configuration()

    # Basic examples (async)
    # Uncomment the examples you want to run:

    # await example_1_basic_usage()
    # await example_2_specific_auctions()
    # await example_3_parallel_processing()
    # await example_4_aggregated_metrics()
    # await example_5_custom_output()
    # await example_6_huggingface_upload()

    print("\n" + "#" * 60)
    print("# Examples complete!")
    print("#" * 60)
    print("\nTo run specific examples, uncomment them in main()")
    print("Warning: These examples will make real API calls!")


if __name__ == "__main__":
    asyncio.run(main())
