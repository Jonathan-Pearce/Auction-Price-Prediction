#!/usr/bin/env python
"""
Example usage of the enriched item scraper.

This script demonstrates how to use the enriched_item_scraper module
programmatically for different use cases.
"""

import asyncio
from pathlib import Path

from src.data.enriched_item_scraper import (
    EnrichedItemDataFetcher,
    scrape_enriched_items,
    transform_enriched_item_data,
    upload_to_huggingface,
)


async def example_basic_scraping():
    """Example 1: Basic scraping with default settings."""
    print("Example 1: Basic scraping")
    print("-" * 50)
    
    # Scrape first 10 items
    df = await scrape_enriched_items(
        limit=10,
        use_progress_tracking=True,
    )
    
    print(f"Scraped {len(df)} items")
    print(f"Columns: {list(df.columns)}")
    print("\nSample data:")
    print(df.head(3))
    print()


async def example_custom_settings():
    """Example 2: Scraping with custom settings."""
    print("Example 2: Custom settings")
    print("-" * 50)
    
    df = await scrape_enriched_items(
        limit=50,
        max_workers=10,  # More parallel workers
        rate_limit=8,    # Lower rate limit to be more conservative
        batch_size=25,   # Smaller batches
    )
    
    print(f"Scraped {len(df)} items with custom settings")
    print()


async def example_specific_items():
    """Example 3: Scraping specific item IDs."""
    print("Example 3: Specific item IDs")
    print("-" * 50)
    
    # Define specific item IDs to scrape
    item_ids = [7433915, 7433916, 7433917]
    
    df = await scrape_enriched_items(
        item_ids=item_ids,
        use_progress_tracking=False,  # No need for progress tracking
    )
    
    print(f"Scraped {len(df)} specific items")
    print(df[["item_id", "auction_id"]])
    print()


async def example_fetcher_low_level():
    """Example 4: Using EnrichedItemDataFetcher directly (low-level API)."""
    print("Example 4: Low-level API")
    print("-" * 50)
    
    async with EnrichedItemDataFetcher(rate_limit=5, max_concurrent=3) as fetcher:
        # Fetch single item
        item_data = await fetcher.fetch_and_process_item(7433915)
        
        if item_data:
            print(f"Fetched item: {item_data.get('amLotId')}")
            print(f"Generated description title: {item_data.get('generatedDescription_title')}")
        
        # Fetch multiple items
        items = await fetcher.fetch_multiple_items([7433916, 7433917])
        print(f"\nFetched {len(items)} items in batch")
    
    # Transform to DataFrame
    if items:
        df = transform_enriched_item_data(items)
        print(f"Transformed to DataFrame with {len(df)} rows")
    
    print()


async def example_with_upload():
    """Example 5: Scraping and uploading to Hugging Face."""
    print("Example 5: Scraping with HF upload")
    print("-" * 50)
    
    # Scrape data
    output_file = Path("data/processed/enriched_items/example_enriched_data.parquet")
    
    df = await scrape_enriched_items(
        limit=20,
        output_file=output_file,
    )
    
    print(f"Scraped {len(df)} items")
    
    # Upload to Hugging Face (requires HF_TOKEN in environment)
    # Uncomment to actually upload:
    # await upload_to_huggingface(
    #     data_file=output_file,
    #     repo_id="your-username/enriched-item-data",
    #     private=False,
    # )
    
    print("(Upload step commented out - set HF_TOKEN and repo_id to enable)")
    print()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Enriched Item Scraper - Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    # Note: Comment out examples that require API access
    
    # await example_basic_scraping()
    # await example_custom_settings()
    # await example_specific_items()
    # await example_fetcher_low_level()
    # await example_with_upload()
    
    print("Examples completed! (Uncomment function calls to run)")
    print()
    print("Quick start:")
    print("  make scrape-enriched-sample  # Scrape 100 items")
    print("  make scrape-enriched         # Scrape all items")
    print()


if __name__ == "__main__":
    asyncio.run(main())
