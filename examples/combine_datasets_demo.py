#!/usr/bin/env python
"""
Example script demonstrating how to use the combine_engineered_datasets module.

This script shows how to:
1. Load datasets from Hugging Face
2. Merge them on auction_id and item_id
3. Optionally save locally
4. Optionally upload to Hugging Face

Usage:
    python examples/combine_datasets_demo.py
"""

from pathlib import Path

from loguru import logger

from src.combine_engineered_datasets import run_pipeline

# =============================================================================
# Example 1: Basic Usage - Just merge and inspect
# =============================================================================


def example_basic():
    """Load and merge datasets without saving."""
    logger.info("Example 1: Basic merge")

    # Run pipeline without saving or uploading
    merged_df = run_pipeline(upload=False, output_file=None)

    # Inspect the result
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    logger.info(f"First few columns: {merged_df.columns[:10].tolist()}")
    logger.info(f"\nFirst few rows:\n{merged_df.head()}")


# =============================================================================
# Example 2: Save Locally
# =============================================================================


def example_save_local():
    """Merge and save to local file."""
    logger.info("Example 2: Merge and save locally")

    output_path = Path("data/processed/my_combined_data.parquet")

    merged_df = run_pipeline(upload=False, output_file=output_path)

    logger.info(f"Saved {len(merged_df):,} rows to {output_path}")


# =============================================================================
# Example 3: Upload to Hugging Face
# =============================================================================


def example_upload():
    """Merge and upload to Hugging Face (requires HF_TOKEN)."""
    logger.info("Example 3: Merge and upload to HuggingFace")

    # Note: This requires HF_TOKEN to be set in environment or .env
    merged_df = run_pipeline(
        upload=True,
        output_file=Path("data/processed/combined_engineered_data.parquet"),
        repo_id="jpearce610/combined_engineered_data",
    )

    logger.info(f"Uploaded {len(merged_df):,} rows to HuggingFace")


# =============================================================================
# Example 4: Custom Merge with Manual Steps
# =============================================================================


def example_manual():
    """Manually load and merge datasets with custom logic."""
    from src.combine_engineered_datasets import (
        load_auction_features,
        load_bid_features,
        load_item_features,
        merge_datasets,
    )

    logger.info("Example 4: Manual merge with custom logic")

    # Load datasets
    logger.info("Loading datasets...")
    auction_df = load_auction_features()
    item_df = load_item_features()
    bid_df = load_bid_features()

    logger.info(f"Loaded {len(auction_df):,} auctions")
    logger.info(f"Loaded {len(item_df):,} items")
    logger.info(f"Loaded {len(bid_df):,} bids")

    # Merge
    logger.info("Merging...")
    merged_df = merge_datasets(auction_df, item_df, bid_df)

    # Custom analysis
    logger.info("\nDataset Statistics:")
    logger.info(f"  Total rows: {len(merged_df):,}")
    logger.info(f"  Total columns: {len(merged_df.columns)}")
    logger.info(f"  Unique auctions: {merged_df['auction_id'].nunique():,}")
    logger.info(f"  Unique items: {merged_df['item_id'].nunique():,}")

    # Save with custom filename
    output_path = Path("data/processed/custom_combined_data.parquet")
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import sys

    # Run specific example or all
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic()
        elif example_name == "save":
            example_save_local()
        elif example_name == "upload":
            example_upload()
        elif example_name == "manual":
            example_manual()
        else:
            logger.error(f"Unknown example: {example_name}")
            logger.info("Available examples: basic, save, upload, manual")
    else:
        logger.info("Running basic example (use argument to run specific example)")
        logger.info("Available examples: basic, save, upload, manual")
        logger.info("Example: python examples/combine_datasets_demo.py basic")
        logger.info("")
        example_basic()
