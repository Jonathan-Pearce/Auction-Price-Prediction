#!/usr/bin/env python
"""
Demo: Batch Feature Engineering Pipeline

This script demonstrates the new batch processing capability for item feature
engineering, showing how data is processed in chunks to reduce memory usage.
"""

from loguru import logger

from src.features_item import (
    load_enriched_item_data_batched,
    load_item_data,
    process_batch_features,
)


def demo_batch_processing(batch_size: int = 10_000, max_batches: int = 3):
    """
    Demonstrate batch processing with small batches for testing.

    Args:
        batch_size: Number of rows per batch (default 10k for demo)
        max_batches: Maximum number of batches to process (for demo only)
    """
    logger.info("=" * 60)
    logger.info("Batch Feature Engineering Demo")
    logger.info("=" * 60)

    # Load item data (small dataset, loaded once)
    logger.info("Loading item data...")
    item_df = load_item_data()
    logger.info(f"Loaded {len(item_df):,} item records")

    # Process enriched data in batches
    logger.info(f"\nProcessing enriched data in batches of {batch_size:,} rows...")
    logger.info(f"(Limited to {max_batches} batches for demo)\n")

    processed_batches = []
    for batch_num, enriched_batch in enumerate(
        load_enriched_item_data_batched(batch_size), 1
    ):
        logger.info(f"--- Batch #{batch_num} ---")
        logger.info(f"Raw batch shape: {enriched_batch.shape}")

        # Process this batch
        engineered_batch = process_batch_features(enriched_batch, item_df)
        processed_batches.append(engineered_batch)

        logger.info(f"Engineered batch shape: {engineered_batch.shape}")
        logger.info(f"Features created: {len(engineered_batch.columns)} columns\n")

        # Stop after max_batches for demo
        if batch_num >= max_batches:
            logger.info(f"Demo limit reached ({max_batches} batches). Stopping.\n")
            break

    # Show results
    logger.info("=" * 60)
    logger.info("Demo Summary")
    logger.info("=" * 60)
    logger.info(f"Batches processed: {len(processed_batches)}")
    logger.info(f"Total rows processed: {sum(len(b) for b in processed_batches):,}")

    if processed_batches:
        example_batch = processed_batches[0]
        logger.info(f"\nExample features created:")
        for col in sorted(example_batch.columns):
            logger.info(f"  - {col}")


if __name__ == "__main__":
    demo_batch_processing()
