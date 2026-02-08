#!/usr/bin/env python3
"""
Resume merging from where we left off.

This script continues processing the remaining bid chunks (58-68)
and then combines all chunks into the final dataset.
"""

import gc
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger

from src.config import settings

# Configuration
CHUNK_SIZE = 500_000
START_CHUNK = 58  # Resume from chunk 58
TEMP_DIR = settings.data_dir / "interim" / "merge_chunks"

def load_auction_item_merged() -> pd.DataFrame:
    """
    Load the pre-merged auction+item dataset.
    
    Since we're resuming, we need to recreate this or load from cache.
    """
    logger.info("Loading auction and item datasets...")
    
    # Load from HuggingFace
    auction_dataset = load_dataset("jpearce610/engineered_auction_data", split="train")
    auction_df = auction_dataset.to_pandas()
    logger.info(f"Loaded {len(auction_df):,} auction records")
    
    item_dataset = load_dataset("jpearce610/engineered_item_data", split="train")
    item_df = item_dataset.to_pandas()
    logger.info(f"Loaded {len(item_df):,} item records")
    
    # Merge
    logger.info("Merging auction + item...")
    merged_df = pd.merge(
        auction_df,
        item_df,
        on="auction_id",
        how="outer",
        suffixes=("", "_item"),
    )
    
    # Reorder columns
    if "auction_id" in merged_df.columns and "item_id" in merged_df.columns:
        other_cols = [c for c in merged_df.columns if c not in ["auction_id", "item_id"]]
        merged_df = merged_df[["auction_id", "item_id"] + other_cols]
    
    logger.info(f"Merged dataset: {len(merged_df):,} records")
    
    del auction_df, item_df
    gc.collect()
    
    return merged_df


def process_remaining_chunks():
    """Process the remaining bid chunks (58-68)."""
    logger.info("=" * 80)
    logger.info(f"Resuming merge from chunk {START_CHUNK}")
    logger.info("=" * 80)
    
    # Load auction+item data
    auction_item_df = load_auction_item_merged()
    
    # Load bid dataset - use skip() to avoid slow iteration
    records_to_skip = (START_CHUNK - 1) * CHUNK_SIZE
    logger.info(f"Loading bid dataset, skipping to record {records_to_skip:,} (chunk {START_CHUNK})...")
    
    bid_dataset = load_dataset(
        "jpearce610/engineered_bid_data",
        split="train",
        streaming=True
    )
    
    # Skip efficiently
    bid_dataset_remaining = bid_dataset.skip(records_to_skip)
    
    logger.info(f"Reached starting point. Processing remaining chunks...")
    
    # Process in chunks using iter_batches
    chunk_num = START_CHUNK
    
    for batch in bid_dataset_remaining.iter(batch_size=CHUNK_SIZE):
        bid_chunk_df = pd.DataFrame(batch)
        logger.info(f"Processing chunk {chunk_num}: {len(bid_chunk_df):,} records...")
        
        # Merge
        merged_chunk = pd.merge(
            bid_chunk_df,
            auction_item_df,
            on=["auction_id", "item_id"],
            how="left",
            suffixes=("_bid", ""),
        )
        
        # Save
        chunk_file = TEMP_DIR / f"merged_chunk_{chunk_num:04d}.parquet"
        merged_chunk.to_parquet(chunk_file, index=False)
        logger.info(f"Saved chunk {chunk_num}: {len(merged_chunk):,} records -> {chunk_file.name}")
        
        # Cleanup
        del bid_chunk_df, merged_chunk
        gc.collect()
        
        chunk_num += 1
    
    logger.info("=" * 80)
    logger.info(f"Finished processing remaining chunks")
    logger.info("=" * 80)


if __name__ == "__main__":
    process_remaining_chunks()
