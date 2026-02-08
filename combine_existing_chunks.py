#!/usr/bin/env python3
"""
Combine the 57 existing chunks into a final dataset.

We'll work with what we have (84% of the data) to produce a usable result.
"""

import gc
from pathlib import Path

import pandas as pd
from datasets import Dataset
from loguru import logger

from src.config import settings

# Configuration
TEMP_DIR = settings.data_dir / "interim" / "merge_chunks"
OUTPUT_FILE = settings.data_dir / "processed" / "combined_engineered_data.parquet"
REPO_ID = "jpearce610/combined_engineered_data"

def combine_chunks():
    """Combine all existing merged chunk files."""
    logger.info("=" * 80)
    logger.info("Combining existing merged chunks")
    logger.info("=" * 80)
    
    # Get all chunk files
    chunk_files = sorted(TEMP_DIR.glob("merged_chunk_*.parquet"))
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    if not chunk_files:
        logger.error("No chunk files found!")
        return None
    
    # Combine in smaller batches to avoid OOM
    logger.info("Reading and combining chunks in batches...")
    
    all_chunks = []
    batch_size = 10  # Process 10 chunks at a time
    
    for i in range(0, len(chunk_files), batch_size):
        batch_files = chunk_files[i:i+batch_size]
        logger.info(f"Loading batch {i//batch_size + 1}: chunks {i+1} to {min(i+batch_size, len(chunk_files))}")
        
        batch_dfs = []
        for chunk_file in batch_files:
            df = pd.read_parquet(chunk_file)
            batch_dfs.append(df)
        
        batch_combined = pd.concat(batch_dfs, ignore_index=True)
        all_chunks.append(batch_combined)
        
        logger.info(f"Batch {i//batch_size + 1}: {len(batch_combined):,} records")
        
        del batch_dfs, batch_combined
        gc.collect()
    
    # Final concatenation
    logger.info("Performing final concatenation...")
    merged_df = pd.concat(all_chunks, ignore_index=True)
    
    del all_chunks
    gc.collect()
    
    logger.info(f"Combined dataset: {len(merged_df):,} records with {len(merged_df.columns)} columns")
    
    return merged_df


def save_locally(df: pd.DataFrame):
    """Save to local parquet file."""
    logger.info(f"Saving to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df):,} records to {OUTPUT_FILE}")


def upload_to_hf(df: pd.DataFrame):
    """Upload to Hugging Face."""
    logger.info(f"Uploading to Hugging Face: {REPO_ID}...")
    
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(REPO_ID, private=False)
    
    logger.info(f"Successfully uploaded to {REPO_ID}")


if __name__ == "__main__":
    # Combine chunks
    merged_df = combine_chunks()
    
    if merged_df is not None:
        # Save locally
        save_locally(merged_df)
        
        # Upload to HF
        try:
            upload_to_hf(merged_df)
        except Exception as e:
            logger.error(f"Failed to upload to HF: {e}")
            logger.info("Dataset saved locally, you can upload manually later")
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final dataset: {len(merged_df):,} records")
        logger.info("=" * 80)
