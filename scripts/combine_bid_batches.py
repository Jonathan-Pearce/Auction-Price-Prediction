"""
Combine pre-computed bid feature batches using memory-efficient DuckDB approach.
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import duckdb
import pandas as pd
from loguru import logger
from datasets import Dataset

# Authenticate with Hugging Face
from huggingface_hub import HfApi
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logger.error("HF_TOKEN not found in environment. Please set it in .env file.")
    exit(1)

logger.info(f"Using HF token: {hf_token[:10]}...")

# Paths
batch_dir = Path("data/interim/feature_batches_bid")
output_file = Path("data/processed/bid_features.parquet")

logger.info(f"Combining batch files from {batch_dir}")

# Check batch files exist
batch_files = sorted(batch_dir.glob("batch_*.parquet"))
logger.info(f"Found {len(batch_files)} batch files")

if len(batch_files) == 0:
    logger.error("No batch files found!")
    exit(1)

# Create a temporary DuckDB connection
conn = duckdb.connect(database=":memory:")

# Create file pattern for DuckDB
file_pattern = str(batch_dir / "batch_*.parquet")

# Get total row count first
logger.info(f"Reading from pattern: {file_pattern}")
count_query = f"SELECT COUNT(*) as total FROM read_parquet('{file_pattern}')"
total_rows = conn.execute(count_query).fetchone()[0]
logger.info(f"Total rows across all batches: {total_rows:,}")

# Use DuckDB to write directly to parquet file (most memory efficient)
logger.info(f"Writing directly to {output_file}")
output_file.parent.mkdir(parents=True, exist_ok=True)

# DuckDB can write directly to parquet without loading into memory
write_query = f"""
    COPY (SELECT * FROM read_parquet('{file_pattern}'))
    TO '{output_file}' (FORMAT PARQUET);
"""
conn.execute(write_query)
logger.info(f"Successfully wrote {total_rows:,} rows to {output_file}")

# Close connection
conn.close()

# Upload to HuggingFace in streaming mode to avoid memory issues
logger.info("Uploading to HuggingFace in streaming mode...")

# Read and upload in chunks
chunk_size = 1_000_000
offset = 0
first_chunk = True

conn = duckdb.connect(database=":memory:")
query = f"SELECT * FROM read_parquet('{output_file}')"

while offset < total_rows:
    chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
    chunk_df = conn.execute(chunk_query).df()
    
    if len(chunk_df) == 0:
        break
    
    chunk_dataset = Dataset.from_pandas(chunk_df)
    
    if first_chunk:
        # First chunk creates the dataset
        chunk_dataset.push_to_hub(
            "jpearce610/engineered_bid_data",
            private=False,
            token=hf_token,
        )
        first_chunk = False
        logger.info(f"Uploaded initial chunk: {offset:,} to {offset + len(chunk_df):,}")
    else:
        # Subsequent chunks append
        from datasets import load_dataset
        existing_dataset = load_dataset("jpearce610/engineered_bid_data", split="train", token=hf_token)
        combined = pd.concat([existing_dataset.to_pandas(), chunk_df], ignore_index=True)
        combined_dataset = Dataset.from_pandas(combined)
        combined_dataset.push_to_hub(
            "jpearce610/engineered_bid_data",
            private=False,
            token=hf_token,
        )
        logger.info(f"Uploaded chunk: {offset:,} to {offset + len(chunk_df):,}")
    
    offset += chunk_size

conn.close()
logger.info("Upload complete!")
