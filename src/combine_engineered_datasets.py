# =============================================================================
# Auction Price Prediction - Combine Engineered Datasets
# =============================================================================
"""
Pipeline to combine engineered auction, item, and bid datasets.

This module downloads three engineered datasets from Hugging Face:
1. jpearce610/engineered_auction_data
2. jpearce610/engineered_item_data
3. jpearce610/engineered_bid_data

The datasets are merged on auction_id and item_id, keeping all columns.
The merged dataset is then uploaded to Hugging Face.

Key features:
- Downloads auction and item datasets fully into memory (smaller datasets)
- Streams bid dataset in chunks to avoid OOM errors (33M+ records, 1.85GB)
- Merges in chunks and saves temporary results to disk
- Combines chunks into final dataset
- Left merge to keep all bid records with matching auction/item data
- Ensures auction_id and item_id are first two columns
- Preserves all columns from all datasets

Memory-efficient design:
- Only one bid chunk in memory at a time
- Intermediate results written to disk
- Explicit cleanup with del statements and garbage collection
- Configurable chunk size (default: 100k records for low-memory environments)
"""

import gc
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from loguru import logger

from src.config import settings

# =============================================================================
# Data Loading
# =============================================================================


def load_auction_features() -> pd.DataFrame:
    """
    Load engineered auction features from Hugging Face.

    Returns:
        DataFrame with auction-level features
    """
    logger.info("Loading engineered auction data from HuggingFace...")
    dataset = load_dataset("jpearce610/engineered_auction_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} auction records with {len(df.columns)} columns")
    return df


def load_item_features() -> pd.DataFrame:
    """
    Load engineered item features from Hugging Face.

    Returns:
        DataFrame with item-level features
    """
    logger.info("Loading engineered item data from HuggingFace...")
    dataset = load_dataset("jpearce610/engineered_item_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} item records with {len(df.columns)} columns")
    return df


def load_bid_features_streaming(chunk_size: int = 100_000):
    """
    Load engineered bid features from Hugging Face in streaming mode.

    The bid dataset is very large (33M+ records, 1.85GB). Loading it all
    at once causes OOM errors. This generator yields chunks of bid data
    for memory-efficient processing.

    Uses the datasets library's batched iteration for much faster streaming
    than record-by-record.

    Args:
        chunk_size: Number of records to load per chunk (default 100K for low-memory environments)

    Yields:
        DataFrames with bid-level features, one chunk at a time
    """
    logger.info(
        f"Loading engineered bid data from HuggingFace in streaming mode "
        f"(chunk_size={chunk_size:,})..."
    )

    # Load dataset in streaming mode
    dataset = load_dataset(
        "jpearce610/engineered_bid_data", split="train", streaming=True
    )

    # Use iter_batches for much faster streaming (batched iteration)
    chunk_count = 0
    for batch in dataset.iter(batch_size=chunk_size):
        chunk_count += 1
        df = pd.DataFrame(batch)
        logger.info(
            f"Yielding bid chunk {chunk_count}: {len(df):,} records "
            f"with {len(df.columns)} columns"
        )
        yield df

    logger.info(f"Finished streaming {chunk_count} bid data chunks")


# =============================================================================
# Merging Functions
# =============================================================================


def merge_auction_and_item(
    auction_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge auction and item datasets on auction_id.

    Args:
        auction_df: DataFrame with auction-level features
        item_df: DataFrame with item-level features

    Returns:
        Merged DataFrame with auction and item features
    """
    logger.info("Merging auction and item data on auction_id...")

    # Check if item_df has auction_id column
    if "auction_id" not in item_df.columns:
        logger.warning(
            "item_df does not have auction_id column, returning item_df as-is"
        )
        return item_df.copy()

    auction_item_df = pd.merge(
        auction_df,
        item_df,
        on="auction_id",
        how="outer",
        suffixes=("_auction", "_item"),
    )
    logger.info(
        f"After auction+item merge: {len(auction_item_df):,} records "
        f"with {len(auction_item_df.columns)} columns"
    )

    return auction_item_df


def merge_with_bid_chunks(
    auction_item_df: pd.DataFrame,
    bid_chunks_generator,
    temp_dir: Path,
) -> list[Path]:
    """
    Merge auction+item data with bid data chunks in a memory-efficient way.

    Instead of loading all bid data at once (OOM), we merge in chunks and
    save each result to disk.

    Args:
        auction_item_df: Merged auction+item DataFrame
        bid_chunks_generator: Generator yielding bid data chunks
        temp_dir: Directory to save temporary chunk files

    Returns:
        List of paths to merged chunk files
    """
    logger.info("Merging with bid data chunks (memory-efficient mode)...")

    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Determine merge keys
    merge_keys = []
    bid_chunk_sample = None

    # Get first chunk to determine merge keys
    try:
        bid_chunk_sample = next(bid_chunks_generator)
    except StopIteration:
        logger.warning("No bid data chunks available")
        return []

    if "auction_id" in auction_item_df.columns and "auction_id" in bid_chunk_sample.columns:
        merge_keys.append("auction_id")
    if "item_id" in auction_item_df.columns and "item_id" in bid_chunk_sample.columns:
        merge_keys.append("item_id")

    if not merge_keys:
        logger.error("No common keys found for merging!")
        raise ValueError("Cannot merge datasets: no common keys (auction_id, item_id)")

    logger.info(f"Merging on keys: {merge_keys}")

    # Process first chunk
    chunk_files = []
    chunk_num = 1

    logger.info(f"Processing bid chunk {chunk_num}...")
    merged_chunk = pd.merge(
        bid_chunk_sample,
        auction_item_df,
        on=merge_keys,
        how="left",
        suffixes=("_bid", ""),
    )
    chunk_file = temp_dir / f"merged_chunk_{chunk_num:04d}.parquet"
    merged_chunk.to_parquet(chunk_file, index=False)
    chunk_files.append(chunk_file)
    logger.info(
        f"Chunk {chunk_num}: {len(merged_chunk):,} records saved to {chunk_file.name}"
    )
    del merged_chunk, bid_chunk_sample
    gc.collect()

    # Process remaining chunks
    for bid_chunk in bid_chunks_generator:
        chunk_num += 1
        logger.info(f"Processing bid chunk {chunk_num}...")

        merged_chunk = pd.merge(
            bid_chunk,
            auction_item_df,
            on=merge_keys,
            how="left",
            suffixes=("_bid", ""),
        )

        chunk_file = temp_dir / f"merged_chunk_{chunk_num:04d}.parquet"
        merged_chunk.to_parquet(chunk_file, index=False)
        chunk_files.append(chunk_file)

        logger.info(
            f"Chunk {chunk_num}: {len(merged_chunk):,} records saved to {chunk_file.name}"
        )

        # Explicit cleanup
        del merged_chunk, bid_chunk
        gc.collect()

    logger.info(
        f"Finished merging {chunk_num} bid chunks. Saved {len(chunk_files)} chunk files."
    )

    return chunk_files


def combine_merged_chunks(
    chunk_files: list[Path], output_file: Path | None = None
) -> pd.DataFrame | None:
    """
    Combine merged chunk files into a single dataset.

    Uses a memory-efficient streaming approach with PyArrow that writes directly
    to a Parquet file without loading all chunks into memory at once.

    Args:
        chunk_files: List of paths to merged chunk files
        output_file: If provided, writes directly to this file and returns None.
                    If None, loads into memory and returns DataFrame.

    Returns:
        Combined DataFrame if output_file is None, otherwise None
    """
    logger.info(f"Combining {len(chunk_files)} merged chunk files...")

    if not chunk_files:
        raise ValueError("No chunk files provided")

    # Read first chunk to determine schema and column order
    logger.info(f"Reading first chunk to determine schema: {chunk_files[0].name}...")
    first_chunk = pd.read_parquet(chunk_files[0])

    # Reorder columns to ensure auction_id and item_id are first
    logger.info("Determining column order (auction_id and item_id first)...")
    cols = list(first_chunk.columns)
    first_cols = []
    if "auction_id" in cols:
        first_cols.append("auction_id")
        cols.remove("auction_id")
    if "item_id" in cols:
        first_cols.append("item_id")
        cols.remove("item_id")

    column_order = first_cols + cols
    logger.info(f"Column order: {column_order[:5]} ...")

    # If output_file is provided, use streaming write with PyArrow
    if output_file:
        logger.info(f"Using PyArrow streaming write to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Reorder first chunk and convert to Arrow table
        first_chunk = first_chunk[column_order]
        schema = pa.Schema.from_pandas(first_chunk)

        # Open Parquet writer
        with pq.ParquetWriter(output_file, schema) as writer:
            # Write first chunk
            table = pa.Table.from_pandas(first_chunk, schema=schema)
            writer.write_table(table)
            row_count = len(first_chunk)
            del first_chunk, table

            logger.info(f"Wrote chunk 1/{len(chunk_files)} ({row_count:,} rows)")

            # Stream remaining chunks
            for i, chunk_file in enumerate(chunk_files[1:], 2):
                logger.info(
                    f"Appending chunk {i}/{len(chunk_files)}: {chunk_file.name}..."
                )
                chunk_df = pd.read_parquet(chunk_file)
                chunk_df = chunk_df[column_order]

                # Convert to Arrow table and write
                table = pa.Table.from_pandas(chunk_df, schema=schema)
                writer.write_table(table)

                row_count += len(chunk_df)
                del chunk_df, table

                if i % 10 == 0:
                    logger.info(
                        f"Progress: {i}/{len(chunk_files)} chunks, {row_count:,} total rows"
                    )

        logger.info(f"Finished combining! Total rows: {row_count:,}")
        logger.info(f"Output written to: {output_file}")
        return None

    else:
        # Load into memory (original behavior)
        logger.info("Loading all chunks into memory...")
        chunks = [first_chunk[column_order]]

        for i, chunk_file in enumerate(chunk_files[1:], 2):
            logger.info(f"Loading chunk {i}/{len(chunk_files)}: {chunk_file.name}...")
            chunk_df = pd.read_parquet(chunk_file)
            chunks.append(chunk_df[column_order])

        logger.info("Concatenating all chunks...")
        merged_df = pd.concat(chunks, ignore_index=True)

        logger.info(
            f"Combined dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns"
        )

        return merged_df


# =============================================================================
# Upload Function
# =============================================================================


def upload_to_huggingface(
    df: pd.DataFrame,
    repo_id: str = "jpearce610/combined_engineered_data",
    token: str | None = None,
) -> None:
    """
    Upload merged dataset to Hugging Face.

    Args:
        df: DataFrame to upload
        repo_id: Hugging Face repository ID
        token: HF API token (uses HF_TOKEN env var if not provided)
    """
    logger.info(f"Uploading merged dataset to {repo_id}...")

    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN") or settings.huggingface.token
        if token is None:
            raise ValueError("HF_TOKEN not found. Set it in .env or pass as argument.")

    # Convert DataFrame to HuggingFace Dataset
    logger.info("Converting DataFrame to HuggingFace Dataset...")
    dataset = Dataset.from_pandas(df)

    # Push to Hub
    logger.info(f"Pushing dataset to {repo_id}...")
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=False,
    )

    logger.info(f"Dataset uploaded successfully to {repo_id}")


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    upload: bool = False,
    output_file: Path | None = None,
    repo_id: str = "jpearce610/combined_engineered_data",
    bid_chunk_size: int = 250_000,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run the full pipeline to combine engineered datasets.

    Uses memory-efficient chunked processing for the large bid dataset
    to avoid OOM errors.

    Args:
        upload: Whether to upload result to Hugging Face
        output_file: Optional path to save local copy
        repo_id: Hugging Face repository ID for upload
        bid_chunk_size: Number of bid records to process per chunk (default 100K for low-memory environments)
        temp_dir: Directory for temporary chunk files (default: data/interim/merge_chunks)

    Returns:
        Merged DataFrame
    """
    logger.info("=" * 80)
    logger.info("Starting Engineered Dataset Combination Pipeline")
    logger.info("=" * 80)

    # Set default temp directory
    if temp_dir is None:
        temp_dir = settings.data_dir / "interim" / "merge_chunks"

    # Step 1: Load auction and item datasets
    logger.info("\n[Step 1/5] Loading auction and item datasets from Hugging Face...")
    auction_df = load_auction_features()
    item_df = load_item_features()

    # Step 2: Merge auction + item
    logger.info("\n[Step 2/5] Merging auction and item datasets...")
    auction_item_df = merge_auction_and_item(auction_df, item_df)

    # Clean up to free memory
    del auction_df, item_df

    # Step 3: Stream bid data and merge in chunks
    logger.info(
        f"\n[Step 3/5] Streaming bid data and merging in chunks "
        f"(chunk_size={bid_chunk_size:,})..."
    )
    bid_chunks = load_bid_features_streaming(chunk_size=bid_chunk_size)
    chunk_files = merge_with_bid_chunks(auction_item_df, bid_chunks, temp_dir)

    # Clean up to free memory
    del auction_item_df

    # Step 4: Combine merged chunks
    logger.info("\n[Step 4/5] Combining merged chunks into final dataset...")

    # Determine if we should use streaming write or load into memory
    # If we have output_file or need to upload, we need to handle accordingly
    if output_file and not upload:
        # Case 1: Only saving locally - use streaming write directly
        logger.info(f"Using memory-efficient streaming write to {output_file}...")
        combine_merged_chunks(chunk_files, output_file=output_file)
        merged_df = None  # No need to load into memory

    elif upload and not output_file:
        # Case 2: Only uploading - load into memory for upload
        logger.info("Loading into memory for Hugging Face upload...")
        merged_df = combine_merged_chunks(chunk_files, output_file=None)

    elif upload and output_file:
        # Case 3: Both saving and uploading - write to file first, then read for upload
        logger.info(f"Writing to {output_file} then loading for upload...")
        combine_merged_chunks(chunk_files, output_file=output_file)
        logger.info("Reading back for Hugging Face upload...")
        merged_df = pd.read_parquet(output_file)

    else:
        # Case 4: Neither saving nor uploading - load into memory anyway
        logger.info("Loading into memory...")
        merged_df = combine_merged_chunks(chunk_files, output_file=None)

    # Clean up temp files
    logger.info("Cleaning up temporary chunk files...")
    for chunk_file in chunk_files:
        chunk_file.unlink()
    logger.info(f"Removed {len(chunk_files)} temporary chunk files")

    # Step 5a: Save locally (optional) - may already be done in step 4
    if output_file and not upload:
        logger.info("\n[Step 5a/5] Local save completed in previous step")
    elif output_file and upload:
        logger.info("\n[Step 5a/5] Local save completed, now uploading...")
    elif not output_file:
        logger.info("\n[Step 5a/5] Skipping local save (no output_file specified)")

    # Step 5b: Upload to Hugging Face (optional)
    if upload:
        logger.info("\n[Step 5b/5] Uploading to Hugging Face...")
        upload_to_huggingface(merged_df, repo_id=repo_id)
    else:
        logger.info("\n[Step 5b/5] Skipping upload (use --upload flag to enable)")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)

    return merged_df


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the dataset combination pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine engineered auction, item, and bid datasets"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload result to Hugging Face",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save local copy (default: data/processed/combined_engineered_data.parquet)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="jpearce610/combined_engineered_data",
        help="Hugging Face repository ID for upload",
    )
    parser.add_argument(
        "--bid-chunk-size",
        type=int,
        default=250_000,
        help="Number of bid records to process per chunk (default: 250,000)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for temporary chunk files (default: data/interim/merge_chunks)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    output_file = None
    if args.output:
        output_file = Path(args.output)
    else:
        # Default output location
        output_file = (
            settings.data_dir / "processed" / "combined_engineered_data.parquet"
        )

    # Set temp directory
    temp_dir = None
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)

    # Run pipeline
    run_pipeline(
        upload=args.upload,
        output_file=output_file,
        repo_id=args.repo_id,
        bid_chunk_size=args.bid_chunk_size,
        temp_dir=temp_dir,
    )


if __name__ == "__main__":
    main()
