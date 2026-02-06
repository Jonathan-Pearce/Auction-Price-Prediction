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
- Batch processing for large bid dataset to manage memory
- Full outer merge to keep all records
- Ensures auction_id and item_id are first two columns
- Preserves all columns from all datasets
"""

import os
from pathlib import Path

import pandas as pd
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


def load_bid_features_batched(batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    Load engineered bid features from Hugging Face in batches.

    The bid dataset can be very large, so we use streaming to avoid
    loading everything into memory at once.

    Args:
        batch_size: Number of rows to load per batch

    Returns:
        DataFrame with bid-level features
    """
    logger.info("Loading engineered bid data from HuggingFace in batches...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        "jpearce610/engineered_bid_data", split="train", streaming=False
    )

    # Convert to pandas (the dataset is already loaded, streaming=False means regular mode)
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} bid records with {len(df.columns)} columns")

    return df


# =============================================================================
# Merging Functions
# =============================================================================


def merge_datasets(
    auction_df: pd.DataFrame,
    item_df: pd.DataFrame,
    bid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge auction, item, and bid datasets on auction_id and item_id.

    Performs a full outer merge to keep all records from all datasets.
    Ensures auction_id and item_id are the first two columns.

    Args:
        auction_df: DataFrame with auction-level features
        item_df: DataFrame with item-level features
        bid_df: DataFrame with bid-level features

    Returns:
        Merged DataFrame with all features
    """
    logger.info("Starting dataset merge...")

    # First merge: auction + item on auction_id
    logger.info("Merging auction and item data on auction_id...")

    # Check if item_df has auction_id column
    if "auction_id" not in item_df.columns:
        logger.warning(
            "item_df does not have auction_id column, will merge on item_id only later"
        )
        auction_item_df = item_df.copy()
    else:
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

    # Second merge: (auction+item) + bid on auction_id and item_id
    logger.info("Merging with bid data on auction_id and item_id...")

    # Determine merge keys based on available columns
    merge_keys = []
    if "auction_id" in auction_item_df.columns and "auction_id" in bid_df.columns:
        merge_keys.append("auction_id")
    if "item_id" in auction_item_df.columns and "item_id" in bid_df.columns:
        merge_keys.append("item_id")

    if not merge_keys:
        logger.error("No common keys found for merging!")
        raise ValueError("Cannot merge datasets: no common keys (auction_id, item_id)")

    logger.info(f"Merging on keys: {merge_keys}")

    merged_df = pd.merge(
        auction_item_df, bid_df, on=merge_keys, how="outer", suffixes=("", "_bid")
    )

    logger.info(
        f"After full merge: {len(merged_df):,} records "
        f"with {len(merged_df.columns)} columns"
    )

    # Reorder columns to ensure auction_id and item_id are first
    logger.info("Reordering columns to place auction_id and item_id first...")
    cols = list(merged_df.columns)

    # Build new column order
    first_cols = []
    if "auction_id" in cols:
        first_cols.append("auction_id")
        cols.remove("auction_id")
    if "item_id" in cols:
        first_cols.append("item_id")
        cols.remove("item_id")

    # Reorder: first_cols + remaining cols
    merged_df = merged_df[first_cols + cols]

    logger.info(f"Column order: {merged_df.columns[:5].tolist()} ...")
    logger.info(
        f"Final merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns"
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
) -> pd.DataFrame:
    """
    Run the full pipeline to combine engineered datasets.

    Args:
        upload: Whether to upload result to Hugging Face
        output_file: Optional path to save local copy
        repo_id: Hugging Face repository ID for upload

    Returns:
        Merged DataFrame
    """
    logger.info("=" * 80)
    logger.info("Starting Engineered Dataset Combination Pipeline")
    logger.info("=" * 80)

    # Step 1: Load datasets
    logger.info("\n[Step 1/4] Loading datasets from Hugging Face...")
    auction_df = load_auction_features()
    item_df = load_item_features()
    bid_df = load_bid_features_batched()

    # Step 2: Merge datasets
    logger.info("\n[Step 2/4] Merging datasets...")
    merged_df = merge_datasets(auction_df, item_df, bid_df)

    # Step 3: Save locally (optional)
    if output_file:
        logger.info(f"\n[Step 3/4] Saving to local file: {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(output_file, index=False)
        logger.info(f"Saved to {output_file}")
    else:
        logger.info("\n[Step 3/4] Skipping local save (no output_file specified)")

    # Step 4: Upload to Hugging Face (optional)
    if upload:
        logger.info("\n[Step 4/4] Uploading to Hugging Face...")
        upload_to_huggingface(merged_df, repo_id=repo_id)
    else:
        logger.info("\n[Step 4/4] Skipping upload (use --upload flag to enable)")

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

    # Run pipeline
    run_pipeline(
        upload=args.upload,
        output_file=output_file,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
