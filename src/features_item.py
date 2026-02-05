# =============================================================================
# Auction Price Prediction - Item-Level Feature Engineering Pipeline
# =============================================================================
"""
Feature engineering pipeline for item-level data with true streaming batch processing.

This module loads item data from Hugging Face, processes enriched item data from
local Parquet file one batch at a time with NO memory accumulation.

Pipeline steps (TRUE STREAMING - no accumulation):
1. Load item data from HuggingFace (jpearce610/item_data) - loaded once
2. Stream enriched item data from local Parquet file in batches (default 100k rows per batch)
3. For each batch:
   a. Load batch from Parquet file using pyarrow
   b. Merge with item data on item_id
   c. Feature engineering:
      - Text length features (title, description, enriched fields)
      - Boolean features (brand populated, seriesLine populated)
      - Categorical encoding (condition, working)
      - Count fields from list/JSON variables
      - Within-auction item closing order
      - Log transformation of winning price
   d. Select final columns with item_ prefix
   e. Save to disk (Parquet)
   f. CLEAR from memory
4. Load all batches from disk and concatenate
5. Save final dataset to data/processed/items/engineered_item_data.parquet
6. Upload to Hugging Face as single dataset (optional)

Benefits of streaming batch processing:
- TRUE memory efficiency: Only ONE batch in memory at a time
- NO accumulation: Process → Save → Clear → Repeat
- Disk-backed: Batches saved as Parquet files during processing
- Configurable: Adjust batch_size based on available memory
- Predictable memory usage: Independent of dataset size
- Efficient local file reading with pyarrow (no HF download)
- Single full upload to HF (more reliable than batch uploads)
"""


import json

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger

from src.config import settings

# =============================================================================
# Data Loading
# =============================================================================


def load_item_data() -> pd.DataFrame:
    """
    Load item data from Hugging Face.

    Returns:
        DataFrame with item data
    """
    logger.info("Loading item data from HuggingFace...")
    dataset = load_dataset("jpearce610/item_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} item records")
    return df


def load_enriched_item_data_batched(batch_size: int = 100_000, parquet_path: str | None = None):
    """
    Load enriched item data from local Parquet file in batches (generator).
    
    Yields batches of data to process incrementally, reducing memory usage.
    Uses pyarrow for memory-efficient reading without loading full file.

    Args:
        batch_size: Number of rows per batch (default 100k)
        parquet_path: Path to local parquet file (default: data/processed/items/enriched_item_data.parquet)

    Yields:
        DataFrame batches of enriched item data
    """
    import psutil
    import os
    import gc
    import pyarrow.parquet as pq
    from pathlib import Path
    
    def log_memory():
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        logger.info(f"Memory usage: {mem_mb:.1f} MB")
    
    # Default to local parquet file
    if parquet_path is None:
        parquet_path = Path("data/processed/items/enriched_item_data.parquet")
    else:
        parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    logger.info(f"Loading enriched item data from local Parquet file: {parquet_path}")
    logger.info(f"Batch size: {batch_size:,} rows")
    log_memory()
    
    try:
        # Get total row count from metadata
        metadata = pq.read_metadata(parquet_path)
        total_rows = metadata.num_rows
        logger.info(f"Total rows in file: {total_rows:,}")
        
        # Open parquet file for batch reading
        parquet_file = pq.ParquetFile(parquet_path)
        logger.info(f"Parquet file opened successfully")
        logger.info(f"Number of row groups: {parquet_file.num_row_groups}")
        log_memory()
        
    except Exception as e:
        logger.error(f"Failed to open parquet file: {e}")
        raise
    
    batch_num = 0
    rows_processed = 0
    
    try:
        logger.info("Starting to read batches from parquet file...")
        
        # Read in batches
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_num += 1
            
            # Convert arrow batch to pandas DataFrame
            batch_df = batch.to_pandas()
            rows_processed += len(batch_df)
            
            logger.info(f"\nBatch #{batch_num}: {len(batch_df):,} rows (total: {rows_processed:,}/{total_rows:,})")
            log_memory()
            
            # Yield the batch
            yield batch_df
            
            # CRITICAL: Clear memory after consumer processes the batch
            del batch_df
            del batch
            gc.collect()
        
        logger.info(f"\n✓ Finished loading all batches")
        logger.info(f"Total: {rows_processed:,} rows in {batch_num} batches")
        log_memory()
        
    except Exception as e:
        logger.error(f"\n❌ Error during data loading: {e}")
        logger.error(f"Processed {rows_processed} rows in {batch_num} batches before error")
        log_memory()
        raise


# =============================================================================
# Feature Engineering Functions
# =============================================================================


def process_batch_features(batch_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process feature engineering for a single batch of enriched data.
    
    This function performs all feature engineering steps on a batch:
    1. Merge with item data
    2. Add all engineered features
    3. Select final columns
    
    Args:
        batch_df: Batch of enriched item data
        item_df: Full item dataset (for merging)
    
    Returns:
        DataFrame with engineered features for this batch
    """
    # Merge with item data on both item_id and auction_id to avoid duplicate columns
    df = item_df.merge(batch_df, on=["item_id", "auction_id"], how="inner")
    logger.debug(f"Batch shape after merge: {df.shape}")
    
    # Feature engineering steps
    df = add_text_length_features(df)
    df = add_boolean_features(df)
    df = encode_categorical_features(df)
    df = add_list_count_features(df)
    df = add_item_closing_order(df)
    df = add_price_features(df)
    df = select_final_columns(df)
    
    logger.debug(f"Batch shape after feature engineering: {df.shape}")
    return df


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add text length features for various text fields.

    Creates length features for:
    - item_title
    - item_description
    - enriched_item_title
    - enriched_item_description
    - enriched_item_qualitativeDescription

    Args:
        df: DataFrame with text columns

    Returns:
        DataFrame with text length features added
    """
    logger.info("Adding text length features...")
    result = df.copy()

    # Text columns to compute length for
    text_columns = {
        "item_title": "item_title_length",
        "item_description": "item_description_length",
        "enriched_item_title": "item_enriched_title_length",
        "enriched_item_description": "item_enriched_description_length",
        "enriched_item_qualitativeDescription": "item_enriched_qualitative_description_length",
    }

    for source_col, target_col in text_columns.items():
        if source_col in result.columns:
            # Handle None/NaN values and compute length
            result[target_col] = (
                result[source_col]
                .fillna("")
                .astype(str)
                .str.len()
            )
            logger.debug(f"Added {target_col} from {source_col}")
        else:
            logger.warning(f"Column {source_col} not found, setting {target_col} to 0")
            result[target_col] = 0

    logger.info("Text length features added")
    return result


def add_boolean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean features for whether certain fields are populated.

    Creates boolean features for:
    - enriched_item_brand is populated
    - enriched_item_seriesLine is populated

    Args:
        df: DataFrame with enriched columns

    Returns:
        DataFrame with boolean features added
    """
    logger.info("Adding boolean features...")
    result = df.copy()

    # Boolean features to create
    boolean_columns = {
        "enriched_item_brand": "item_has_brand",
        "enriched_item_seriesLine": "item_has_series_line",
    }

    for source_col, target_col in boolean_columns.items():
        if source_col in result.columns:
            # Check if populated (not null, not empty string, not 'nan')
            result[target_col] = (
                result[source_col].notna()
                & (result[source_col] != "")
                & (result[source_col].astype(str) != "nan")
                & (result[source_col].astype(str) != "None")
            ).astype(int)
            logger.debug(f"Added {target_col} from {source_col}")
        else:
            logger.warning(f"Column {source_col} not found, setting {target_col} to 0")
            result[target_col] = 0

    logger.info("Boolean features added")
    return result


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables: enriched_item_condition, enriched_item_working.

    Uses one-hot encoding for these low-cardinality categorical variables.

    Args:
        df: DataFrame with categorical columns

    Returns:
        DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features...")
    result = df.copy()

    categorical_columns = {
        "enriched_item_condition": "item_condition",
        "enriched_item_working": "item_working",
    }

    for source_col, prefix in categorical_columns.items():
        if source_col not in result.columns:
            logger.warning(f"Column {source_col} not found, skipping encoding")
            continue

        # Fill missing values and empty strings with "Unknown"
        result[source_col] = result[source_col].fillna("Unknown")
        result[source_col] = result[source_col].replace("", "Unknown")
        result[source_col] = result[source_col].replace("unknown", "Unknown")

        # Get value counts
        value_counts = result[source_col].value_counts()
        n_unique = len(value_counts)
        logger.info(f"Encoding {source_col}: {n_unique} unique values")

        # One-hot encoding
        dummies = pd.get_dummies(
            result[source_col], prefix=prefix, dummy_na=False
        )
        # Clean column names (remove spaces, special chars)
        dummies.columns = [
            col.replace(" ", "_").replace("/", "_").replace("-", "_").lower()
            for col in dummies.columns
        ]
        result = pd.concat([result, dummies], axis=1)

    logger.info("Categorical encoding complete")
    return result


def _safe_parse_json_or_list(value: str | list | dict | None, count_dict_keys: bool = False) -> list:
    """
    Safely parse a JSON string or return a list.

    Args:
        value: JSON string, list, dict, or None
        count_dict_keys: If True and value is a dict, return list of keys
                         (useful for counting attributes). If False, wrap
                         dict in a list.

    Returns:
        List of items, or empty list if parsing fails
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, dict):
        return list(value.keys()) if count_dict_keys else [value]
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "" or value == "nan" or value == "None":
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.keys()) if count_dict_keys else [parsed]
            return [parsed] if parsed else []
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, treat as single item if non-empty
            return [value] if value.strip() else []
    return []


def add_list_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add count features for list/JSON variables.

    Counts the number of items in:
    - enriched_item_brands (list of brands)
    - enriched_item_categories (list of categories)
    - enriched_item_items (list of items)
    - enriched_item_attributes (dict of attributes - counts keys)

    Args:
        df: DataFrame with list/JSON columns

    Returns:
        DataFrame with count features added
    """
    logger.info("Adding list count features...")
    result = df.copy()

    # List columns (parsed as lists)
    list_columns = {
        "enriched_item_brands": "item_brands_count",
        "enriched_item_categories": "item_categories_count",
        "enriched_item_items": "item_items_count",
    }

    for source_col, target_col in list_columns.items():
        if source_col in result.columns:
            result[target_col] = (
                result[source_col]
                .apply(_safe_parse_json_or_list)
                .apply(len)
            )
            logger.debug(f"Added {target_col} from {source_col}")
        else:
            logger.warning(f"Column {source_col} not found, setting {target_col} to 0")
            result[target_col] = 0

    # Attributes column (dict - count keys)
    if "enriched_item_attributes" in result.columns:
        result["item_attributes_count"] = (
            result["enriched_item_attributes"]
            .apply(lambda x: _safe_parse_json_or_list(x, count_dict_keys=True))
            .apply(len)
        )
        logger.debug("Added item_attributes_count from enriched_item_attributes")
    else:
        logger.warning("Column enriched_item_attributes not found, setting item_attributes_count to 0")
        result["item_attributes_count"] = 0

    logger.info("List count features added")
    return result


def add_item_closing_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-auction item closing order feature.

    Creates a rank from 1 to N indicating the order in which items ended
    within each auction (1 = first to close, N = last to close).

    Args:
        df: DataFrame with auction_id and item_end_time columns

    Returns:
        DataFrame with item_closing_order feature added
    """
    logger.info("Adding item closing order feature...")
    result = df.copy()

    end_time_col = None
    for col in ["item_end_time", "item_closes"]:
        if col in result.columns:
            end_time_col = col
            break

    if end_time_col is None:
        logger.warning("No end time column found, setting item_closing_order to 0")
        result["item_closing_order"] = 0
        return result

    if "auction_id" not in result.columns:
        logger.warning("auction_id column not found, setting item_closing_order to 0")
        result["item_closing_order"] = 0
        return result

    # Convert end_time to datetime if needed
    result[end_time_col] = pd.to_datetime(result[end_time_col], errors="coerce", utc=True)

    # Rank items within each auction by end time
    # Items with the same end time get the same rank (method='dense')
    result["item_closing_order"] = (
        result.groupby("auction_id")[end_time_col]
        .rank(method="dense", ascending=True)
        .fillna(0)
        .astype(int)
    )

    logger.info("Item closing order feature added")
    return result


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-related features.

    - Renames item_current_bid to item_winning_price_raw
    - Creates item_winning_price_log1p using log(x + 1) transformation

    Args:
        df: DataFrame with price columns

    Returns:
        DataFrame with price features added
    """
    logger.info("Adding price features...")
    result = df.copy()

    # Find the current bid column
    current_bid_col = None
    for col in ["item_current_bid", "item_currentBid"]:
        if col in result.columns:
            current_bid_col = col
            break

    if current_bid_col is not None:
        # Rename to item_winning_price_raw
        result["item_winning_price_raw"] = result[current_bid_col]

        # Apply log(x + 1) transformation
        # Handle negative values by treating them as 0
        prices = result["item_winning_price_raw"].fillna(0)
        prices = prices.clip(lower=0)  # Ensure non-negative
        result["item_winning_price_log1p"] = np.log1p(prices)

        logger.debug(f"Created item_winning_price_raw from {current_bid_col}")
        logger.debug("Created item_winning_price_log1p")
    else:
        logger.warning("No current bid column found")
        result["item_winning_price_raw"] = 0
        result["item_winning_price_log1p"] = 0

    logger.info("Price features added")
    return result


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename final columns for output dataset.

    Keeps:
    - Raw variables: item_winning_price_raw, item_bidding_extended,
      item_number_of_images, item_end_time, item_id, auction_id
    - All engineered features with item_ prefix

    Ensures all columns (except auction_id) have item_ prefix.

    Args:
        df: DataFrame with all features

    Returns:
        DataFrame with only selected columns
    """
    logger.info("Selecting final columns...")
    result = df.copy()

    # Raw variables to keep
    raw_cols_to_keep = [
        "item_id",
        "auction_id",
        "item_winning_price_raw",
        "item_number_of_images",
        "item_end_time",
    ]

    # Handle bidding_extended column (might have different names)
    bidding_extended_col = None
    for col in ["item_bidding_extended", "item_biddingExtended"]:
        if col in result.columns:
            bidding_extended_col = col
            break

    if bidding_extended_col is not None:
        result["item_bidding_extended"] = result[bidding_extended_col]
        raw_cols_to_keep.append("item_bidding_extended")

    # Handle number_of_images column (might have different names)
    images_col = None
    for col in ["item_number_of_images", "item_numberOfImages", "item_numImages"]:
        if col in result.columns:
            images_col = col
            break

    if images_col is not None and images_col != "item_number_of_images":
        result["item_number_of_images"] = result[images_col]

    # Handle end_time column (might have different names)
    end_time_col = None
    for col in ["item_end_time", "item_closes", "item_endTime"]:
        if col in result.columns:
            end_time_col = col
            break

    if end_time_col is not None and end_time_col != "item_end_time":
        result["item_end_time"] = result[end_time_col]

    # Columns to explicitly drop (raw text that shouldn't be in final dataset)
    cols_to_drop = [
        # Original text columns
        "item_title",
        "item_description",
        "item_name",
        # Enriched raw text columns
        "enriched_item_title",
        "enriched_item_description",
        "enriched_item_qualitativeDescription",
        "enriched_item_brand",
        "enriched_item_seriesLine",
        "enriched_item_brands",
        "enriched_item_categories",
        "enriched_item_items",
        "enriched_item_attributes",
        "enriched_item_condition",
        "enriched_item_working",
        # Other columns we don't want
        "item_currentBid",
        "item_current_bid",
        "item_biddingExtended",
        "item_closes",
        "item_numberOfImages",
        "item_numImages",
    ]

    # Get all columns that start with item_ or are auction_id
    all_item_cols = [
        col for col in result.columns
        if col.startswith("item_") or col == "auction_id"
    ]

    # Remove columns that should be dropped
    final_cols = [col for col in all_item_cols if col not in cols_to_drop]
    
    # Remove columns that are exactly "item_condition_" or "item_working_"
    # (these come from empty string values in categorical encoding)
    invalid_cols = {"item_condition_", "item_working_"}
    final_cols = [col for col in final_cols if col not in invalid_cols]

    # Make sure raw cols to keep are included
    for col in raw_cols_to_keep:
        if col in result.columns and col not in final_cols:
            final_cols.append(col)

    # Filter to only columns that exist
    final_cols = [col for col in final_cols if col in result.columns]

    # Ensure auction_id and item_id are in the final columns
    if "auction_id" not in final_cols and "auction_id" in result.columns:
        final_cols.append("auction_id")
    if "item_id" not in final_cols and "item_id" in result.columns:
        final_cols.append("item_id")
    
    # Sort columns alphabetically, but put item_id and auction_id first
    final_cols = sorted(set(final_cols))
    priority_cols = []
    if "item_id" in final_cols:
        final_cols.remove("item_id")
        priority_cols.append("item_id")
    if "auction_id" in final_cols:
        final_cols.remove("auction_id")
        priority_cols.append("auction_id")

    final_cols = priority_cols + final_cols

    result = result[final_cols]

    logger.info(f"Selected {len(final_cols)} columns for final dataset")
    return result


# =============================================================================
# Main Pipeline
# =============================================================================


def run_item_feature_pipeline(
    upload_to_hf: bool = False,
    hf_repo_id: str = "engineered_item_data",
    hf_token: str | None = None,
    batch_size: int = 100_000,
    temp_dir: str | None = None,
    enriched_parquet_path: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete item feature engineering pipeline with batch processing.
    
    Processes enriched data one batch at a time with NO memory accumulation:
    1. Load one batch from local Parquet file
    2. Merge with item data
    3. Feature engineer
    4. Save to disk (Parquet)
    5. Clear batch from memory
    6. Repeat for next batch
    7. Load all batches from disk and concatenate
    8. Save final dataset to data/processed/items/engineered_item_data.parquet
    9. Upload to HuggingFace as single full dataset (optional)

    Args:
        upload_to_hf: Whether to upload result to Hugging Face
        hf_repo_id: Hugging Face repository ID for upload
        hf_token: Hugging Face token for authentication
        batch_size: Number of rows to process per batch (default 100k)
        temp_dir: Directory for temporary batch files (default: data/interim/feature_batches)
        enriched_parquet_path: Path to enriched item parquet file (default: data/processed/items/enriched_item_data.parquet)

    Returns:
        DataFrame with engineered item features
    """
    from pathlib import Path
    import tempfile
    import shutil
    import gc
    
    logger.info("=" * 60)
    logger.info("Starting Item Feature Engineering Pipeline (Streaming Batch Mode)")
    logger.info("=" * 60)

    # Setup temporary directory for batch files
    if temp_dir is None:
        temp_dir = Path("data/interim/feature_batches")
    else:
        temp_dir = Path(temp_dir)
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary batch directory: {temp_dir}")
    
    # Clean up any existing batch files
    for f in temp_dir.glob("batch_*.parquet"):
        f.unlink()
        logger.debug(f"Removed old batch file: {f.name}")

    # Step 1: Load item data (smaller dataset, can load fully)
    item_df = load_item_data()
    logger.info(f"Loaded item data: {len(item_df):,} records")

    # Step 2: Process enriched data in batches (NO ACCUMULATION)
    logger.info("Processing enriched data in batches...")
    logger.info(f"Batch size: {batch_size:,} rows")
    logger.info("Memory strategy: Process → Save → Clear → Repeat")
    
    batch_files = []
    batch_count = 0
    total_processed = 0
    
    for batch_num, enriched_batch in enumerate(load_enriched_item_data_batched(batch_size, enriched_parquet_path), 1):
        logger.info(f"\n--- Processing Batch #{batch_num} ---")
        logger.info(f"Batch shape: {enriched_batch.shape}")
        
        # Process this batch through the feature engineering pipeline
        engineered_batch = process_batch_features(enriched_batch, item_df)
        
        # Save to disk immediately (Parquet format for efficiency)
        batch_file = temp_dir / f"batch_{batch_num:04d}.parquet"
        engineered_batch.to_parquet(batch_file, index=False)
        batch_files.append(batch_file)
        
        batch_count += 1
        total_processed += len(engineered_batch)
        
        logger.info(f"Batch #{batch_num} processed: {len(engineered_batch):,} rows")
        logger.info(f"Saved to: {batch_file.name}")
        logger.info(f"Total processed so far: {total_processed:,} rows")
        
        # CRITICAL: Clear both batches from memory immediately
        del enriched_batch
        del engineered_batch
        gc.collect()  # Force garbage collection
        
    logger.info(f"\n{'='*60}")
    logger.info(f"All {batch_count} batches processed and saved to disk")
    logger.info(f"Total rows: {total_processed:,}")
    logger.info(f"{'='*60}\n")
    
    # Step 3: Load all batches from disk and concatenate
    logger.info(f"Loading {len(batch_files)} batches from disk...")
    df = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")
    
    # Step 4: Save final dataset to repo
    output_path = Path("data/processed/items/engineered_item_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving final dataset to: {output_path}")
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Dataset saved to {output_path}")
    
    # Step 5: Upload to Hugging Face if requested
    if upload_to_hf:
        logger.info("Uploading full dataset to Hugging Face...")
        upload_to_huggingface(df, hf_repo_id, hf_token)
    
    # Clean up temporary batch files
    logger.info("Cleaning up temporary batch files...")
    for batch_file in batch_files:
        batch_file.unlink()
    logger.info("✓ Temporary files cleaned up")
    
    logger.info("=" * 60)
    logger.info("Item Feature Engineering Pipeline Complete")
    logger.info(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 60)

    return df


def upload_to_huggingface(
    df: pd.DataFrame,
    repo_id: str = "engineered_item_data",
    token: str | None = None,
) -> None:
    """
    Upload engineered dataset to Hugging Face.

    Args:
        df: DataFrame to upload
        repo_id: Repository ID (user/repo format or just repo name)
        token: HuggingFace API token
    """
    logger.info(f"Uploading dataset to Hugging Face: {repo_id}")

    # Use token from settings if not provided
    if token is None:
        token = settings.huggingface.token

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Push to Hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=False,
    )

    logger.info(f"Dataset uploaded successfully to {repo_id}")


def upload_batches_to_huggingface(
    batch_files: list,
    repo_id: str = "engineered_item_data",
    token: str | None = None,
) -> None:
    """
    DEPRECATED: Use upload_to_huggingface() instead for single full upload.
    
    Upload processed batches to Hugging Face incrementally.
    
    This function is deprecated because batch-by-batch uploading is complex,
    error-prone, and requires downloading the dataset between uploads.
    Use upload_to_huggingface() for a simpler, more reliable single upload.

    Args:
        batch_files: List of Parquet file paths containing processed batches
        repo_id: Repository ID (user/repo format or just repo name)
        token: HuggingFace API token
    """
    logger.warning(
        "upload_batches_to_huggingface() is deprecated. "
        "Use upload_to_huggingface() for single full upload instead."
    )
    
    # Load all batches and do a single upload
    logger.info(f"Loading {len(batch_files)} batches from disk...")
    df = pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)
    logger.info(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    
    # Single upload
    upload_to_huggingface(df, repo_id, token)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the item feature engineering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run item feature engineering pipeline with batch processing"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload result to Hugging Face",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="engineered_item_data",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of rows to process per batch (default: 100,000)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for batch files (default: data/interim/feature_batches)",
    )
    parser.add_argument(
        "--enriched-parquet",
        type=str,
        default=None,
        help="Path to enriched item parquet file (default: data/processed/items/enriched_item_data.parquet)",
    )

    args = parser.parse_args()

    # Run pipeline
    df = run_item_feature_pipeline(
        upload_to_hf=args.upload,
        hf_repo_id=args.repo_id,
        hf_token=args.token,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
        enriched_parquet_path=args.enriched_parquet,
    )

    print(f"Pipeline complete. Dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")


if __name__ == "__main__":
    main()
