# =============================================================================
# Auction Price Prediction - Item-Level Feature Engineering Pipeline
# =============================================================================
"""
Feature engineering pipeline for item-level data.

This module loads item data from Hugging Face, merges with enriched item data,
performs feature engineering, and uploads the final dataset.

Pipeline steps:
1. Load item data from HuggingFace (jpearce610/item_data)
2. Load enriched item data from HuggingFace (jpearce610/enriched_item_data)
3. Merge item data with enriched data on item_id
4. Feature engineering:
   - Text length features (title, description, enriched fields)
   - Boolean features (brand populated, seriesLine populated)
   - Categorical encoding (condition, working)
   - Count fields from list/JSON variables
   - Within-auction item closing order
   - Log transformation of winning price
5. Keep only specified raw variables and engineered features
6. Ensure all variables have item_ prefix (except auction_id)
7. Upload to Hugging Face as engineered_item_data
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


def load_enriched_item_data() -> pd.DataFrame:
    """
    Load enriched item data from Hugging Face.

    Returns:
        DataFrame with enriched item data
    """
    logger.info("Loading enriched item data from HuggingFace...")
    dataset = load_dataset("jpearce610/enriched_item_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} enriched item records")
    return df


# =============================================================================
# Feature Engineering Functions
# =============================================================================


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

        # Fill missing values
        result[source_col] = result[source_col].fillna("Unknown")

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
    result[end_time_col] = pd.to_datetime(result[end_time_col], errors="coerce")

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

    # Make sure raw cols to keep are included
    for col in raw_cols_to_keep:
        if col in result.columns and col not in final_cols:
            final_cols.append(col)

    # Filter to only columns that exist
    final_cols = [col for col in final_cols if col in result.columns]

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
) -> pd.DataFrame:
    """
    Run the complete item feature engineering pipeline.

    Args:
        upload_to_hf: Whether to upload result to Hugging Face
        hf_repo_id: Hugging Face repository ID for upload
        hf_token: Hugging Face token for authentication

    Returns:
        DataFrame with engineered item features
    """
    logger.info("=" * 60)
    logger.info("Starting Item Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Step 1: Load item data
    item_df = load_item_data()

    # Step 2: Load enriched item data
    enriched_df = load_enriched_item_data()

    # Step 3: Merge item with enriched data
    logger.info("Merging item data with enriched data...")
    df = item_df.merge(enriched_df, on="item_id", how="left")
    logger.info(f"Merged dataset shape: {df.shape}")

    # Step 4: Feature engineering
    logger.info("Running feature engineering...")

    # 4a: Text length features
    df = add_text_length_features(df)

    # 4b: Boolean features (brand, seriesLine)
    df = add_boolean_features(df)

    # 4c: Categorical encoding (condition, working)
    df = encode_categorical_features(df)

    # 4d: List count features
    df = add_list_count_features(df)

    # 4e: Item closing order within auction
    df = add_item_closing_order(df)

    # 4f: Price features (rename + log transform)
    df = add_price_features(df)

    # Step 5: Select final columns
    df = select_final_columns(df)

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Final columns: {df.columns.tolist()}")

    # Step 6: Upload to Hugging Face (optional)
    if upload_to_hf:
        upload_to_huggingface(df, hf_repo_id, hf_token)

    logger.info("=" * 60)
    logger.info("Item Feature Engineering Pipeline Complete")
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


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the item feature engineering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run item feature engineering pipeline"
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

    args = parser.parse_args()

    # Run pipeline
    df = run_item_feature_pipeline(
        upload_to_hf=args.upload,
        hf_repo_id=args.repo_id,
        hf_token=args.token,
    )

    print(f"Pipeline complete. Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
