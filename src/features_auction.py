# =============================================================================
# Auction Price Prediction - Auction-Level Feature Engineering Pipeline
# =============================================================================
"""
Feature engineering pipeline for auction-level data.

This module loads auction data from Hugging Face, merges with enriched data,
performs feature engineering, and uploads the final dataset.

Pipeline steps:
1. Load auction data from HuggingFace (jpearce610/auction_data)
2. Load enriched auction data from HuggingFace (jpearce610/enriched_auction_data)
3. Merge auction data with enriched data on auction_id
4. Merge with enriched postal code datasets from /data/enriched
5. Feature engineering:
   - Auction length in hours
   - Pickup window features
   - Auction partner boolean
   - Normalized auction totals
   - Categorical encoding
   - Geospatial features
6. Keep only specified raw variables and engineered features
7. Ensure all variables have auction_ prefix
8. Upload to Hugging Face as engineered_auction_data
"""


import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger

from src.config import settings
from src.datetime_features import add_auction_duration_features
from src.enriched import (
    extract_fsa,
    load_all_enriched_data,
)
from src.features import extract_pickup_windows

# =============================================================================
# Data Loading
# =============================================================================


def load_auction_data() -> pd.DataFrame:
    """
    Load auction data from Hugging Face.

    Returns:
        DataFrame with auction data
    """
    logger.info("Loading auction data from HuggingFace...")
    dataset = load_dataset("jpearce610/auction_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} auction records")
    return df


def load_enriched_auction_data() -> pd.DataFrame:
    """
    Load enriched auction data from Hugging Face.

    Returns:
        DataFrame with enriched auction data
    """
    logger.info("Loading enriched auction data from HuggingFace...")
    dataset = load_dataset("jpearce610/enriched_auction_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} enriched auction records")
    return df


# =============================================================================
# Feature Engineering Functions
# =============================================================================


def add_auction_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add auction length features using DatetimeFeatureEngineer.

    Creates features for auction duration between starts and ends columns.

    Args:
        df: DataFrame with auction_starts and auction_ends columns

    Returns:
        DataFrame with auction length features added
    """
    logger.info("Adding auction length features...")
    result = df.copy()

    # Ensure datetime columns are parsed (handle mixed timezones by converting to UTC)
    for col in ["auction_starts", "auction_ends", "auction_last_item_closes"]:
        if col in result.columns:
            # Convert to datetime with UTC to handle mixed timezone formats
            result[col] = pd.to_datetime(result[col], utc=True, errors="coerce")
            # Convert to timezone-naive for easier calculation
            result[col] = result[col].dt.tz_localize(None)

    # Add auction duration features using existing function
    if "auction_starts" in result.columns and "auction_ends" in result.columns:
        result = add_auction_duration_features(
            result,
            start_col="auction_starts",
            end_col="auction_ends",
            prefix="auction_length_",
        )

    # Calculate auction length in hours directly
    if "auction_starts" in result.columns and "auction_ends" in result.columns:
        duration = result["auction_ends"] - result["auction_starts"]
        result["auction_length_hours"] = duration.dt.total_seconds() / 3600

    # Calculate actual auction length (using last_item_closes for soft-close)
    if (
        "auction_starts" in result.columns
        and "auction_last_item_closes" in result.columns
    ):
        actual_duration = result["auction_last_item_closes"] - result["auction_starts"]
        result["auction_actual_length_hours"] = (
            actual_duration.dt.total_seconds() / 3600
        )

        # Extension due to soft-close bidding
        if "auction_length_hours" in result.columns:
            result["auction_extension_hours"] = (
                result["auction_actual_length_hours"] - result["auction_length_hours"]
            )

    logger.info("Auction length features added")
    return result


def add_pickup_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pickup window features using extract_pickup_windows.

    Args:
        df: DataFrame with auction_removal_info column

    Returns:
        DataFrame with pickup window features added
    """
    logger.info("Adding pickup window features...")
    result = df.copy()

    if "auction_removal_info" not in result.columns:
        logger.warning("auction_removal_info column not found, skipping pickup features")
        return result

    # Extract pickup window features
    pickup_features = result["auction_removal_info"].apply(extract_pickup_windows)
    pickup_df = pd.DataFrame(pickup_features.tolist())

    # Rename columns with auction_ prefix
    pickup_df = pickup_df.rename(
        columns={
            "num_pickup_windows": "auction_num_pickup_windows",
            "total_pickup_hours": "auction_total_pickup_hours",
            "first_pickup_start_hour": "auction_first_pickup_start_hour",
            "last_pickup_end_hour": "auction_last_pickup_end_hour",
            "pickup_day_of_week": "auction_pickup_day_of_week",
            "has_category_windows": "auction_has_category_pickup_windows",
        }
    )

    # Concatenate with original dataframe
    result = pd.concat([result, pickup_df], axis=1)

    logger.info(f"Added pickup window features: {list(pickup_df.columns)}")
    return result


def add_auction_partner_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add auction partner boolean feature.

    Args:
        df: DataFrame with auction_partner_url column

    Returns:
        DataFrame with auction_has_partner feature added
    """
    logger.info("Adding auction partner feature...")
    result = df.copy()

    if "auction_partner_url" in result.columns:
        # Create boolean: True if partner URL is non-empty, False otherwise
        result["auction_has_partner"] = (
            result["auction_partner_url"].notna()
            & (result["auction_partner_url"] != "")
            & (result["auction_partner_url"] != "nan")
        ).astype(int)
    else:
        logger.warning("auction_partner_url column not found")
        result["auction_has_partner"] = 0

    logger.info("Auction partner feature added")
    return result


def add_normalized_auction_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add auction total variables normalized by auction_item_count.

    Args:
        df: DataFrame with auction_total_* columns and auction_item_count

    Returns:
        DataFrame with normalized features added
    """
    logger.info("Adding normalized auction totals...")
    result = df.copy()

    if "auction_item_count" not in result.columns:
        logger.warning("auction_item_count not found, skipping normalization")
        return result

    # Avoid division by zero
    item_count = result["auction_item_count"].replace(0, np.nan)

    # Normalize auction totals
    if "auction_total_viewed" in result.columns:
        result["auction_avg_views_per_item"] = (
            result["auction_total_viewed"] / item_count
        )

    if "auction_total_winning_price" in result.columns:
        result["auction_avg_price_per_item"] = (
            result["auction_total_winning_price"] / item_count
        )

    if "auction_total_bid_count" in result.columns:
        result["auction_avg_bids_per_item"] = (
            result["auction_total_bid_count"] / item_count
        )

    if "auction_total_images" in result.columns:
        result["auction_avg_images_per_item"] = (
            result["auction_total_images"] / item_count
        )

    logger.info("Normalized auction totals added")
    return result


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geospatial features from postal code and lat/lng.

    Features:
    - FSA (Forward Sortation Area - first 3 chars of postal code)
    - First character of postal code (postal zone)
    - Latitude and longitude based features

    Args:
        df: DataFrame with postal code and lat/lng columns

    Returns:
        DataFrame with geospatial features added
    """
    logger.info("Adding geospatial features...")
    result = df.copy()

    postal_col = "enriched_auction_approxLocation_postalCode"
    lat_col = "enriched_auction_approxLocation_lat"
    lng_col = "enriched_auction_approxLocation_lng"

    # Extract FSA from postal code
    if postal_col in result.columns:
        result["auction_fsa"] = result[postal_col].apply(extract_fsa)

        # Extract first character (postal zone)
        result["auction_postal_zone"] = result["auction_fsa"].str[0].fillna("Unknown")

    # Latitude/Longitude features
    if lat_col in result.columns and lng_col in result.columns:
        # Copy lat/lng with auction prefix
        result["auction_latitude"] = result[lat_col]
        result["auction_longitude"] = result[lng_col]

        # Distance from reference points (major city centers)
        # Toronto coordinates: 43.6532, -79.3832
        toronto_lat, toronto_lng = 43.6532, -79.3832
        result["auction_distance_from_toronto"] = np.sqrt(
            (result[lat_col] - toronto_lat) ** 2
            + (result[lng_col] - toronto_lng) ** 2
        )

        # North/South indicator (relative to Canadian average ~45 degrees)
        result["auction_is_northern"] = (result[lat_col] > 49).astype(int)

        # Coastal proximity (very rough approximation based on longitude)
        # West coast: lng < -120, East coast: lng > -65
        result["auction_is_west_coast"] = (result[lng_col] < -120).astype(int)
        result["auction_is_east_coast"] = (result[lng_col] > -65).astype(int)

    logger.info("Geospatial features added")
    return result


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using appropriate methods.

    For high-cardinality variables, use frequency encoding.
    For low-cardinality variables, use one-hot encoding.

    Args:
        df: DataFrame with categorical columns

    Returns:
        DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features...")
    result = df.copy()

    # Define categorical columns to encode
    categorical_cols = {
        "enriched_auction_category": "auction_category",
        "enriched_auction_type": "auction_type",
        "enriched_auction_displayRegion": "auction_region",
        "enriched_auction_approxLocation_city": "auction_city",
        "enriched_auction_approxLocation_countryCode": "auction_country",
        "enriched_auction_approxLocation_regionCode": "auction_province",
    }

    for old_col, new_prefix in categorical_cols.items():
        if old_col not in result.columns:
            continue

        # Fill missing values
        result[old_col] = result[old_col].fillna("Unknown")

        # Get value counts for frequency encoding
        value_counts = result[old_col].value_counts()
        n_unique = len(value_counts)

        logger.info(f"Encoding {old_col}: {n_unique} unique values")

        if n_unique <= 20:
            # One-hot encoding for low-cardinality
            dummies = pd.get_dummies(
                result[old_col], prefix=new_prefix, dummy_na=False
            )
            # Clean column names (remove spaces, special chars)
            dummies.columns = [
                col.replace(" ", "_").replace("/", "_").replace("-", "_")
                for col in dummies.columns
            ]
            result = pd.concat([result, dummies], axis=1)
        else:
            # Frequency encoding for high-cardinality
            freq_map = value_counts / len(result)
            result[f"{new_prefix}_frequency"] = result[old_col].map(freq_map)

            # Also add count encoding
            result[f"{new_prefix}_count"] = result[old_col].map(value_counts)

    logger.info("Categorical encoding complete")
    return result


def merge_with_postal_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge auction data with enriched postal code datasets.

    Args:
        df: DataFrame with auction_fsa column

    Returns:
        DataFrame merged with postal enrichment data
    """
    logger.info("Merging with postal code enrichment data...")
    result = df.copy()

    if "auction_fsa" not in result.columns:
        logger.warning("auction_fsa not found, skipping postal enrichment")
        return result

    try:
        # Load all enriched data (population + tax stats)
        enriched_data = load_all_enriched_data()

        # Merge on FSA
        result = result.merge(
            enriched_data,
            left_on="auction_fsa",
            right_index=True,
            how="left",
        )

        # Rename columns with auction_ prefix
        rename_map = {
            "Population, 2021": "auction_area_population",
            "Total private dwellings, 2021": "auction_area_total_dwellings",
            "Private dwellings occupied by usual residents, 2021": "auction_area_occupied_dwellings",
            "Number of Returns": "auction_area_tax_returns",
            "Total Income": "auction_area_total_income",
            "Average Total Income": "auction_area_avg_income",
            "Median Total Income": "auction_area_median_income",
        }
        result = result.rename(columns=rename_map)

        # Compute derived enriched features
        result = compute_postal_enrichment_features(result)

        logger.info("Postal enrichment merge complete")

    except FileNotFoundError as e:
        logger.warning(f"Could not load enriched data: {e}")

    return result


def compute_postal_enrichment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from postal enrichment data.

    Args:
        df: DataFrame with postal enrichment columns

    Returns:
        DataFrame with derived features
    """
    result = df.copy()

    # Persons per dwelling (density proxy)
    if (
        "auction_area_population" in result.columns
        and "auction_area_occupied_dwellings" in result.columns
    ):
        result["auction_area_persons_per_dwelling"] = (
            result["auction_area_population"]
            / result["auction_area_occupied_dwellings"]
        )

    # Occupancy rate
    if (
        "auction_area_occupied_dwellings" in result.columns
        and "auction_area_total_dwellings" in result.columns
    ):
        result["auction_area_occupancy_rate"] = (
            result["auction_area_occupied_dwellings"]
            / result["auction_area_total_dwellings"]
        )

    # Tax filing rate
    if (
        "auction_area_tax_returns" in result.columns
        and "auction_area_population" in result.columns
    ):
        result["auction_area_tax_filing_rate"] = (
            result["auction_area_tax_returns"] / result["auction_area_population"]
        )

    return result


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename final columns for output dataset.

    Keeps only:
    - Raw variables: auction_starts, auction_ends, auction_last_item_closes,
      auction_item_count, auction_id
    - All engineered features with auction_ prefix (excluding raw text columns)

    Args:
        df: DataFrame with all features

    Returns:
        DataFrame with only selected columns
    """
    logger.info("Selecting final columns...")
    result = df.copy()

    # Raw variables to keep (as specified in requirements)
    raw_cols_to_keep = [
        "auction_id",
        "auction_starts",
        "auction_ends",
        "auction_last_item_closes",
        "auction_item_count",
    ]

    # Columns to explicitly drop (raw text/HTML that shouldn't be in final dataset)
    cols_to_drop = [
        "auction_title",
        "auction_intro",
        "auction_removal_info",
        "auction_partner_url",
        "auction_pickup_time",
        "auction_fsa",  # Drop raw FSA, keep only FSA-derived features (postal_zone)
    ]

    # Get all columns that start with auction_
    all_auction_cols = [col for col in result.columns if col.startswith("auction_")]

    # Remove columns that should be dropped
    final_cols = [col for col in all_auction_cols if col not in cols_to_drop]

    # Make sure raw cols to keep are included
    for col in raw_cols_to_keep:
        if col in result.columns and col not in final_cols:
            final_cols.append(col)

    # Filter to only columns that exist
    final_cols = [col for col in final_cols if col in result.columns]

    # Sort columns alphabetically, but put auction_id first
    final_cols = sorted(set(final_cols))
    if "auction_id" in final_cols:
        final_cols.remove("auction_id")
        final_cols = ["auction_id"] + final_cols

    result = result[final_cols]

    logger.info(f"Selected {len(final_cols)} columns for final dataset")
    return result


# =============================================================================
# Main Pipeline
# =============================================================================


def run_auction_feature_pipeline(
    upload_to_hf: bool = False,
    hf_repo_id: str = "engineered_auction_data",
    hf_token: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete auction feature engineering pipeline.

    Args:
        upload_to_hf: Whether to upload result to Hugging Face
        hf_repo_id: Hugging Face repository ID for upload
        hf_token: Hugging Face token for authentication

    Returns:
        DataFrame with engineered auction features
    """
    logger.info("=" * 60)
    logger.info("Starting Auction Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Step 1: Load auction data
    auction_df = load_auction_data()

    # Step 2: Load enriched auction data
    enriched_df = load_enriched_auction_data()

    # Step 3: Merge auction with enriched data
    logger.info("Merging auction data with enriched data...")
    df = auction_df.merge(enriched_df, on="auction_id", how="left")
    logger.info(f"Merged dataset shape: {df.shape}")

    # Step 4: Feature engineering
    logger.info("Running feature engineering...")

    # 4a: Auction length features
    df = add_auction_length_features(df)

    # 4b: Pickup window features
    df = add_pickup_window_features(df)

    # 4c: Auction partner boolean
    df = add_auction_partner_feature(df)

    # 4d: Normalized auction totals
    df = add_normalized_auction_totals(df)

    # 4e: Geospatial features
    df = add_geospatial_features(df)

    # 4f: Merge with postal enrichment data
    df = merge_with_postal_enrichment(df)

    # 4g: Categorical encoding
    df = encode_categorical_features(df)

    # Step 5: Select final columns (keep raw + engineered, drop others)
    df = select_final_columns(df)

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Final columns: {df.columns.tolist()}")

    # Step 6: Upload to Hugging Face (optional)
    if upload_to_hf:
        upload_to_huggingface(df, hf_repo_id, hf_token)

    logger.info("=" * 60)
    logger.info("Auction Feature Engineering Pipeline Complete")
    logger.info("=" * 60)

    return df


def upload_to_huggingface(
    df: pd.DataFrame,
    repo_id: str = "engineered_auction_data",
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
    """Main entry point for the auction feature engineering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run auction feature engineering pipeline"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload result to Hugging Face",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="engineered_auction_data",
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
    df = run_auction_feature_pipeline(
        upload_to_hf=args.upload,
        hf_repo_id=args.repo_id,
        hf_token=args.token,
    )

    print(f"Pipeline complete. Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
