# =============================================================================
# Auction Price Prediction - Sequential Bid Feature Engineering
# =============================================================================
"""
Feature engineering pipeline for sequential auction bid data.

This module generates features for each bid within an auction item, supporting
accurate prediction of final prices. Features are computed causally - for any
bid, only that bid and all previous bids are used (no leakage from future bids).

Key features:
1. Time-series features: time_since_first_bid, time_since_last_bid, elapsed_time_fraction
2. Bid count/intensity: rolling counts, bid velocity, bid acceleration
3. Bid amount features: increments, percentiles, deviation from stats
4. Auction progress features: bid ratios, price trajectory

Dataset source: jpearce610/bid_data on Hugging Face
Key columns: auction_id, item_id, bid_time, bid_number, bid_amount, is_proxy_bid

Leakage Prevention:
- All features use only data available up to and including the current bid
- No future-aware statistics or aggregate information
- Each bid row is treated as a prediction point in time
"""


import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger

from src.config import settings

# =============================================================================
# MaxSold Bid Increment Rules
# =============================================================================

# MaxSold bid increment rules (approximate based on common patterns)
# These define the minimum increment required based on current bid amount
MAXSOLD_BID_INCREMENTS = [
    (0, 25, 1),        # $0-$25: $1 increments
    (25, 100, 5),      # $25-$100: $5 increments
    (100, 500, 10),    # $100-$500: $10 increments
    (500, 1000, 25),   # $500-$1000: $25 increments
    (1000, 5000, 50),  # $1000-$5000: $50 increments
    (5000, float('inf'), 100),  # $5000+: $100 increments
]


def get_expected_increment(current_bid: float) -> float:
    """
    Get the expected minimum bid increment based on current bid amount.

    Based on MaxSold bid increment rules:
    https://support.maxsold.com/hc/en-us/articles/203144054-How-do-bid-increments-work

    Args:
        current_bid: The current highest bid amount

    Returns:
        Expected minimum increment for the next bid
    """
    for low, high, increment in MAXSOLD_BID_INCREMENTS:
        if low <= current_bid < high:
            return increment
    return 100  # Default for very high bids


def is_unusual_increment(previous_bid: float, current_bid: float) -> bool:
    """
    Check if a bid increment deviates from MaxSold increment rules.

    An increment is considered unusual if:
    - It's less than the expected minimum increment
    - It's more than 3x the expected increment (aggressive bidding)

    Args:
        previous_bid: The previous bid amount
        current_bid: The current bid amount

    Returns:
        True if the increment is unusual, False otherwise
    """
    if pd.isna(previous_bid) or pd.isna(current_bid):
        return False

    increment = current_bid - previous_bid
    expected = get_expected_increment(previous_bid)

    # Unusual if less than expected (shouldn't happen normally)
    # or more than 3x expected (aggressive jump)
    return increment < expected or increment > 3 * expected


def calculate_increment_deviation(previous_bid: float, current_bid: float) -> float:
    """
    Calculate how much the actual increment deviates from expected.

    Returns the ratio of actual increment to expected increment.
    - < 1.0: below expected minimum
    - = 1.0: exactly expected
    - > 1.0: above expected (more aggressive)

    Args:
        previous_bid: The previous bid amount
        current_bid: The current bid amount

    Returns:
        Ratio of actual to expected increment (NaN if not applicable)
    """
    if pd.isna(previous_bid) or pd.isna(current_bid) or previous_bid <= 0:
        return np.nan

    increment = current_bid - previous_bid
    expected = get_expected_increment(previous_bid)

    if expected <= 0:
        return np.nan

    return increment / expected


# =============================================================================
# Data Loading
# =============================================================================


def load_bid_data() -> pd.DataFrame:
    """
    Load bid data from Hugging Face.

    Returns:
        DataFrame with bid data containing columns:
        - auction_id, item_id, bid_time, bid_number, bid_amount, is_proxy_bid
    """
    logger.info("Loading bid data from HuggingFace...")
    dataset = load_dataset("jpearce610/bid_data", split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} bid records")
    return df


# =============================================================================
# Time-Series Features
# =============================================================================


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series features for each bid.

    Features computed (all causal - no leakage):
    - time_since_first_bid: Seconds since the first bid for this item
    - time_since_last_bid: Seconds since the previous bid (0 for first bid)
    - elapsed_time_fraction: Fraction of elapsed time relative to time from
      first bid to current bid (normalized within available history)

    Args:
        df: DataFrame with bid data, must have item_id and bid_time columns

    Returns:
        DataFrame with time-series features added
    """
    logger.info("Adding time-series features...")
    result = df.copy()

    # Ensure bid_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")

    # Sort by item_id and bid_time to ensure proper ordering
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Calculate first bid time per item (simple transform - first value per group)
    result["_first_bid_time"] = result.groupby("item_id")["bid_time"].transform("first")

    # Time since first bid (seconds)
    result["time_since_first_bid_seconds"] = (
        result["bid_time"] - result["_first_bid_time"]
    ).dt.total_seconds()

    # Time since last bid (seconds) - shift within each item group
    result["_prev_bid_time"] = result.groupby("item_id")["bid_time"].shift(1)
    result["time_since_last_bid_seconds"] = (
        result["bid_time"] - result["_prev_bid_time"]
    ).dt.total_seconds().fillna(0)

    # Elapsed time fraction (0 for first bid, 1 for most recent)
    # This is the ratio of time since first bid to the total elapsed time so far
    # For single-bid items, this is 0 (no elapsed time yet)
    # Use cummax to get the maximum elapsed time up to current bid (causal)
    max_elapsed = result.groupby("item_id")["time_since_first_bid_seconds"].cummax()
    result["elapsed_time_fraction"] = np.where(
        max_elapsed > 0,
        result["time_since_first_bid_seconds"] / max_elapsed,
        0.0
    )

    # Convert to more useful units
    result["time_since_first_bid_minutes"] = result["time_since_first_bid_seconds"] / 60
    result["time_since_first_bid_hours"] = result["time_since_first_bid_seconds"] / 3600
    result["time_since_last_bid_minutes"] = result["time_since_last_bid_seconds"] / 60

    # Clean up temporary columns
    result = result.drop(columns=["_first_bid_time", "_prev_bid_time"])

    logger.info("Time-series features added")
    return result


# =============================================================================
# Bid Count and Intensity Features
# =============================================================================


def add_bid_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bid count and intensity features.

    Features computed (all causal - no leakage):
    - number_of_bids_so_far: Count of bids up to and including current bid
    - is_first_bid: Boolean indicating if this is the first bid
    - rolling_bid_count_1min: Bids in the last 1 minute
    - rolling_bid_count_5min: Bids in the last 5 minutes
    - bid_velocity: Bids per minute in recent window
    - bid_acceleration: Change in velocity compared to previous window
    - time_weighted_bid_velocity: Exponentially weighted bid count

    Args:
        df: DataFrame with bid data, must have item_id, bid_time columns

    Returns:
        DataFrame with bid count features added
    """
    logger.info("Adding bid count features...")
    result = df.copy()

    # Ensure bid_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")

    # Sort by item_id and bid_time
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Number of bids so far (cumulative count within each item)
    result["number_of_bids_so_far"] = result.groupby("item_id").cumcount() + 1

    # Is first bid
    result["is_first_bid"] = (result["number_of_bids_so_far"] == 1).astype(int)

    # Initialize rolling count and velocity columns
    result["rolling_bid_count_1min"] = 0
    result["rolling_bid_count_5min"] = 0
    result["bid_velocity_5min"] = 0.0
    result["bid_acceleration"] = 0.0
    result["time_weighted_bid_velocity"] = 0.0

    # Process each item group
    for _item_id, group in result.groupby("item_id"):
        idx = group.index
        bid_times = group["bid_time"].values

        # Pre-compute all features for this group
        counts_1min = []
        counts_5min = []
        accelerations = []
        weighted_velocities = []

        decay_rate = np.log(2) / 120  # 2-minute half-life

        for i, current_time in enumerate(bid_times):
            # Rolling bid count in last 1 minute (60 seconds)
            window_start_1min = current_time - np.timedelta64(60, "s")
            count_1min = np.sum(bid_times[:i+1] >= window_start_1min)
            counts_1min.append(count_1min)

            # Rolling bid count in last 5 minutes (300 seconds)
            window_start_5min = current_time - np.timedelta64(300, "s")
            count_5min = np.sum(bid_times[:i+1] >= window_start_5min)
            counts_5min.append(count_5min)

            # Acceleration: compare current 5-min window to previous 5-min window
            current_velocity = count_5min / 5.0
            prev_window_end = window_start_5min
            prev_window_start = current_time - np.timedelta64(600, "s")
            prev_count = np.sum((bid_times[:i+1] >= prev_window_start) & (bid_times[:i+1] < prev_window_end))
            prev_velocity = prev_count / 5.0
            accelerations.append(current_velocity - prev_velocity)

            # Time-weighted velocity (exponential decay)
            weighted_sum = 0.0
            for j in range(i + 1):
                time_diff = (current_time - bid_times[j]) / np.timedelta64(1, "s")
                weighted_sum += np.exp(-decay_rate * time_diff)
            weighted_velocities.append(weighted_sum)

        # Assign computed values back to result
        result.loc[idx, "rolling_bid_count_1min"] = counts_1min
        result.loc[idx, "rolling_bid_count_5min"] = counts_5min
        result.loc[idx, "bid_acceleration"] = accelerations
        result.loc[idx, "time_weighted_bid_velocity"] = weighted_velocities

    # Bid velocity (bids per minute in last 5 minutes)
    result["bid_velocity_5min"] = result["rolling_bid_count_5min"] / 5.0

    logger.info("Bid count features added")
    return result


# =============================================================================
# Bid Amount Features
# =============================================================================


def add_bid_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bid amount features.

    Features computed (all causal - no leakage):
    - bid_increment: Change in bid amount from previous bid
    - bid_increment_pct: Percentage increase from previous bid
    - is_unusual_increment: Boolean if increment deviates from MaxSold rules
    - increment_deviation_ratio: Ratio of actual to expected increment
    - proxy_bid_flag: Value from is_proxy_bid column
    - bid_amount_percentile_so_far: Percentile rank among all previous bids
    - bid_amount_relative_to_mean: Deviation from running mean
    - bid_amount_relative_to_median: Deviation from running median
    - bid_amount_relative_to_max: Ratio to running max
    - cumulative_bid_increment_sum: Sum of all increments so far
    - rolling_avg_increment: Average increment in recent bids

    Args:
        df: DataFrame with bid data, must have item_id, bid_amount columns

    Returns:
        DataFrame with bid amount features added
    """
    logger.info("Adding bid amount features...")
    result = df.copy()

    # Ensure proper sorting
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Previous bid amount (shifted within item group)
    result["_prev_bid_amount"] = result.groupby("item_id")["bid_amount"].shift(1)

    # Bid increment
    result["bid_increment"] = result["bid_amount"] - result["_prev_bid_amount"]
    result["bid_increment"] = result["bid_increment"].fillna(result["bid_amount"])  # First bid

    # Bid increment percentage
    result["bid_increment_pct"] = np.where(
        result["_prev_bid_amount"] > 0,
        (result["bid_increment"] / result["_prev_bid_amount"]) * 100,
        np.nan  # First bid has no percentage increase
    )

    # Unusual increment detection using MaxSold rules
    result["is_unusual_increment"] = result.apply(
        lambda row: is_unusual_increment(row["_prev_bid_amount"], row["bid_amount"]),
        axis=1
    ).astype(int)

    # Increment deviation ratio
    result["increment_deviation_ratio"] = result.apply(
        lambda row: calculate_increment_deviation(row["_prev_bid_amount"], row["bid_amount"]),
        axis=1
    )

    # Proxy bid flag (copy from source column if exists)
    if "is_proxy_bid" in result.columns:
        result["proxy_bid_flag"] = result["is_proxy_bid"].astype(int)
    else:
        result["proxy_bid_flag"] = 0

    # Running statistics (causal - only use bids up to current)
    def compute_running_stats(group: pd.DataFrame) -> pd.DataFrame:
        """Compute running statistics for bid amounts."""
        amounts = group["bid_amount"].values
        percentiles = []
        mean_deviations = []
        median_deviations = []
        max_ratios = []

        for i in range(len(amounts)):
            current = amounts[i]
            history = amounts[:i+1]

            # Percentile of current bid among all bids so far
            percentile = (np.sum(history <= current) / len(history)) * 100
            percentiles.append(percentile)

            # Deviation from running mean
            running_mean = np.mean(history)
            mean_deviations.append(current - running_mean)

            # Deviation from running median
            running_median = np.median(history)
            median_deviations.append(current - running_median)

            # Ratio to running max
            running_max = np.max(history)
            max_ratios.append(current / running_max if running_max > 0 else 1.0)

        return pd.DataFrame({
            "bid_amount_percentile_so_far": percentiles,
            "bid_amount_relative_to_mean": mean_deviations,
            "bid_amount_relative_to_median": median_deviations,
            "bid_amount_relative_to_max": max_ratios
        }, index=group.index)

    stats_df = result.groupby("item_id", group_keys=False).apply(compute_running_stats)
    result = pd.concat([result, stats_df], axis=1)

    # Cumulative sum of bid increments
    result["cumulative_bid_increment_sum"] = result.groupby("item_id")["bid_increment"].cumsum()

    # Rolling average of last 5 increments
    result["rolling_avg_increment_5"] = result.groupby("item_id")["bid_increment"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Clean up temporary columns
    result = result.drop(columns=["_prev_bid_amount"])

    logger.info("Bid amount features added")
    return result


# =============================================================================
# Auction Progress Features
# =============================================================================


def add_auction_progress_features(
    df: pd.DataFrame,
    starting_bid_col: str | None = None,
) -> pd.DataFrame:
    """
    Add auction progress features.

    Features computed (all causal - no leakage):
    - bid_number_normalized: Bid sequence number normalized by total bids so far
    - current_to_starting_bid_ratio: Ratio of current bid to starting bid
    - current_to_highest_ratio: Ratio of current bid to highest bid so far
    - is_new_high: Boolean if this bid sets a new high for the item
    - bid_momentum: Rate of price increase (recent vs overall)

    Args:
        df: DataFrame with bid data
        starting_bid_col: Column name for starting bid (optional)

    Returns:
        DataFrame with auction progress features added
    """
    logger.info("Adding auction progress features...")
    result = df.copy()

    # Ensure proper sorting
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Get bid number if not already present
    if "bid_number" not in result.columns:
        result["bid_number"] = result.groupby("item_id").cumcount() + 1

    # Bid number normalized (1/n for first bid, 2/n for second, etc.)
    result["bid_number_normalized"] = result["bid_number"] / result.groupby("item_id")["bid_number"].transform(
        lambda x: x.expanding().max()
    )

    # Ratio to starting bid (if starting bid column provided)
    if starting_bid_col and starting_bid_col in result.columns:
        result["current_to_starting_bid_ratio"] = np.where(
            result[starting_bid_col] > 0,
            result["bid_amount"] / result[starting_bid_col],
            np.nan
        )
    else:
        # Use first bid amount as proxy for starting bid
        first_bids = result.groupby("item_id")["bid_amount"].transform("first")
        result["current_to_starting_bid_ratio"] = np.where(
            first_bids > 0,
            result["bid_amount"] / first_bids,
            np.nan
        )

    # Running max bid amount (causal)
    result["_running_max"] = result.groupby("item_id")["bid_amount"].transform(
        lambda x: x.expanding().max()
    )

    # Ratio to highest bid so far
    result["current_to_highest_ratio"] = np.where(
        result["_running_max"] > 0,
        result["bid_amount"] / result["_running_max"],
        1.0
    )

    # Is this bid a new high?
    result["is_new_high"] = (result["bid_amount"] == result["_running_max"]).astype(int)

    # Bid momentum: compare recent price increase rate to overall rate
    # Process each item group explicitly to avoid apply issues
    result["bid_momentum"] = 0.0

    for _item_id, group in result.groupby("item_id"):
        idx = group.index
        amounts = group["bid_amount"].values
        momentums = []

        for i in range(len(amounts)):
            if i < 2:
                # Not enough data for momentum
                momentums.append(0.0)
                continue

            # Overall rate: total increase from first bid
            overall_increase = amounts[i] - amounts[0]

            # Recent rate: increase in last 3 bids (or available)
            recent_start = max(0, i - 2)
            recent_increase = amounts[i] - amounts[recent_start]

            # Normalize by number of bids in each window
            overall_rate = overall_increase / (i + 1) if i > 0 else 0
            recent_rate = recent_increase / (i - recent_start + 1)

            # Momentum is the difference (positive = accelerating, negative = slowing)
            if overall_rate > 0:
                momentum = (recent_rate - overall_rate) / overall_rate
            else:
                momentum = 0.0

            momentums.append(momentum)

        result.loc[idx, "bid_momentum"] = momentums

    # Clean up
    result = result.drop(columns=["_running_max"])

    logger.info("Auction progress features added")
    return result


# =============================================================================
# Temporal Activity Features
# =============================================================================


def add_temporal_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal activity/clustering features.

    Features computed (all causal - no leakage):
    - is_in_bid_cluster: Boolean if bid is part of a rapid bidding cluster
    - bid_cluster_size: Size of the current bidding cluster
    - time_since_cluster_start: Time since the start of current cluster

    A cluster is defined as a sequence of bids where each bid is within
    30 seconds of the previous bid.

    Args:
        df: DataFrame with bid data

    Returns:
        DataFrame with temporal activity features added
    """
    logger.info("Adding temporal activity features...")
    result = df.copy()

    # Ensure proper sorting and datetime
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Time since previous bid
    if "time_since_last_bid_seconds" not in result.columns:
        result["_prev_time"] = result.groupby("item_id")["bid_time"].shift(1)
        result["time_since_last_bid_seconds"] = (
            result["bid_time"] - result["_prev_time"]
        ).dt.total_seconds().fillna(float("inf"))
        result = result.drop(columns=["_prev_time"])

    # Define cluster threshold (30 seconds)
    cluster_threshold = 30

    # Mark cluster boundaries
    result["_is_cluster_start"] = (
        result["time_since_last_bid_seconds"] > cluster_threshold
    ).astype(int)

    # Compute cluster ID within each item
    result["_cluster_id"] = result.groupby("item_id")["_is_cluster_start"].cumsum()

    # Is in bid cluster (more than 1 bid in the cluster so far)
    cluster_counts = result.groupby(["item_id", "_cluster_id"]).cumcount() + 1
    result["bid_cluster_size_so_far"] = cluster_counts
    result["is_in_bid_cluster"] = (cluster_counts > 1).astype(int)

    # Time since cluster start - process each item group explicitly
    result["time_since_cluster_start_seconds"] = 0.0

    for _item_id, group in result.groupby("item_id"):
        idx = group.index
        cluster_ids = group["_cluster_id"].values
        bid_times = group["bid_time"].values
        times_since_start = []

        cluster_starts = {}
        for cid, bt in zip(cluster_ids, bid_times, strict=False):
            if cid not in cluster_starts:
                cluster_starts[cid] = bt
            time_diff = (bt - cluster_starts[cid]) / np.timedelta64(1, "s")
            times_since_start.append(time_diff)

        result.loc[idx, "time_since_cluster_start_seconds"] = times_since_start

    # Clean up
    result = result.drop(columns=["_is_cluster_start", "_cluster_id"])

    logger.info("Temporal activity features added")
    return result


# =============================================================================
# Main Feature Pipeline
# =============================================================================


def engineer_bid_features(
    df: pd.DataFrame,
    starting_bid_col: str | None = None,
    include_time_series: bool = True,
    include_bid_counts: bool = True,
    include_bid_amounts: bool = True,
    include_auction_progress: bool = True,
    include_temporal_activity: bool = True,
) -> pd.DataFrame:
    """
    Run the complete bid feature engineering pipeline.

    All features are computed causally - for each bid, only information
    available up to and including that bid is used. No leakage from future bids.

    Args:
        df: DataFrame with bid data containing columns:
            - item_id: Unique identifier for the auction item
            - bid_time: Timestamp of the bid
            - bid_amount: Amount of the bid
            - Optional: is_proxy_bid, bid_number
        starting_bid_col: Column name for starting bid (optional)
        include_time_series: Include time-series features
        include_bid_counts: Include bid count and intensity features
        include_bid_amounts: Include bid amount features
        include_auction_progress: Include auction progress features
        include_temporal_activity: Include temporal clustering features

    Returns:
        DataFrame with all engineered features added
    """
    logger.info("=" * 60)
    logger.info("Starting Bid Feature Engineering Pipeline")
    logger.info("=" * 60)

    result = df.copy()

    # Validate required columns
    required_cols = ["item_id", "bid_time", "bid_amount"]
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Handle empty dataframe early
    if len(result) == 0:
        logger.info("Empty dataframe, returning with original columns")
        return result

    # Ensure bid_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(result["bid_time"]):
        result["bid_time"] = pd.to_datetime(result["bid_time"], errors="coerce")

    # Sort data
    result = result.sort_values(["item_id", "bid_time"]).reset_index(drop=True)

    # Apply feature engineering in order
    if include_time_series:
        result = add_time_series_features(result)

    if include_bid_counts:
        result = add_bid_count_features(result)

    if include_bid_amounts:
        result = add_bid_amount_features(result)

    if include_auction_progress:
        result = add_auction_progress_features(result, starting_bid_col)

    if include_temporal_activity:
        result = add_temporal_activity_features(result)

    logger.info(f"Final dataset shape: {result.shape}")
    logger.info(f"Engineered features: {len(result.columns) - len(df.columns)} new columns")
    logger.info("=" * 60)
    logger.info("Bid Feature Engineering Pipeline Complete")
    logger.info("=" * 60)

    return result


def run_bid_feature_pipeline(
    upload_to_hf: bool = False,
    hf_repo_id: str = "engineered_bid_data",
    hf_token: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete bid feature engineering pipeline with data loading.

    Args:
        upload_to_hf: Whether to upload result to Hugging Face
        hf_repo_id: Hugging Face repository ID for upload
        hf_token: Hugging Face token for authentication

    Returns:
        DataFrame with engineered bid features
    """
    # Load data
    df = load_bid_data()

    # Engineer features
    result = engineer_bid_features(df)

    # Upload to Hugging Face (optional)
    if upload_to_hf:
        upload_to_huggingface(result, hf_repo_id, hf_token)

    return result


def upload_to_huggingface(
    df: pd.DataFrame,
    repo_id: str = "engineered_bid_data",
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
# Feature Documentation
# =============================================================================

FEATURE_DOCUMENTATION = """
# Bid Feature Engineering - Leakage Prevention Documentation

All features in this module are computed **causally** - for any bid row,
features are computed using only data available up to and including that bid.
No future bids or final auction outcomes are used.

## Feature Categories and Leakage Prevention

### 1. Time-Series Features
| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| time_since_first_bid_seconds | Seconds since first bid | Uses only bids ≤ current |
| time_since_last_bid_seconds | Seconds since previous bid | Uses only previous bid |
| elapsed_time_fraction | Fraction of elapsed time | Normalized by max observed so far |

### 2. Bid Count Features
| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| number_of_bids_so_far | Cumulative bid count | Expanding count up to current |
| is_first_bid | Boolean for first bid | Current bid only |
| rolling_bid_count_1min | Bids in last 1 minute | Time window from past only |
| rolling_bid_count_5min | Bids in last 5 minutes | Time window from past only |
| bid_velocity_5min | Bids per minute (5min window) | Based on rolling count |
| bid_acceleration | Change in velocity | Compares two past windows |
| time_weighted_bid_velocity | Exponentially weighted count | Decay from past bids |

### 3. Bid Amount Features
| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| bid_increment | Increase from previous bid | Uses only previous bid |
| bid_increment_pct | Percentage increase | Uses only previous bid |
| is_unusual_increment | Deviates from MaxSold rules | Current vs previous only |
| increment_deviation_ratio | Actual/expected increment | Current vs previous only |
| proxy_bid_flag | Is this a proxy bid | Current bid only |
| bid_amount_percentile_so_far | Percentile among past bids | Expanding window |
| bid_amount_relative_to_mean | Deviation from running mean | Expanding mean |
| bid_amount_relative_to_median | Deviation from running median | Expanding median |
| bid_amount_relative_to_max | Ratio to running max | Expanding max |
| cumulative_bid_increment_sum | Sum of all increments | Cumulative sum |
| rolling_avg_increment_5 | Average of last 5 increments | Rolling window from past |

### 4. Auction Progress Features
| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| bid_number_normalized | Normalized bid position | Normalized by max observed |
| current_to_starting_bid_ratio | Ratio to first bid | Uses first bid only |
| current_to_highest_ratio | Ratio to highest so far | Expanding max |
| is_new_high | This bid is highest so far | Expanding max comparison |
| bid_momentum | Rate of price increase | Compares recent vs overall |

### 5. Temporal Activity Features
| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| is_in_bid_cluster | Part of rapid bid sequence | Past cluster membership |
| bid_cluster_size_so_far | Size of current cluster | Expanding within cluster |
| time_since_cluster_start_seconds | Time since cluster began | Cluster start from past |
"""


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the bid feature engineering pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run bid feature engineering pipeline"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload result to Hugging Face",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="engineered_bid_data",
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
    df = run_bid_feature_pipeline(
        upload_to_hf=args.upload,
        hf_repo_id=args.repo_id,
        hf_token=args.token,
    )

    print(f"Pipeline complete. Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
