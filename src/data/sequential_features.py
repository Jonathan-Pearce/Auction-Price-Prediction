# =============================================================================
# Sequential Feature Engineering for Bid Data
# =============================================================================
"""
Feature engineering utilities for sequential deep learning models (LSTM/GRU).

Transforms raw bid data into sequences of features suitable for training.
All configuration is loaded from sequential_config.yaml.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.sequential_loader import (
    get_enabled_per_bid_features,
    get_max_sequence_length,
    get_minmax_range,
    get_padding_strategy,
    get_padding_value,
    get_per_bid_features_config,
    get_sequence_level_features_config,
    get_truncation_strategy,
)

# =============================================================================
# Normalization Statistics
# =============================================================================


class NormalizationStats:
    """
    Stores and applies normalization statistics for features.

    Supports min-max scaling, standard scaling, and log transformation.
    """

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float]] = {}

    def fit(
        self,
        feature_name: str,
        values: np.ndarray,
        method: str = "minmax",
    ) -> None:
        """
        Compute and store normalization statistics for a feature.

        Args:
            feature_name: Name of the feature
            values: Array of values to fit on
            method: Normalization method (minmax, standard, log)
        """
        values = values[~np.isnan(values)]  # Remove NaN values

        if method == "minmax":
            self.stats[feature_name] = {
                "method": "minmax",
                "min": float(np.min(values)) if len(values) > 0 else 0.0,
                "max": float(np.max(values)) if len(values) > 0 else 1.0,
            }
        elif method == "standard":
            self.stats[feature_name] = {
                "method": "standard",
                "mean": float(np.mean(values)) if len(values) > 0 else 0.0,
                "std": float(np.std(values)) if len(values) > 0 else 1.0,
            }
        elif method == "log":
            self.stats[feature_name] = {
                "method": "log",
                "min": float(np.min(values)) if len(values) > 0 else 0.0,
                "max": float(np.max(np.log1p(values))) if len(values) > 0 else 1.0,
            }
        else:
            self.stats[feature_name] = {"method": "none"}

    def transform(
        self,
        feature_name: str,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Apply normalization to values using stored statistics.

        Args:
            feature_name: Name of the feature
            values: Array of values to transform

        Returns:
            Transformed values
        """
        if feature_name not in self.stats:
            logger.warning(f"No stats for feature {feature_name}, returning unchanged")
            return values

        stat = self.stats[feature_name]
        method = stat.get("method", "none")

        if method == "minmax":
            min_val = stat["min"]
            max_val = stat["max"]
            range_val = max_val - min_val
            if range_val == 0:
                return np.zeros_like(values)
            minmax_range = get_minmax_range()
            normalized = (values - min_val) / range_val
            return normalized * (minmax_range[1] - minmax_range[0]) + minmax_range[0]

        elif method == "standard":
            mean_val = stat["mean"]
            std_val = stat["std"]
            if std_val == 0:
                return np.zeros_like(values)
            return (values - mean_val) / std_val

        elif method == "log":
            min_val = stat["min"]
            max_val = stat["max"]
            log_values = np.log1p(np.maximum(values, 0))  # log1p for stability
            range_val = max_val - min_val
            if range_val == 0:
                return np.zeros_like(values)
            minmax_range = get_minmax_range()
            normalized = (log_values - min_val) / range_val
            return normalized * (minmax_range[1] - minmax_range[0]) + minmax_range[0]

        return values

    def save(self, path: Path) -> None:
        """Save normalization statistics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved normalization stats to {path}")

    def load(self, path: Path) -> None:
        """Load normalization statistics from JSON file."""
        with open(path) as f:
            self.stats = json.load(f)
        logger.info(f"Loaded normalization stats from {path}")


# =============================================================================
# Per-Bid Feature Extraction
# =============================================================================


def extract_bid_amount(bids_df: pd.DataFrame, amount_col: str = "bid_amount") -> np.ndarray:
    """
    Extract raw bid amounts from bid data.

    Args:
        bids_df: DataFrame with bid data for a single item
        amount_col: Name of the bid amount column

    Returns:
        Array of bid amounts
    """
    return bids_df[amount_col].values.astype(np.float32)


def extract_time_since_previous_bid(
    bids_df: pd.DataFrame,
    time_col: str = "bid_time",
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Calculate time since previous bid in seconds.

    Args:
        bids_df: DataFrame with bid data for a single item (sorted by time)
        time_col: Name of the bid time column
        clip_max: Maximum value to clip to (optional)

    Returns:
        Array of time differences (first bid has 0)
    """
    times = pd.to_datetime(bids_df[time_col])
    time_diffs = times.diff().dt.total_seconds().fillna(0).values.astype(np.float32)

    if clip_max is not None:
        time_diffs = np.clip(time_diffs, 0, clip_max)

    return time_diffs


def extract_relative_bid_amount(
    bids_df: pd.DataFrame,
    amount_col: str = "bid_amount",
) -> np.ndarray:
    """
    Calculate bid amount relative to the first bid.

    Args:
        bids_df: DataFrame with bid data for a single item
        amount_col: Name of the bid amount column

    Returns:
        Array of relative bid amounts (first bid = 1.0)
    """
    amounts = bids_df[amount_col].values.astype(np.float32)
    first_bid = amounts[0] if len(amounts) > 0 else 1.0
    if first_bid == 0:
        first_bid = 1.0  # Avoid division by zero
    return amounts / first_bid


def extract_bid_increment(
    bids_df: pd.DataFrame,
    amount_col: str = "bid_amount",
) -> np.ndarray:
    """
    Calculate bid increment (current bid - previous bid).

    Args:
        bids_df: DataFrame with bid data for a single item
        amount_col: Name of the bid amount column

    Returns:
        Array of bid increments (first bid has 0)
    """
    amounts = bids_df[amount_col].values.astype(np.float32)
    increments = np.diff(amounts, prepend=amounts[0])
    increments[0] = 0.0  # First bid has no increment
    return increments


def extract_bid_position(bids_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate normalized bid position in sequence (0 to 1).

    Args:
        bids_df: DataFrame with bid data for a single item

    Returns:
        Array of normalized positions
    """
    n = len(bids_df)
    if n <= 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(0, 1, n, dtype=np.float32)


def extract_cumulative_bid_count(bids_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate cumulative bid count at each position.

    Args:
        bids_df: DataFrame with bid data for a single item

    Returns:
        Array of cumulative counts (1, 2, 3, ...)
    """
    n = len(bids_df)
    return np.arange(1, n + 1, dtype=np.float32)


def extract_is_proxy(
    bids_df: pd.DataFrame,
    proxy_col: str = "bid_is_proxy",
) -> np.ndarray:
    """
    Extract proxy bid flag (0 or 1).

    Args:
        bids_df: DataFrame with bid data for a single item
        proxy_col: Name of the proxy bid column

    Returns:
        Array of proxy flags
    """
    if proxy_col in bids_df.columns:
        return bids_df[proxy_col].fillna(0).astype(np.float32).values
    return np.zeros(len(bids_df), dtype=np.float32)


# =============================================================================
# Sequence-Level Feature Extraction
# =============================================================================


def extract_total_bids(bids_df: pd.DataFrame) -> float:
    """
    Extract total number of bids for an item.

    Args:
        bids_df: DataFrame with bid data for a single item

    Returns:
        Total bid count
    """
    return float(len(bids_df))


def extract_starting_bid(
    bids_df: pd.DataFrame,
    amount_col: str = "bid_amount",
) -> float:
    """
    Extract starting bid (first bid amount).

    Args:
        bids_df: DataFrame with bid data for a single item
        amount_col: Name of the bid amount column

    Returns:
        Starting bid amount
    """
    if len(bids_df) == 0:
        return 0.0
    return float(bids_df[amount_col].iloc[0])


# =============================================================================
# Feature Extraction Pipeline
# =============================================================================


def extract_per_bid_features(
    bids_df: pd.DataFrame,
    feature_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract all enabled per-bid features from bid data.

    Args:
        bids_df: DataFrame with bid data for a single item (sorted by time)
        feature_config: Feature configuration (defaults to config file)

    Returns:
        Dictionary mapping feature names to value arrays
    """
    if feature_config is None:
        feature_config = get_per_bid_features_config()

    features = {}

    for feature_name, config in feature_config.items():
        if not config.get("enabled", False):
            continue

        if feature_name == "bid_amount":
            features[feature_name] = extract_bid_amount(bids_df)

        elif feature_name == "time_since_previous_bid":
            clip_max = config.get("clip_max")
            features[feature_name] = extract_time_since_previous_bid(
                bids_df, clip_max=clip_max
            )

        elif feature_name == "relative_bid_amount":
            features[feature_name] = extract_relative_bid_amount(bids_df)

        elif feature_name == "bid_increment":
            features[feature_name] = extract_bid_increment(bids_df)

        elif feature_name == "bid_position":
            features[feature_name] = extract_bid_position(bids_df)

        elif feature_name == "cumulative_bid_count":
            features[feature_name] = extract_cumulative_bid_count(bids_df)

        elif feature_name == "is_proxy":
            features[feature_name] = extract_is_proxy(bids_df)

    return features


def extract_sequence_level_features(
    bids_df: pd.DataFrame,
    feature_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Extract sequence-level (static) features from bid data.

    Args:
        bids_df: DataFrame with bid data for a single item
        feature_config: Feature configuration (defaults to config file)

    Returns:
        Dictionary mapping feature names to values
    """
    if feature_config is None:
        feature_config = get_sequence_level_features_config()

    features = {}

    for feature_name, config in feature_config.items():
        if not config.get("enabled", False):
            continue

        if feature_name == "total_bids":
            features[feature_name] = extract_total_bids(bids_df)

        elif feature_name == "starting_bid":
            features[feature_name] = extract_starting_bid(bids_df)

    return features


def extract_label(
    bids_df: pd.DataFrame,
    amount_col: str = "bid_amount",
) -> float:
    """
    Extract the prediction label (final/highest bid amount).

    Args:
        bids_df: DataFrame with bid data for a single item (sorted by time)
        amount_col: Name of the bid amount column

    Returns:
        Final bid amount (label)
    """
    if len(bids_df) == 0:
        return 0.0
    return float(bids_df[amount_col].iloc[-1])


# =============================================================================
# Sequence Padding and Truncation
# =============================================================================


def pad_sequence(
    features: dict[str, np.ndarray],
    max_length: int | None = None,
    padding_strategy: str | None = None,
    padding_value: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Pad feature sequences to a fixed length.

    Args:
        features: Dictionary of feature arrays
        max_length: Maximum sequence length (defaults to config)
        padding_strategy: 'pre' or 'post' (defaults to config)
        padding_value: Value to use for padding (defaults to config)

    Returns:
        Dictionary of padded feature arrays
    """
    max_length = max_length if max_length is not None else get_max_sequence_length()
    padding_strategy = padding_strategy or get_padding_strategy()
    padding_value = padding_value if padding_value is not None else get_padding_value()

    padded = {}
    for name, values in features.items():
        seq_len = len(values)

        if seq_len >= max_length:
            # No padding needed
            padded[name] = values
        else:
            pad_len = max_length - seq_len
            pad_array = np.full(pad_len, padding_value, dtype=values.dtype)

            if padding_strategy == "pre":
                padded[name] = np.concatenate([pad_array, values])
            else:  # post
                padded[name] = np.concatenate([values, pad_array])

    return padded


def truncate_sequence(
    features: dict[str, np.ndarray],
    max_length: int | None = None,
    truncation_strategy: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Truncate feature sequences to a maximum length.

    Args:
        features: Dictionary of feature arrays
        max_length: Maximum sequence length (defaults to config)
        truncation_strategy: 'pre' (keep last N) or 'post' (keep first N)

    Returns:
        Dictionary of truncated feature arrays
    """
    max_length = max_length if max_length is not None else get_max_sequence_length()
    truncation_strategy = truncation_strategy or get_truncation_strategy()

    truncated = {}
    for name, values in features.items():
        if len(values) <= max_length:
            truncated[name] = values
        else:
            if truncation_strategy == "pre":
                # Keep last N bids (most recent)
                truncated[name] = values[-max_length:]
            else:  # post
                # Keep first N bids
                truncated[name] = values[:max_length]

    return truncated


def create_padding_mask(
    seq_length: int,
    max_length: int | None = None,
    padding_strategy: str | None = None,
) -> np.ndarray:
    """
    Create a padding mask (1 for real values, 0 for padding).

    Args:
        seq_length: Actual sequence length
        max_length: Padded sequence length (defaults to config)
        padding_strategy: 'pre' or 'post' (defaults to config)

    Returns:
        Binary mask array
    """
    max_length = max_length if max_length is not None else get_max_sequence_length()
    padding_strategy = padding_strategy or get_padding_strategy()

    mask = np.zeros(max_length, dtype=np.float32)
    if seq_length >= max_length:
        mask[:] = 1.0
    elif padding_strategy == "pre":
        mask[-seq_length:] = 1.0
    else:  # post
        mask[:seq_length] = 1.0

    return mask


# =============================================================================
# Full Feature Pipeline
# =============================================================================


def process_item_bids(
    bids_df: pd.DataFrame,
    norm_stats: NormalizationStats | None = None,
    apply_normalization: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """
    Process bids for a single item into model-ready features.

    Args:
        bids_df: DataFrame with bid data for a single item (sorted by time)
        norm_stats: NormalizationStats instance (optional)
        apply_normalization: Whether to apply normalization

    Returns:
        Tuple of:
        - sequence_features: (seq_len, num_features) array
        - padding_mask: (max_length,) binary mask
        - label: float (final bid amount)
        - static_features: dict of sequence-level features
    """
    # Extract per-bid features
    per_bid_features = extract_per_bid_features(bids_df)

    # Truncate if needed
    per_bid_features = truncate_sequence(per_bid_features)

    # Get sequence length after truncation
    seq_len = len(next(iter(per_bid_features.values()))) if per_bid_features else 0

    # Apply normalization if stats provided
    if apply_normalization and norm_stats is not None:
        config = get_per_bid_features_config()
        for name, values in per_bid_features.items():
            if config.get(name, {}).get("normalize", False):
                per_bid_features[name] = norm_stats.transform(name, values)

    # Pad sequences
    per_bid_features = pad_sequence(per_bid_features)

    # Create padding mask
    max_length = get_max_sequence_length()
    padding_mask = create_padding_mask(min(seq_len, max_length), max_length)

    # Stack features into array (seq_len x num_features)
    feature_names = get_enabled_per_bid_features()
    if feature_names:
        sequence_features = np.stack(
            [per_bid_features[name] for name in feature_names if name in per_bid_features],
            axis=-1,
        )
    else:
        sequence_features = np.zeros((max_length, 1), dtype=np.float32)

    # Extract sequence-level features
    static_features = extract_sequence_level_features(bids_df)

    # Apply normalization to static features if stats provided
    if apply_normalization and norm_stats is not None:
        config = get_sequence_level_features_config()
        for name, value in static_features.items():
            if config.get(name, {}).get("normalize", False):
                static_features[name] = float(
                    norm_stats.transform(name, np.array([value]))[0]
                )

    # Extract label
    label = extract_label(bids_df)

    return sequence_features, padding_mask, label, static_features


def compute_normalization_stats(
    grouped_bids: dict[tuple[int, int], pd.DataFrame],
    item_ids: list[tuple[int, int]] | None = None,
) -> NormalizationStats:
    """
    Compute normalization statistics from a set of items.

    Args:
        grouped_bids: Dictionary of (auction_id, item_id) -> bids DataFrame
        item_ids: Subset of items to compute stats from (all if None)

    Returns:
        NormalizationStats instance with computed statistics
    """
    logger.info("Computing normalization statistics...")

    stats = NormalizationStats()
    per_bid_config = get_per_bid_features_config()
    seq_level_config = get_sequence_level_features_config()

    # Collect all values for each feature
    all_values: dict[str, list[np.ndarray]] = {}

    items_to_process = item_ids if item_ids else list(grouped_bids.keys())

    for item_id in items_to_process:
        if item_id not in grouped_bids:
            continue

        bids_df = grouped_bids[item_id]

        # Per-bid features
        per_bid_features = extract_per_bid_features(bids_df)
        for name, values in per_bid_features.items():
            if name not in all_values:
                all_values[name] = []
            all_values[name].append(values)

        # Sequence-level features
        static_features = extract_sequence_level_features(bids_df)
        for name, value in static_features.items():
            if name not in all_values:
                all_values[name] = []
            all_values[name].append(np.array([value]))

    # Fit statistics for each feature
    for name, values_list in all_values.items():
        combined = np.concatenate(values_list)

        # Determine normalization method from config
        if name in per_bid_config:
            config = per_bid_config[name]
        elif name in seq_level_config:
            config = seq_level_config[name]
        else:
            config = {}

        if config.get("normalize", False):
            method = config.get("normalization_method", "minmax")
            stats.fit(name, combined, method=method)
        else:
            stats.fit(name, combined, method="none")

    logger.info(f"Computed normalization stats for {len(all_values)} features")
    return stats
