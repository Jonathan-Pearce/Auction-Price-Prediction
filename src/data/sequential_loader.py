# =============================================================================
# Sequential Data Loader for Bid Data
# =============================================================================
"""
Data loading utilities for sequential deep learning models (LSTM/GRU).

Loads bid data from Hugging Face datasets and prepares it for sequence modeling.
All configuration is loaded from sequential_config.yaml.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

# Path to YAML config file
_CONFIG_FILE: Path = Path(__file__).parent / "sequential_config.yaml"


# =============================================================================
# Configuration Loading
# =============================================================================


@lru_cache(maxsize=1)
def load_sequential_config() -> dict[str, Any]:
    """
    Load sequential model configuration from YAML file.

    Returns:
        Dictionary containing all configuration values
    """
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def get_config_value(*keys: str, default: Any = None) -> Any:
    """
    Get a configuration value by nested keys.

    Args:
        *keys: Nested keys to traverse (e.g., 'sequence', 'max_length')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> get_config_value('sequence', 'max_length')
        100
    """
    config = load_sequential_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config


# =============================================================================
# Data Source Configuration Accessors
# =============================================================================


def get_hf_dataset_repo() -> str:
    """Get Hugging Face dataset repository for bid data."""
    return get_config_value("data_source", "hf_dataset_repo", default="jpearce610/bid_data")


def get_hf_dataset_split() -> str:
    """Get default split to load from Hugging Face dataset."""
    return get_config_value("data_source", "hf_dataset_split", default="train")


def get_streaming_enabled() -> bool:
    """Check if streaming mode is enabled."""
    return get_config_value("data_source", "streaming", default=False)


# =============================================================================
# Sequence Configuration Accessors
# =============================================================================


def get_max_sequence_length() -> int:
    """Get maximum sequence length for padding/truncation."""
    return get_config_value("sequence", "max_length", default=100)


def get_min_sequence_length() -> int:
    """Get minimum sequence length (items below this are excluded)."""
    return get_config_value("sequence", "min_length", default=1)


def get_padding_strategy() -> str:
    """Get padding strategy ('pre' or 'post')."""
    return get_config_value("sequence", "padding_strategy", default="post")


def get_truncation_strategy() -> str:
    """Get truncation strategy ('pre' or 'post')."""
    return get_config_value("sequence", "truncation_strategy", default="pre")


def get_padding_value() -> float:
    """Get padding value for numeric features."""
    return get_config_value("sequence", "padding_value", default=0.0)


def get_include_zero_bid_items() -> bool:
    """Check if items with no bids should be included."""
    return get_config_value("sequence", "include_zero_bid_items", default=False)


# =============================================================================
# Feature Configuration Accessors
# =============================================================================


def get_per_bid_features_config() -> dict[str, dict[str, Any]]:
    """Get configuration for per-bid features."""
    return get_config_value("features", "per_bid", default={})


def get_sequence_level_features_config() -> dict[str, dict[str, Any]]:
    """Get configuration for sequence-level features."""
    return get_config_value("features", "sequence_level", default={})


def get_enabled_per_bid_features() -> list[str]:
    """Get list of enabled per-bid feature names."""
    config = get_per_bid_features_config()
    return [name for name, cfg in config.items() if cfg.get("enabled", False)]


def get_enabled_sequence_level_features() -> list[str]:
    """Get list of enabled sequence-level feature names."""
    config = get_sequence_level_features_config()
    return [name for name, cfg in config.items() if cfg.get("enabled", False)]


# =============================================================================
# Normalization Configuration Accessors
# =============================================================================


def get_normalization_config() -> dict[str, Any]:
    """Get normalization configuration."""
    return get_config_value("normalization", default={})


def get_fit_on_train_only() -> bool:
    """Check if normalizers should be fit on training data only."""
    return get_config_value("normalization", "fit_on_train_only", default=True)


def get_minmax_range() -> tuple[float, float]:
    """Get min-max normalization range."""
    range_list = get_config_value("normalization", "minmax_range", default=[0.0, 1.0])
    return tuple(range_list)


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_bid_data_from_huggingface(
    split: str | None = None,
    streaming: bool | None = None,
) -> pd.DataFrame:
    """
    Load bid data from Hugging Face dataset.

    Args:
        split: Dataset split to load (defaults to config value)
        streaming: Whether to stream the dataset (defaults to config value)

    Returns:
        DataFrame with bid data
    """
    from datasets import load_dataset

    repo = get_hf_dataset_repo()
    split = split or get_hf_dataset_split()
    streaming = streaming if streaming is not None else get_streaming_enabled()

    logger.info(f"Loading bid data from Hugging Face: {repo}, split={split}")

    dataset = load_dataset(repo, split=split, streaming=streaming)

    if streaming:
        # For streaming, convert iterator to list (limited for memory)
        logger.warning("Streaming mode: loading data iteratively")
        data = list(dataset)
        df = pd.DataFrame(data)
    else:
        df = dataset.to_pandas()

    logger.info(f"Loaded {len(df)} bid records")
    return df


def group_bids_by_item(
    bids_df: pd.DataFrame,
    auction_id_col: str = "auction_id",
    item_id_col: str = "item_id",
    time_col: str = "bid_time",
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Group bids by item, sorted by bid time.

    Args:
        bids_df: DataFrame with bid data
        auction_id_col: Name of auction ID column
        item_id_col: Name of item ID column
        time_col: Name of bid time column

    Returns:
        Dictionary mapping (auction_id, item_id) to DataFrame of bids
    """
    logger.info("Grouping bids by item...")

    # Sort by bid time within each item
    bids_df = bids_df.sort_values([auction_id_col, item_id_col, time_col])

    # Group by item
    grouped = {}
    for (auction_id, item_id), group in bids_df.groupby([auction_id_col, item_id_col]):
        grouped[(auction_id, item_id)] = group.reset_index(drop=True)

    logger.info(f"Grouped bids into {len(grouped)} items")
    return grouped


def filter_items_by_sequence_length(
    grouped_bids: dict[tuple[int, int], pd.DataFrame],
    min_length: int | None = None,
    max_length: int | None = None,
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Filter items based on sequence length constraints.

    Args:
        grouped_bids: Dictionary of (auction_id, item_id) -> bids DataFrame
        min_length: Minimum number of bids (defaults to config)
        max_length: Maximum number of bids (defaults to config)

    Returns:
        Filtered dictionary
    """
    min_length = min_length if min_length is not None else get_min_sequence_length()

    filtered = {}
    for key, bids in grouped_bids.items():
        if len(bids) >= min_length:
            filtered[key] = bids

    logger.info(
        f"Filtered items: {len(grouped_bids)} -> {len(filtered)} "
        f"(min_length={min_length})"
    )
    return filtered


def get_item_ids_from_bid_data(bids_df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Extract unique (auction_id, item_id) pairs from bid data.

    Args:
        bids_df: DataFrame with bid data

    Returns:
        List of (auction_id, item_id) tuples
    """
    pairs = bids_df[["auction_id", "item_id"]].drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


# =============================================================================
# Data Splitting Functions
# =============================================================================


def get_splitting_config() -> dict[str, Any]:
    """Get data splitting configuration."""
    return get_config_value("splitting", default={})


def split_item_ids(
    item_ids: list[tuple[int, int]],
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    random_seed: int | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Split item IDs into train, validation, and test sets.

    Args:
        item_ids: List of (auction_id, item_id) tuples
        train_ratio: Train set ratio (defaults to config)
        val_ratio: Validation set ratio (defaults to config)
        test_ratio: Test set ratio (defaults to config)
        random_seed: Random seed (defaults to config)

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    from sklearn.model_selection import train_test_split

    config = get_splitting_config()
    train_ratio = train_ratio if train_ratio is not None else config.get("train_ratio", 0.7)
    val_ratio = val_ratio if val_ratio is not None else config.get("val_ratio", 0.15)
    test_ratio = test_ratio if test_ratio is not None else config.get("test_ratio", 0.15)
    random_seed = random_seed if random_seed is not None else config.get("random_seed", 42)

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        item_ids,
        train_size=train_ratio,
        random_state=random_seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test_ratio,
        random_state=random_seed,
    )

    logger.info(
        f"Split {len(item_ids)} items: train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    return train_ids, val_ids, test_ids
