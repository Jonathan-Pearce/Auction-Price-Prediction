# =============================================================================
# PyTorch Dataset for Sequential Bid Data
# =============================================================================
"""
PyTorch Dataset and DataLoader utilities for sequential models (LSTM/GRU).

Provides ready-to-use datasets for training, validation, and testing.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.data.sequential_features import (
    NormalizationStats,
    compute_normalization_stats,
    process_item_bids,
)
from src.data.sequential_loader import (
    filter_items_by_sequence_length,
    get_config_value,
    get_enabled_per_bid_features,
    get_enabled_sequence_level_features,
    get_max_sequence_length,
    group_bids_by_item,
    load_bid_data_from_huggingface,
    split_item_ids,
)

# =============================================================================
# PyTorch Dataset
# =============================================================================


class BidSequenceDataset(Dataset):
    """
    PyTorch Dataset for bid sequence data.

    Loads bid data, extracts features, and provides ready-to-use tensors
    for training sequential models.
    """

    def __init__(
        self,
        grouped_bids: dict[tuple[int, int], pd.DataFrame],
        item_ids: list[tuple[int, int]],
        norm_stats: NormalizationStats | None = None,
        apply_normalization: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            grouped_bids: Dictionary mapping (auction_id, item_id) to bid DataFrame
            item_ids: List of (auction_id, item_id) tuples to include
            norm_stats: NormalizationStats for feature normalization
            apply_normalization: Whether to apply normalization
        """
        self.grouped_bids = grouped_bids
        self.item_ids = item_ids
        self.norm_stats = norm_stats
        self.apply_normalization = apply_normalization

        # Pre-compute feature dimensions
        self.num_per_bid_features = len(get_enabled_per_bid_features())
        self.num_static_features = len(get_enabled_sequence_level_features())
        self.max_seq_length = get_max_sequence_length()

        logger.info(
            f"BidSequenceDataset initialized with {len(item_ids)} items, "
            f"{self.num_per_bid_features} per-bid features, "
            f"{self.num_static_features} static features"
        )

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
            - 'sequence': (max_seq_length, num_features) tensor
            - 'mask': (max_seq_length,) tensor
            - 'label': scalar tensor
            - 'static_features': (num_static_features,) tensor (if any)
            - 'auction_id': int
            - 'item_id': int
        """
        auction_id, item_id = self.item_ids[idx]
        bids_df = self.grouped_bids[(auction_id, item_id)]

        # Process bids into features
        sequence_features, padding_mask, label, static_features = process_item_bids(
            bids_df,
            norm_stats=self.norm_stats,
            apply_normalization=self.apply_normalization,
        )

        # Convert to tensors
        sample = {
            "sequence": torch.from_numpy(sequence_features).float(),
            "mask": torch.from_numpy(padding_mask).float(),
            "label": torch.tensor(label, dtype=torch.float32),
            "auction_id": auction_id,
            "item_id": item_id,
        }

        # Add static features if any
        if static_features:
            static_array = np.array(
                list(static_features.values()), dtype=np.float32
            )
            sample["static_features"] = torch.from_numpy(static_array).float()

        return sample

    @property
    def feature_dim(self) -> int:
        """Number of features per timestep."""
        return self.num_per_bid_features

    @property
    def static_feature_dim(self) -> int:
        """Number of static/sequence-level features."""
        return self.num_static_features


# =============================================================================
# Data Preparation Functions
# =============================================================================


def prepare_sequential_data(
    bids_df: pd.DataFrame | None = None,
    split: str | None = None,
) -> tuple[
    dict[tuple[int, int], pd.DataFrame],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """
    Prepare bid data for sequential modeling.

    Args:
        bids_df: DataFrame with bid data (loads from HF if None)
        split: Optional HF dataset split to load

    Returns:
        Tuple of:
        - grouped_bids: Dictionary of (auction_id, item_id) -> bids DataFrame
        - train_ids: List of training item IDs
        - val_ids: List of validation item IDs
        - test_ids: List of test item IDs
    """
    # Load data if not provided
    if bids_df is None:
        bids_df = load_bid_data_from_huggingface(split=split)

    # Group by item
    grouped_bids = group_bids_by_item(bids_df)

    # Filter by sequence length
    grouped_bids = filter_items_by_sequence_length(grouped_bids)

    # Get all item IDs
    item_ids = list(grouped_bids.keys())

    # Split into train/val/test
    train_ids, val_ids, test_ids = split_item_ids(item_ids)

    return grouped_bids, train_ids, val_ids, test_ids


def create_datasets(
    grouped_bids: dict[tuple[int, int], pd.DataFrame],
    train_ids: list[tuple[int, int]],
    val_ids: list[tuple[int, int]],
    test_ids: list[tuple[int, int]],
    fit_normalization_on_train: bool = True,
) -> tuple[BidSequenceDataset, BidSequenceDataset, BidSequenceDataset, NormalizationStats]:
    """
    Create PyTorch datasets for train, validation, and test sets.

    Args:
        grouped_bids: Dictionary of (auction_id, item_id) -> bids DataFrame
        train_ids: Training item IDs
        val_ids: Validation item IDs
        test_ids: Test item IDs
        fit_normalization_on_train: Whether to fit normalization on training data only

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, norm_stats)
    """
    # Compute normalization statistics
    if fit_normalization_on_train:
        norm_stats = compute_normalization_stats(grouped_bids, train_ids)
    else:
        all_ids = train_ids + val_ids + test_ids
        norm_stats = compute_normalization_stats(grouped_bids, all_ids)

    # Create datasets
    train_dataset = BidSequenceDataset(
        grouped_bids, train_ids, norm_stats, apply_normalization=True
    )
    val_dataset = BidSequenceDataset(
        grouped_bids, val_ids, norm_stats, apply_normalization=True
    )
    test_dataset = BidSequenceDataset(
        grouped_bids, test_ids, norm_stats, apply_normalization=True
    )

    logger.info(
        f"Created datasets: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, norm_stats


def create_dataloaders(
    train_dataset: BidSequenceDataset,
    val_dataset: BidSequenceDataset,
    test_dataset: BidSequenceDataset,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (defaults to config)
        num_workers: Number of data loading workers (defaults to config)
        pin_memory: Whether to pin memory (defaults to config)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or get_config_value("dataloader", "batch_size", default=32)
    num_workers = num_workers or get_config_value("dataloader", "num_workers", default=4)
    pin_memory = pin_memory if pin_memory is not None else get_config_value(
        "dataloader", "pin_memory", default=True
    )
    shuffle_train = get_config_value("dataloader", "shuffle_train", default=True)
    shuffle_val = get_config_value("dataloader", "shuffle_val", default=False)
    shuffle_test = get_config_value("dataloader", "shuffle_test", default=False)
    drop_last = get_config_value("dataloader", "drop_last", default=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"Created DataLoaders with batch_size={batch_size}, "
        f"num_workers={num_workers}"
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Convenience Function
# =============================================================================


def get_sequential_dataloaders(
    bids_df: pd.DataFrame | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, NormalizationStats, dict[str, Any]]:
    """
    High-level function to get ready-to-use DataLoaders from bid data.

    This is the main entry point for loading sequential data.

    Args:
        bids_df: DataFrame with bid data (loads from HF if None)
        batch_size: Batch size override

    Returns:
        Tuple of:
        - train_loader: Training DataLoader
        - val_loader: Validation DataLoader
        - test_loader: Test DataLoader
        - norm_stats: NormalizationStats instance
        - metadata: Dictionary with dataset info

    Example:
        >>> train_loader, val_loader, test_loader, norm_stats, meta = get_sequential_dataloaders()
        >>> for batch in train_loader:
        ...     sequences = batch['sequence']  # (B, T, F)
        ...     masks = batch['mask']          # (B, T)
        ...     labels = batch['label']        # (B,)
    """
    # Prepare data
    grouped_bids, train_ids, val_ids, test_ids = prepare_sequential_data(bids_df)

    # Create datasets
    train_dataset, val_dataset, test_dataset, norm_stats = create_datasets(
        grouped_bids, train_ids, val_ids, test_ids
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    # Compile metadata
    metadata = {
        "num_items": len(grouped_bids),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "max_seq_length": get_max_sequence_length(),
        "feature_dim": train_dataset.feature_dim,
        "static_feature_dim": train_dataset.static_feature_dim,
        "per_bid_features": get_enabled_per_bid_features(),
        "static_features": get_enabled_sequence_level_features(),
    }

    return train_loader, val_loader, test_loader, norm_stats, metadata


# =============================================================================
# Save/Load Utilities
# =============================================================================


def save_normalization_stats(
    norm_stats: NormalizationStats,
    path: Path | str,
) -> None:
    """
    Save normalization statistics to a file.

    Args:
        norm_stats: NormalizationStats instance
        path: Path to save the stats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats.save(path)


def load_normalization_stats(path: Path | str) -> NormalizationStats:
    """
    Load normalization statistics from a file.

    Args:
        path: Path to load the stats from

    Returns:
        NormalizationStats instance
    """
    norm_stats = NormalizationStats()
    norm_stats.load(Path(path))
    return norm_stats
