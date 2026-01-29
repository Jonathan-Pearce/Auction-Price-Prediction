# =============================================================================
# Hugging Face Bid Data Loader
# =============================================================================
"""
Utilities for loading bid data from Hugging Face datasets.

Provides a clean interface for loading the bid_data dataset programmatically
and converting it to pandas DataFrames for feature engineering.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.feature_engineering.feature_config import FeatureConfig, load_feature_config


class HFBidDataLoader:
    """
    Loader for bid data from Hugging Face datasets.

    This class provides methods to load bid data from the Hugging Face dataset
    and convert it to pandas DataFrames for feature engineering.

    Args:
        config: FeatureConfig instance with data source settings.
                If None, loads from default config file.

    Example:
        >>> loader = HFBidDataLoader()
        >>> df = loader.load_as_dataframe()
        >>> print(df.columns)
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or load_feature_config()
        self._dataset = None

    @property
    def dataset_repo(self) -> str:
        """Get the Hugging Face dataset repository path."""
        return self.config.data_source.hf_dataset_repo

    @property
    def split(self) -> str | None:
        """Get the dataset split to use."""
        return self.config.data_source.split

    @property
    def streaming(self) -> bool:
        """Check if streaming mode is enabled."""
        return self.config.data_source.streaming

    @property
    def dev_limit(self) -> int | None:
        """Get the development limit for rows."""
        return self.config.data_source.dev_limit

    def load_dataset(self) -> Any:
        """
        Load the Hugging Face dataset.

        Returns:
            Hugging Face Dataset object.

        Raises:
            ImportError: If datasets library is not installed.
        """
        from datasets import load_dataset

        logger.info(f"Loading dataset from Hugging Face: {self.dataset_repo}")

        if self.split:
            dataset = load_dataset(
                self.dataset_repo,
                split=self.split,
                streaming=self.streaming,
            )
        else:
            # Load all splits
            dataset = load_dataset(
                self.dataset_repo,
                streaming=self.streaming,
            )
            # If no split specified and we get a DatasetDict, use the first split
            if hasattr(dataset, "keys"):
                first_split = list(dataset.keys())[0]
                logger.info(f"Using split: {first_split}")
                dataset = dataset[first_split]

        self._dataset = dataset
        logger.info("Dataset loaded successfully")
        return dataset

    def load_as_dataframe(self) -> pd.DataFrame:
        """
        Load the dataset as a pandas DataFrame.

        Returns:
            pandas DataFrame with bid data.

        Note:
            For large datasets, consider using streaming mode and
            processing in batches instead.
        """
        if self._dataset is None:
            self.load_dataset()

        logger.info("Converting dataset to pandas DataFrame...")

        if self.streaming:
            # For streaming datasets, iterate and collect
            rows = []
            for i, row in enumerate(self._dataset):
                rows.append(row)
                if self.dev_limit and i >= self.dev_limit - 1:
                    break
            df = pd.DataFrame(rows)
        else:
            # For non-streaming datasets, convert directly
            df = self._dataset.to_pandas()

            if self.dev_limit:
                df = df.head(self.dev_limit)

        logger.info(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")
        return df

    def iter_batches(self, batch_size: int = 10000) -> Any:
        """
        Iterate over the dataset in batches.

        Args:
            batch_size: Number of rows per batch.

        Yields:
            pandas DataFrame for each batch.
        """
        if self._dataset is None:
            self.load_dataset()

        if self.streaming:
            batch = []
            for i, row in enumerate(self._dataset):
                batch.append(row)
                if len(batch) >= batch_size:
                    yield pd.DataFrame(batch)
                    batch = []
                if self.dev_limit and i >= self.dev_limit - 1:
                    break
            if batch:
                yield pd.DataFrame(batch)
        else:
            df = self._dataset.to_pandas()
            if self.dev_limit:
                df = df.head(self.dev_limit)
            for start in range(0, len(df), batch_size):
                yield df.iloc[start : start + batch_size]

    def get_unique_items(self) -> pd.DataFrame:
        """
        Get unique item IDs and auction IDs from the dataset.

        Returns:
            DataFrame with unique (auction_id, item_id) pairs.
        """
        df = self.load_as_dataframe()
        unique_items = df[["auction_id", "item_id"]].drop_duplicates()
        logger.info(f"Found {len(unique_items):,} unique items")
        return unique_items

    def get_item_count(self) -> int:
        """
        Get the number of unique items in the dataset.

        Returns:
            Number of unique items.
        """
        df = self.load_as_dataframe()
        return df["item_id"].nunique()


def load_bid_data(
    config: FeatureConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Convenience function to load bid data from Hugging Face.

    Args:
        config: FeatureConfig instance. If None, loads from default config.
        limit: Optional limit on number of rows to load (overrides config).

    Returns:
        pandas DataFrame with bid data.

    Example:
        >>> df = load_bid_data(limit=1000)
        >>> print(df.head())
    """
    loader = HFBidDataLoader(config)

    if limit is not None:
        # Override the dev_limit if a limit is provided
        loader.config.data_source.dev_limit = limit

    return loader.load_as_dataframe()
