# =============================================================================
# Preprocessing Utilities
# =============================================================================
"""
Preprocessing utilities for feature engineering.

Provides functions for:
- Missing value handling
- Feature normalization/standardization
- Feature selection
- Preparing final tabular datasets
"""

import json
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT
from src.feature_engineering.feature_config import FeatureConfig, load_feature_config


def handle_missing_values(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Handle missing values in the feature DataFrame.

    Args:
        df: DataFrame with features.
        config: FeatureConfig instance. If None, loads from default config.
        exclude_columns: Columns to exclude from missing value handling.

    Returns:
        DataFrame with missing values handled.
    """
    config = config or load_feature_config()
    missing_config = config.preprocessing.missing_values

    df = df.copy()
    exclude_columns = exclude_columns or ["auction_id", "item_id"]

    # Get numeric columns (excluding identifiers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    # Check for missing values
    missing_counts = df[numeric_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        logger.info(f"Handling missing values in {len(cols_with_missing)} columns")

        # Create indicators if configured
        if missing_config.create_indicators:
            for col in cols_with_missing.index:
                indicator_col = f"{col}_missing"
                df[indicator_col] = df[col].isnull().astype(int)
                logger.debug(f"Created missing indicator: {indicator_col}")

        # Fill missing values based on strategy
        strategy = missing_config.numeric_strategy

        if strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0.0)
        elif strategy == "mean":
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == "constant":
            df[numeric_cols] = df[numeric_cols].fillna(missing_config.fill_value)
        else:
            logger.warning(f"Unknown missing value strategy: {strategy}, using zero")
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

        logger.info(f"Filled missing values using strategy: {strategy}")
    else:
        logger.info("No missing values found in numeric columns")

    return df


def normalize_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    fit: bool = True,
    scaler_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Normalize/standardize numeric features.

    Args:
        df: DataFrame with features.
        config: FeatureConfig instance. If None, loads from default config.
        fit: Whether to fit the scaler (True for training, False for inference).
        scaler_params: Pre-fitted scaler parameters (mean, std, min, max).

    Returns:
        Tuple of (normalized DataFrame, scaler parameters).
    """
    config = config or load_feature_config()
    norm_config = config.preprocessing.normalization

    if not norm_config.enabled:
        logger.info("Normalization is disabled, returning original DataFrame")
        return df, {}

    df = df.copy()
    exclude_cols = norm_config.exclude_features

    # Get numeric columns to normalize
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_normalize:
        logger.warning("No columns to normalize")
        return df, {}

    method = norm_config.method
    params = scaler_params or {}

    # Validate scaler_params when fit=False (inference mode)
    if not fit:
        if not scaler_params:
            raise ValueError(
                "scaler_params must be provided when fit=False (inference mode)"
            )
        required_keys = {
            "standard": ["mean", "std"],
            "minmax": ["min", "max"],
            "robust": ["median", "iqr"],
        }
        if method in required_keys:
            missing_keys = [k for k in required_keys[method] if k not in scaler_params]
            if missing_keys:
                raise ValueError(
                    f"scaler_params missing required keys for {method} method: {missing_keys}"
                )

    if method == "standard":
        # Z-score normalization: (x - mean) / std
        if fit:
            params["mean"] = df[cols_to_normalize].mean().to_dict()
            params["std"] = df[cols_to_normalize].std().to_dict()

        for col in cols_to_normalize:
            mean = params["mean"].get(col, 0.0)
            std = params["std"].get(col, 1.0)
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = df[col] - mean

    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        if fit:
            params["min"] = df[cols_to_normalize].min().to_dict()
            params["max"] = df[cols_to_normalize].max().to_dict()

        for col in cols_to_normalize:
            min_val = params["min"].get(col, 0.0)
            max_val = params["max"].get(col, 1.0)
            range_val = max_val - min_val
            if range_val > 0:
                df[col] = (df[col] - min_val) / range_val
            else:
                df[col] = 0.0

    elif method == "robust":
        # Robust normalization: (x - median) / IQR
        if fit:
            params["median"] = df[cols_to_normalize].median().to_dict()
            q1 = df[cols_to_normalize].quantile(0.25).to_dict()
            q3 = df[cols_to_normalize].quantile(0.75).to_dict()
            params["iqr"] = {col: q3[col] - q1[col] for col in cols_to_normalize}

        for col in cols_to_normalize:
            median = params["median"].get(col, 0.0)
            iqr = params["iqr"].get(col, 1.0)
            if iqr > 0:
                df[col] = (df[col] - median) / iqr
            else:
                df[col] = df[col] - median

    else:
        logger.warning(f"Unknown normalization method: {method}")
        return df, {}

    params["method"] = method
    params["columns"] = cols_to_normalize

    logger.info(f"Normalized {len(cols_to_normalize)} features using {method} method")
    return df, params


def select_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Perform feature selection based on configuration.

    Args:
        df: DataFrame with features.
        config: FeatureConfig instance. If None, loads from default config.
        exclude_columns: Columns to exclude from feature selection.

    Returns:
        DataFrame with selected features.
    """
    config = config or load_feature_config()
    selection_config = config.preprocessing.feature_selection

    df = df.copy()
    exclude_columns = exclude_columns or ["auction_id", "item_id", "winning_price"]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_columns]
    original_count = len(feature_cols)

    # Remove zero variance features
    if selection_config.remove_zero_variance:
        variances = df[feature_cols].var()
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            logger.info(f"Removing {len(zero_var_cols)} zero-variance features")
            df = df.drop(columns=zero_var_cols)
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]

    # Remove highly correlated features
    if selection_config.remove_highly_correlated:
        corr_threshold = selection_config.correlation_threshold
        corr_matrix = df[feature_cols].corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns with correlation above threshold
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]

        if to_drop:
            logger.info(
                f"Removing {len(to_drop)} highly correlated features "
                f"(threshold={corr_threshold})"
            )
            df = df.drop(columns=to_drop)

    final_count = len([col for col in df.columns if col not in exclude_columns])
    logger.info(f"Feature selection: {original_count} -> {final_count} features")

    return df


def prepare_tabular_dataset(
    bid_df: pd.DataFrame,
    config: FeatureConfig | None = None,
    save_to_disk: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare a complete tabular dataset from raw bid data.

    This function performs the full feature engineering pipeline:
    1. Extract features from bid data
    2. Handle missing values
    3. Normalize features (if enabled)
    4. Select features (if enabled)
    5. Optionally save to disk

    Args:
        bid_df: DataFrame with raw bid data.
        config: FeatureConfig instance. If None, loads from default config.
        save_to_disk: Whether to save the result to disk.

    Returns:
        Tuple of (feature DataFrame, metadata dict).

    Example:
        >>> from src.feature_engineering import load_bid_data, prepare_tabular_dataset
        >>> bid_df = load_bid_data(limit=10000)
        >>> features_df, metadata = prepare_tabular_dataset(bid_df)
    """
    from src.feature_engineering.bid_features import BidFeatureExtractor

    config = config or load_feature_config()

    logger.info("Starting tabular dataset preparation...")

    # Step 1: Extract features
    extractor = BidFeatureExtractor(config)
    features_df = extractor.extract_features(bid_df)

    # Step 2: Handle missing values
    features_df = handle_missing_values(features_df, config)

    # Step 3: Normalize features
    features_df, scaler_params = normalize_features(features_df, config)

    # Step 4: Feature selection
    features_df = select_features(features_df, config)

    # Build metadata
    # Count non-feature columns (identifiers and target)
    non_feature_cols = ["auction_id", "item_id", "winning_price"]
    n_feature_cols = len(
        [col for col in features_df.columns if col not in non_feature_cols]
    )
    metadata = {
        "n_items": len(features_df),
        "n_features": n_feature_cols,
        "feature_names": extractor.get_feature_names(),
        "feature_metadata": extractor.get_feature_metadata(),
        "scaler_params": scaler_params,
        "config": {
            "normalization_enabled": config.preprocessing.normalization.enabled,
            "normalization_method": config.preprocessing.normalization.method,
            "missing_value_strategy": config.preprocessing.missing_values.numeric_strategy,
        },
    }

    # Step 5: Save to disk if requested
    if save_to_disk:
        output_dir = PROJECT_ROOT / config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        output_path = output_dir / config.output.filename
        output_format = config.output.format

        if output_format == "parquet":
            features_df.to_parquet(output_path, index=False)
        elif output_format == "csv":
            features_df.to_csv(output_path, index=False)
        elif output_format == "feather":
            features_df.to_feather(output_path)
        else:
            logger.warning(f"Unknown format {output_format}, using parquet")
            features_df.to_parquet(output_path, index=False)

        logger.info(f"Saved features to {output_path}")

        # Save metadata
        if config.output.include_metadata:
            metadata_path = output_dir / config.output.metadata_filename
            with open(metadata_path, "w") as f:
                # Convert numpy types to Python types for JSON serialization
                serializable_metadata = _make_json_serializable(metadata)
                json.dump(serializable_metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

    logger.info("Tabular dataset preparation complete!")
    return features_df, metadata


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj


def get_train_test_split(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features into train and test sets.

    Args:
        features_df: DataFrame with features.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        stratify_col: Column to stratify by (optional).

    Returns:
        Tuple of (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    stratify = features_df[stratify_col] if stratify_col else None

    train_df, test_df = train_test_split(
        features_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(f"Split data: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df


def get_feature_target_split(
    features_df: pd.DataFrame,
    target_col: str = "winning_price",
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Args:
        features_df: DataFrame with features and target.
        target_col: Name of target column.
        exclude_cols: Additional columns to exclude from features.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    exclude_cols = exclude_cols or []
    all_exclude = ["auction_id", "item_id", target_col] + exclude_cols

    feature_cols = [col for col in features_df.columns if col not in all_exclude]

    X = features_df[feature_cols]
    y = features_df[target_col]

    logger.info(f"Feature/target split: X shape={X.shape}, y shape={y.shape}")
    return X, y
