# =============================================================================
# Feature Engineering Configuration Loader
# =============================================================================
"""
Configuration management for feature engineering.

Provides functions to load and access feature engineering configuration
from a central YAML file, following the pattern used in scraper_config.py.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

# Path to YAML config file
_CONFIG_FILE: Path = Path(__file__).parent / "feature_config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns:
        Dictionary containing all configuration values.
    """
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def get_config_value(*keys: str, default: Any = None) -> Any:
    """
    Get a configuration value by nested keys.

    Args:
        *keys: Nested keys to traverse (e.g., 'features', 'bid_amount', 'enabled')
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> get_config_value('data_source', 'hf_dataset_repo')
        'jpearce610/bid_data'
    """
    config = load_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class TargetConfig:
    """Configuration for target/label variable."""

    name: str = "winning_price"
    aggregation: str = "max"


@dataclass
class BidAmountConfig:
    """Configuration for bid amount features."""

    enabled: bool = True
    statistics: list[str] = field(
        default_factory=lambda: ["max", "min", "mean", "median", "std"]
    )
    derived: list[str] = field(
        default_factory=lambda: ["range", "coefficient_of_variation"]
    )


@dataclass
class BidCountConfig:
    """Configuration for bid count features."""

    enabled: bool = True
    features: list[str] = field(
        default_factory=lambda: ["total_bids", "unique_bidders_proxy"]
    )


@dataclass
class TimeFeatureConfig:
    """Configuration for time-based features."""

    enabled: bool = True
    features: list[str] = field(
        default_factory=lambda: [
            "bidding_duration_seconds",
            "first_last_bid_delta",
            "mean_time_between_bids",
            "median_time_between_bids",
        ]
    )


@dataclass
class LastNBidsConfig:
    """Configuration for last N bids features."""

    enabled: bool = True
    n_values: list[int] = field(default_factory=lambda: [3, 5, 10])
    features: list[str] = field(
        default_factory=lambda: ["mean_amount", "amount_growth_rate"]
    )


@dataclass
class DistributionFeatureConfig:
    """Configuration for distribution/concentration features."""

    enabled: bool = True
    features: list[str] = field(
        default_factory=lambda: [
            "bid_concentration_last_25pct",
            "bid_concentration_last_10pct",
        ]
    )
    last_n_bids: LastNBidsConfig = field(default_factory=LastNBidsConfig)


@dataclass
class ProxyFeatureConfig:
    """Configuration for proxy bid features."""

    enabled: bool = True
    features: list[str] = field(
        default_factory=lambda: ["proxy_bid_count", "proxy_bid_ratio"]
    )


@dataclass
class MissingValueConfig:
    """Configuration for missing value handling."""

    numeric_strategy: str = "zero"
    fill_value: float = 0.0
    create_indicators: bool = True


@dataclass
class NormalizationConfig:
    """Configuration for feature normalization."""

    enabled: bool = False
    method: str = "standard"
    exclude_features: list[str] = field(
        default_factory=lambda: ["item_id", "auction_id", "winning_price"]
    )


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    remove_zero_variance: bool = True
    remove_highly_correlated: bool = False
    correlation_threshold: float = 0.95


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""

    missing_values: MissingValueConfig = field(default_factory=MissingValueConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )


@dataclass
class DataSourceConfig:
    """Configuration for data source."""

    hf_dataset_repo: str = "jpearce610/bid_data"
    split: str | None = None
    streaming: bool = False
    dev_limit: int | None = None


@dataclass
class OutputConfig:
    """Configuration for output."""

    format: str = "parquet"
    directory: str = "data/processed/features"
    filename: str = "tabular_features.parquet"
    include_metadata: bool = True
    metadata_filename: str = "feature_metadata.json"


@dataclass
class ProcessingConfig:
    """Configuration for processing."""

    batch_size: int = 10000
    n_workers: int = 0
    show_progress: bool = True
    random_seed: int = 42


@dataclass
class FeaturesConfig:
    """Configuration for all feature groups."""

    target: TargetConfig = field(default_factory=TargetConfig)
    bid_amount: BidAmountConfig = field(default_factory=BidAmountConfig)
    bid_count: BidCountConfig = field(default_factory=BidCountConfig)
    time_features: TimeFeatureConfig = field(default_factory=TimeFeatureConfig)
    distribution_features: DistributionFeatureConfig = field(
        default_factory=DistributionFeatureConfig
    )
    proxy_features: ProxyFeatureConfig = field(default_factory=ProxyFeatureConfig)


@dataclass
class FeatureConfig:
    """Main configuration class for feature engineering."""

    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _parse_last_n_bids_config(config_dict: dict[str, Any]) -> LastNBidsConfig:
    """Parse last N bids configuration from dictionary."""
    if not config_dict:
        return LastNBidsConfig()
    return LastNBidsConfig(
        enabled=config_dict.get("enabled", True),
        n_values=config_dict.get("n_values", [3, 5, 10]),
        features=config_dict.get("features", ["mean_amount", "amount_growth_rate"]),
    )


def _parse_distribution_config(
    config_dict: dict[str, Any],
) -> DistributionFeatureConfig:
    """Parse distribution feature configuration from dictionary."""
    if not config_dict:
        return DistributionFeatureConfig()
    return DistributionFeatureConfig(
        enabled=config_dict.get("enabled", True),
        features=config_dict.get(
            "features",
            ["bid_concentration_last_25pct", "bid_concentration_last_10pct"],
        ),
        last_n_bids=_parse_last_n_bids_config(config_dict.get("last_n_bids", {})),
    )


def _parse_features_config(config_dict: dict[str, Any]) -> FeaturesConfig:
    """Parse features configuration from dictionary."""
    if not config_dict:
        return FeaturesConfig()

    target_dict = config_dict.get("target", {})
    bid_amount_dict = config_dict.get("bid_amount", {})
    bid_count_dict = config_dict.get("bid_count", {})
    time_dict = config_dict.get("time_features", {})
    dist_dict = config_dict.get("distribution_features", {})
    proxy_dict = config_dict.get("proxy_features", {})

    return FeaturesConfig(
        target=TargetConfig(
            name=target_dict.get("name", "winning_price"),
            aggregation=target_dict.get("aggregation", "max"),
        ),
        bid_amount=BidAmountConfig(
            enabled=bid_amount_dict.get("enabled", True),
            statistics=bid_amount_dict.get(
                "statistics", ["max", "min", "mean", "median", "std"]
            ),
            derived=bid_amount_dict.get(
                "derived", ["range", "coefficient_of_variation"]
            ),
        ),
        bid_count=BidCountConfig(
            enabled=bid_count_dict.get("enabled", True),
            features=bid_count_dict.get(
                "features", ["total_bids", "unique_bidders_proxy"]
            ),
        ),
        time_features=TimeFeatureConfig(
            enabled=time_dict.get("enabled", True),
            features=time_dict.get(
                "features",
                [
                    "bidding_duration_seconds",
                    "first_last_bid_delta",
                    "mean_time_between_bids",
                    "median_time_between_bids",
                ],
            ),
        ),
        distribution_features=_parse_distribution_config(dist_dict),
        proxy_features=ProxyFeatureConfig(
            enabled=proxy_dict.get("enabled", True),
            features=proxy_dict.get("features", ["proxy_bid_count", "proxy_bid_ratio"]),
        ),
    )


def _parse_preprocessing_config(config_dict: dict[str, Any]) -> PreprocessingConfig:
    """Parse preprocessing configuration from dictionary."""
    if not config_dict:
        return PreprocessingConfig()

    missing_dict = config_dict.get("missing_values", {})
    norm_dict = config_dict.get("normalization", {})
    selection_dict = config_dict.get("feature_selection", {})

    return PreprocessingConfig(
        missing_values=MissingValueConfig(
            numeric_strategy=missing_dict.get("numeric_strategy", "zero"),
            fill_value=missing_dict.get("fill_value", 0.0),
            create_indicators=missing_dict.get("create_indicators", True),
        ),
        normalization=NormalizationConfig(
            enabled=norm_dict.get("enabled", False),
            method=norm_dict.get("method", "standard"),
            exclude_features=norm_dict.get(
                "exclude_features", ["item_id", "auction_id", "winning_price"]
            ),
        ),
        feature_selection=FeatureSelectionConfig(
            remove_zero_variance=selection_dict.get("remove_zero_variance", True),
            remove_highly_correlated=selection_dict.get(
                "remove_highly_correlated", False
            ),
            correlation_threshold=selection_dict.get("correlation_threshold", 0.95),
        ),
    )


def load_feature_config(config_path: Path | str | None = None) -> FeatureConfig:
    """
    Load feature engineering configuration from YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses default path.

    Returns:
        FeatureConfig dataclass with all configuration values.
    """
    if config_path is not None:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = load_config()

    # Parse data source config
    data_source_dict = config_dict.get("data_source", {})
    data_source = DataSourceConfig(
        hf_dataset_repo=data_source_dict.get("hf_dataset_repo", "jpearce610/bid_data"),
        split=data_source_dict.get("split"),
        streaming=data_source_dict.get("streaming", False),
        dev_limit=data_source_dict.get("dev_limit"),
    )

    # Parse output config
    output_dict = config_dict.get("output", {})
    output = OutputConfig(
        format=output_dict.get("format", "parquet"),
        directory=output_dict.get("directory", "data/processed/features"),
        filename=output_dict.get("filename", "tabular_features.parquet"),
        include_metadata=output_dict.get("include_metadata", True),
        metadata_filename=output_dict.get("metadata_filename", "feature_metadata.json"),
    )

    # Parse processing config
    processing_dict = config_dict.get("processing", {})
    processing = ProcessingConfig(
        batch_size=processing_dict.get("batch_size", 10000),
        n_workers=processing_dict.get("n_workers", 0),
        show_progress=processing_dict.get("show_progress", True),
        random_seed=processing_dict.get("random_seed", 42),
    )

    return FeatureConfig(
        data_source=data_source,
        features=_parse_features_config(config_dict.get("features", {})),
        preprocessing=_parse_preprocessing_config(config_dict.get("preprocessing", {})),
        output=output,
        processing=processing,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_hf_dataset_repo() -> str:
    """Get Hugging Face dataset repository path."""
    return get_config_value(
        "data_source", "hf_dataset_repo", default="jpearce610/bid_data"
    )


def get_output_directory() -> Path:
    """Get output directory path."""
    directory = get_config_value(
        "output", "directory", default="data/processed/features"
    )
    return PROJECT_ROOT / directory


def get_enabled_feature_groups() -> list[str]:
    """Get list of enabled feature groups."""
    enabled = []
    features = get_config_value("features", default={})

    if features.get("bid_amount", {}).get("enabled", True):
        enabled.append("bid_amount")
    if features.get("bid_count", {}).get("enabled", True):
        enabled.append("bid_count")
    if features.get("time_features", {}).get("enabled", True):
        enabled.append("time_features")
    if features.get("distribution_features", {}).get("enabled", True):
        enabled.append("distribution_features")
    if features.get("proxy_features", {}).get("enabled", True):
        enabled.append("proxy_features")

    return enabled


def is_normalization_enabled() -> bool:
    """Check if normalization is enabled."""
    return get_config_value("preprocessing", "normalization", "enabled", default=False)


def get_normalization_method() -> str:
    """Get normalization method."""
    return get_config_value(
        "preprocessing", "normalization", "method", default="standard"
    )
