# =============================================================================
# Web Scraping Configuration Helper Functions
# =============================================================================
"""
Helper functions and utilities for MaxSold auction data scraping.

This module provides functions to access configuration from a YAML file
and utilities for field transformations, path management, and other
scraping-related operations.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

# Path to YAML config file
_CONFIG_FILE: Path = Path(__file__).parent / "scraper_config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns:
        Dictionary containing all configuration values
    """
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def get_config_value(*keys: str, default: Any = None) -> Any:
    """
    Get a configuration value by nested keys.

    Args:
        *keys: Nested keys to traverse (e.g., 'api', 'base_url')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> get_config_value('api', 'base_url')
        'https://maxsold.maxsold.com/msapi'
    """
    config = load_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config


# =============================================================================
# Configuration Access Functions
# =============================================================================


def get_api_base_url() -> str:
    """Get MaxSold API base URL."""
    return get_config_value("api", "base_url")


def get_api_endpoint(endpoint_name: str) -> str:
    """
    Get full API endpoint URL.

    Args:
        endpoint_name: Name of the endpoint ('auction_items' or 'enriched_item')

    Returns:
        Full endpoint URL
    """
    base_url = get_api_base_url()
    endpoint_path = get_config_value("api", "endpoints", endpoint_name)
    return f"{base_url}{endpoint_path}"


def get_rate_limit_config() -> dict[str, int]:
    """
    Get rate limiting configuration.

    Returns:
        Dictionary with 'requests_per_second' and 'concurrent_requests'
    """
    return {
        "requests_per_second": get_config_value(
            "rate_limiting", "requests_per_second", default=10
        ),
        "concurrent_requests": get_config_value(
            "rate_limiting", "concurrent_requests", default=5
        ),
    }


def get_field_rename_map() -> dict[str, str]:
    """Get field name mapping configuration."""
    return get_config_value("fields", "field_rename_map", default={})


def get_column_prefix() -> str:
    """Get column prefix configuration."""
    return get_config_value("fields", "column_prefix", default="auction_")


def get_item_column_prefix() -> str:
    """Get item column prefix configuration."""
    return get_config_value("fields", "item_column_prefix", default="item_")


def get_item_fields() -> list[str]:
    """Get item-level fields configuration."""
    return get_config_value("fields", "item_fields", default=[])


def get_item_field_rename_map() -> dict[str, str]:
    """Get item field name mapping configuration."""
    return get_config_value("fields", "item_field_rename_map", default={})


# =============================================================================
# Path Management Functions
# =============================================================================


def get_auction_ids_file() -> Path:
    """Get path to auction IDs input file."""
    directory = get_config_value("storage", "input", "directory")
    filename = get_config_value("storage", "input", "file")
    return PROJECT_ROOT / directory / filename


def get_auction_ids_column() -> str:
    """Get name of the column containing auction IDs."""
    return get_config_value("storage", "input", "id_column", default="amAuctionId")


def get_output_directory(processed: bool = True) -> Path:
    """
    Get output directory path.

    Args:
        processed: If True, returns processed directory; otherwise raw directory

    Returns:
        Path to output directory
    """
    key = "processed_directory" if processed else "raw_directory"
    directory = get_config_value("storage", "output", key)
    return PROJECT_ROOT / directory


def get_progress_file() -> Path:
    """Get path to progress tracking file."""
    raw_dir = get_output_directory(processed=False)
    filename = get_config_value(
        "storage", "progress", "filename", default="scraper_progress.json"
    )
    return raw_dir / filename


def get_item_progress_file() -> Path:
    """Get path to item scraper progress tracking file."""
    item_raw_dir = get_item_output_directory(processed=False)
    filename = get_config_value(
        "storage", "progress", "item_progress_filename", default="item_scraper_progress.json"
    )
    return item_raw_dir / filename


def get_item_output_directory(processed: bool = True) -> Path:
    """
    Get item output directory path.

    Args:
        processed: If True, returns processed directory; otherwise raw directory

    Returns:
        Path to output directory
    """
    key = "item_processed_directory" if processed else "item_raw_directory"
    directory = get_config_value("storage", "output", key)
    return PROJECT_ROOT / directory


def get_hf_dataset_repo() -> str:
    """Get Hugging Face dataset repository for loading auction IDs."""
    return get_config_value("storage", "input", "hf_dataset_repo", default="jpearce610/auction_data")


def get_hf_auction_id_column() -> str:
    """Get the column name for auction IDs in the Hugging Face dataset."""
    return get_config_value("storage", "input", "hf_auction_id_column", default="auction_id")


# =============================================================================
# Field Transformation Functions
# =============================================================================


def get_renamed_field_name(field_name: str) -> str:
    """
    Get renamed field name based on configuration mapping.

    Args:
        field_name: Original field name

    Returns:
        Renamed field name (or original if no mapping exists)
    """
    rename_map = get_field_rename_map()
    return rename_map.get(field_name, field_name)


def get_renamed_item_field_name(field_name: str) -> str:
    """
    Get renamed item field name based on configuration mapping.

    Args:
        field_name: Original item field name

    Returns:
        Renamed field name (or original if no mapping exists)
    """
    rename_map = get_item_field_rename_map()
    return rename_map.get(field_name, field_name)


def get_prefixed_field_name(field_name: str) -> str:
    """
    Add configured prefix to field name.

    Args:
        field_name: Original field name

    Returns:
        Prefixed field name
    """
    prefix = get_column_prefix()
    return f"{prefix}{field_name}"


def get_prefixed_item_field_name(field_name: str) -> str:
    """
    Add configured item prefix to field name.

    Args:
        field_name: Original field name

    Returns:
        Prefixed field name with item_ prefix
    """
    prefix = get_item_column_prefix()
    return f"{prefix}{field_name}"


def apply_field_transformations(field_name: str) -> str:
    """
    Apply all field transformations (rename + prefix).

    Args:
        field_name: Original field name

    Returns:
        Fully transformed field name
    """
    # First rename
    renamed = get_renamed_field_name(field_name)
    # Then add prefix
    return get_prefixed_field_name(renamed)


def apply_item_field_transformations(field_name: str) -> str:
    """
    Apply all item field transformations (rename + item_ prefix).

    Args:
        field_name: Original item field name

    Returns:
        Fully transformed field name with item_ prefix
    """
    # First rename
    renamed = get_renamed_item_field_name(field_name)
    # Then add item_ prefix
    return get_prefixed_item_field_name(renamed)


def transform_field_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform all field names in a dictionary.

    Args:
        data: Dictionary with original field names

    Returns:
        Dictionary with transformed field names
    """
    return {apply_field_transformations(k): v for k, v in data.items()}


# =============================================================================
# Validation Functions
# =============================================================================


def validate_auction_fields(data: dict[str, Any]) -> list[str]:
    """
    Check if data contains all required auction fields.

    Args:
        data: Auction data dictionary

    Returns:
        List of missing field names (empty if all present)
    """
    required_fields = get_config_value("fields", "auction_fields", default=[])
    return [field for field in required_fields if field not in data]


def validate_numeric_fields(data: dict[str, Any]) -> list[str]:
    """
    Check if numeric fields have valid numeric values.

    Args:
        data: Data dictionary to validate

    Returns:
        List of field names with invalid values
    """
    numeric_fields = get_config_value("validation", "numeric_fields", default=[])
    invalid = []

    for field in numeric_fields:
        if field in data:
            value = data[field]
            if value is not None and not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    invalid.append(field)

    return invalid


# =============================================================================
# Directory Management Functions
# =============================================================================


def ensure_directories() -> None:
    """Create necessary output directories if they don't exist."""
    raw_dir = get_output_directory(processed=False)
    processed_dir = get_output_directory(processed=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


def get_output_filepath(filename: str | None = None, processed: bool = True) -> Path:
    """
    Get full path for an output file.

    Args:
        filename: Name of the file (uses default if None)
        processed: If True, uses processed directory; otherwise raw directory

    Returns:
        Full path to output file
    """
    directory = get_output_directory(processed=processed)

    if filename is None:
        filename = get_config_value(
            "storage", "output", "auction_data_filename", default="auction_data.parquet"
        )

    return directory / filename


# =============================================================================
# Backwards Compatibility - Module-level Constants
# =============================================================================
# These constants are provided for backwards compatibility with existing code.
# They are loaded lazily from the YAML config via helper functions.

# API Configuration
MAXSOLD_API_BASE_URL = get_api_base_url()
AUCTION_ITEMS_ENDPOINT = get_api_endpoint("auction_items")
ENRICHED_ITEM_ENDPOINT = f"{get_config_value('api', 'enriched_base_url')}{get_config_value('api', 'endpoints', 'enriched_item')}"
DEFAULT_ITEMS_LIMIT = get_config_value("api", "parameters", "items_limit", default=2500)
REQUEST_TIMEOUT = get_config_value("api", "parameters", "timeout", default=30)
MAX_RETRIES = get_config_value("api", "parameters", "max_retries", default=3)
RETRY_DELAY = get_config_value("api", "parameters", "retry_delay", default=1.0)
DEFAULT_RATE_LIMIT = get_config_value(
    "rate_limiting", "requests_per_second", default=10
)
CONCURRENT_REQUESTS = get_config_value(
    "rate_limiting", "concurrent_requests", default=5
)

# Field Configuration
AUCTION_FIELDS = get_config_value("fields", "auction_fields", default=[])
ITEM_AGGREGATE_FIELDS = get_config_value("fields", "item_aggregate_fields", default=[])
ITEM_FIELDS = get_config_value("fields", "item_fields", default=[])
FIELD_RENAME_MAP = get_field_rename_map()
ITEM_FIELD_RENAME_MAP = get_item_field_rename_map()
COLUMN_PREFIX = get_column_prefix()
ITEM_COLUMN_PREFIX = get_item_column_prefix()

# Storage Configuration
AUCTION_IDS_FILE = get_auction_ids_file()
AUCTION_IDS_COLUMN = get_auction_ids_column()
HF_DATASET_REPO = get_hf_dataset_repo()
HF_AUCTION_ID_COLUMN = get_hf_auction_id_column()
RAW_OUTPUT_DIR = get_output_directory(processed=False)
PROCESSED_OUTPUT_DIR = get_output_directory(processed=True)
ITEM_RAW_OUTPUT_DIR = get_item_output_directory(processed=False)
ITEM_PROCESSED_OUTPUT_DIR = get_item_output_directory(processed=True)
AUCTION_DATA_FILENAME = get_config_value(
    "storage", "output", "auction_data_filename", default="auction_data.parquet"
)
ITEM_DATA_FILENAME = get_config_value(
    "storage", "output", "item_data_filename", default="item_data.parquet"
)
METADATA_FILENAME = get_config_value(
    "storage", "output", "metadata_filename", default="metadata.json"
)
PROGRESS_FILE = get_progress_file()
ITEM_PROGRESS_FILE = get_item_progress_file()

# Parallel Processing Configuration
DEFAULT_BATCH_SIZE = get_config_value(
    "parallel_processing", "batch", "default_size", default=100
)
MIN_BATCH_SIZE = get_config_value("parallel_processing", "batch", "min_size", default=1)
MAX_BATCH_SIZE = get_config_value(
    "parallel_processing", "batch", "max_size", default=1000
)
DEFAULT_MAX_WORKERS = get_config_value(
    "parallel_processing", "workers", "default_max", default=5
)
MIN_WORKERS = get_config_value("parallel_processing", "workers", "min", default=1)
MAX_WORKERS = get_config_value("parallel_processing", "workers", "max", default=20)

# Hugging Face Configuration
HF_DATASET_NAME = get_config_value(
    "huggingface", "dataset", "name", default="maxsold-auction-data"
)
HF_DATASET_DESCRIPTION = get_config_value(
    "huggingface", "dataset", "description", default=""
)
HF_DATASET_LICENSE = get_config_value(
    "huggingface", "dataset", "license", default="cc-by-4.0"
)
HF_DATASET_TAGS = get_config_value("huggingface", "dataset", "tags", default=[])
HF_UPLOAD_FILES = get_config_value("huggingface", "upload_files", default=[])

# Item Dataset Configuration
HF_ITEM_DATASET_NAME = get_config_value(
    "huggingface", "item_dataset", "name", default="maxsold-item-data"
)
HF_ITEM_DATASET_DESCRIPTION = get_config_value(
    "huggingface", "item_dataset", "description", default=""
)
HF_ITEM_DATASET_LICENSE = get_config_value(
    "huggingface", "item_dataset", "license", default="cc-by-4.0"
)
HF_ITEM_DATASET_TAGS = get_config_value("huggingface", "item_dataset", "tags", default=[])
HF_ITEM_UPLOAD_FILES = get_config_value("huggingface", "item_upload_files", default=[])

# Validation Configuration
REQUIRED_OUTPUT_FIELDS = get_config_value(
    "validation", "required_output_fields", default=[]
)
NUMERIC_FIELDS = get_config_value("validation", "numeric_fields", default=[])
