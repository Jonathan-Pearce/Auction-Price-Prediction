# =============================================================================
# Web Scraping Configuration
# =============================================================================
"""
Configuration loader for MaxSold auction data scraping.

This module loads configuration from a YAML file and provides constants
for the web scraping process, including API endpoints, field mappings,
and scraping parameters.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from src.config import PROJECT_ROOT

# Path to YAML config file
CONFIG_FILE: Final[Path] = Path(__file__).parent / "scraper_config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


# Load config once
_config = _load_config()

# =============================================================================
# API Configuration
# =============================================================================

# MaxSold API base URLs
MAXSOLD_API_BASE_URL: Final[str] = _config["api"]["base_url"]
MAXSOLD_ENRICHED_API_BASE_URL: Final[str] = _config["api"]["enriched_base_url"]

# API endpoints
AUCTION_ITEMS_ENDPOINT: Final[str] = (
    f"{MAXSOLD_API_BASE_URL}{_config['api']['endpoints']['auction_items']}"
)
ENRICHED_ITEM_ENDPOINT: Final[str] = (
    f"{MAXSOLD_ENRICHED_API_BASE_URL}{_config['api']['endpoints']['enriched_item']}"
)

# API parameters
DEFAULT_ITEMS_LIMIT: Final[int] = _config["api"]["parameters"]["items_limit"]
REQUEST_TIMEOUT: Final[int] = _config["api"]["parameters"]["timeout"]
MAX_RETRIES: Final[int] = _config["api"]["parameters"]["max_retries"]
RETRY_DELAY: Final[float] = _config["api"]["parameters"]["retry_delay"]

# Rate limiting
DEFAULT_RATE_LIMIT: Final[int] = _config["rate_limiting"]["requests_per_second"]
CONCURRENT_REQUESTS: Final[int] = _config["rate_limiting"]["concurrent_requests"]

# =============================================================================
# Data Field Configuration
# =============================================================================

# Auction-level fields to extract from API response
AUCTION_FIELDS: Final[list[str]] = _config["fields"]["auction_fields"]

# Item-level fields to aggregate
ITEM_AGGREGATE_FIELDS: Final[list[str]] = _config["fields"]["item_aggregate_fields"]

# Field name mappings (old_name -> new_name)
FIELD_RENAME_MAP: Final[dict[str, str]] = _config["fields"]["field_rename_map"]

# Prefix to add to all column names
COLUMN_PREFIX: Final[str] = _config["fields"]["column_prefix"]

# =============================================================================
# Data Storage Configuration
# =============================================================================

# Input data
AUCTION_IDS_FILE: Final[Path] = (
    PROJECT_ROOT
    / _config["storage"]["input"]["directory"]
    / _config["storage"]["input"]["file"]
)
AUCTION_IDS_COLUMN: Final[str] = _config["storage"]["input"]["id_column"]

# Output data
RAW_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / _config["storage"]["output"]["raw_directory"]
)
PROCESSED_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / _config["storage"]["output"]["processed_directory"]
)
AUCTION_DATA_FILENAME: Final[str] = _config["storage"]["output"][
    "auction_data_filename"
]
METADATA_FILENAME: Final[str] = _config["storage"]["output"]["metadata_filename"]

# Progress tracking
PROGRESS_FILE: Final[Path] = RAW_OUTPUT_DIR / _config["storage"]["progress"]["filename"]

# =============================================================================
# Parallel Processing Configuration
# =============================================================================

# Batch processing
DEFAULT_BATCH_SIZE: Final[int] = _config["parallel_processing"]["batch"]["default_size"]
MIN_BATCH_SIZE: Final[int] = _config["parallel_processing"]["batch"]["min_size"]
MAX_BATCH_SIZE: Final[int] = _config["parallel_processing"]["batch"]["max_size"]

# Parallel workers
DEFAULT_MAX_WORKERS: Final[int] = _config["parallel_processing"]["workers"][
    "default_max"
]
MIN_WORKERS: Final[int] = _config["parallel_processing"]["workers"]["min"]
MAX_WORKERS: Final[int] = _config["parallel_processing"]["workers"]["max"]

# =============================================================================
# Hugging Face Configuration
# =============================================================================

# Dataset metadata
HF_DATASET_NAME: Final[str] = _config["huggingface"]["dataset"]["name"]
HF_DATASET_DESCRIPTION: Final[str] = _config["huggingface"]["dataset"]["description"]
HF_DATASET_LICENSE: Final[str] = _config["huggingface"]["dataset"]["license"]
HF_DATASET_TAGS: Final[list[str]] = _config["huggingface"]["dataset"]["tags"]

# Files to upload to Hugging Face
HF_UPLOAD_FILES: Final[list[str]] = _config["huggingface"]["upload_files"]

# =============================================================================
# Validation Configuration
# =============================================================================

# Required fields in final dataset (after renaming and prefixing)
REQUIRED_OUTPUT_FIELDS: Final[list[str]] = _config["validation"][
    "required_output_fields"
]

# Data type validation
NUMERIC_FIELDS: Final[list[str]] = _config["validation"]["numeric_fields"]

# =============================================================================
# Helper Functions
# =============================================================================


def get_prefixed_field_name(field_name: str) -> str:
    """
    Add auction_ prefix to field name.

    Args:
        field_name: Original field name

    Returns:
        Prefixed field name
    """
    return f"{COLUMN_PREFIX}{field_name}"


def get_renamed_field_name(field_name: str) -> str:
    """
    Get renamed field name based on mapping.

    Args:
        field_name: Original field name

    Returns:
        Renamed field name (or original if no mapping exists)
    """
    return FIELD_RENAME_MAP.get(field_name, field_name)


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


# =============================================================================
# Initialization
# =============================================================================


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
