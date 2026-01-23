# =============================================================================
# Web Scraping Configuration
# =============================================================================
"""
Configuration constants for MaxSold auction data scraping.

This module contains all configuration values for the web scraping process,
including API endpoints, field mappings, and scraping parameters.
"""

from pathlib import Path
from typing import Final

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# =============================================================================
# API Configuration
# =============================================================================

# MaxSold API base URLs
MAXSOLD_API_BASE_URL: Final[str] = "https://maxsold.maxsold.com/msapi"
MAXSOLD_ENRICHED_API_BASE_URL: Final[str] = "https://api.maxsold.com"

# API endpoints
AUCTION_ITEMS_ENDPOINT: Final[str] = f"{MAXSOLD_API_BASE_URL}/auctions/items"
ENRICHED_ITEM_ENDPOINT: Final[str] = f"{MAXSOLD_ENRICHED_API_BASE_URL}/listings/am/{{item_id}}/enriched"

# API parameters
DEFAULT_ITEMS_LIMIT: Final[int] = 2500
REQUEST_TIMEOUT: Final[int] = 30  # seconds
MAX_RETRIES: Final[int] = 3
RETRY_DELAY: Final[float] = 1.0  # seconds

# Rate limiting
DEFAULT_RATE_LIMIT: Final[int] = 10  # requests per second
CONCURRENT_REQUESTS: Final[int] = 5  # maximum concurrent auction fetches

# =============================================================================
# Data Field Configuration
# =============================================================================

# Auction-level fields to extract from API response
AUCTION_FIELDS: Final[list[str]] = [
    "id",
    "title",
    "starts",
    "ends",
    "last_item_closes",
    "removal_info",
    "intro",
    "pickup_time",
    "partner_url",
    "extended_bidding",
    "extended_bidding_interval",
    "extended_bidding_threshold",
    "catalog_lots",
]

# Item-level fields to aggregate
ITEM_AGGREGATE_FIELDS: Final[list[str]] = [
    "viewed",
    "current_bid",
    "bid_count",
]

# Field name mappings (old_name -> new_name)
FIELD_RENAME_MAP: Final[dict[str, str]] = {
    "catalog_lots": "item_count",
    "current_bid": "winning_price",
    "id": "id",  # Keep id as-is initially, will add prefix later
}

# Prefix to add to all column names
COLUMN_PREFIX: Final[str] = "auction_"

# Fields that should be excluded from prefix (internal use only)
EXCLUDE_FROM_PREFIX: Final[set[str]] = set()

# =============================================================================
# Data Storage Configuration
# =============================================================================

# Input data
AUCTION_IDS_FILE: Final[Path] = RAW_DATA_DIR / "auction_location_data.parquet"
AUCTION_IDS_COLUMN: Final[str] = "amAuctionId"

# Output data
RAW_OUTPUT_DIR: Final[Path] = RAW_DATA_DIR / "auctions"
PROCESSED_OUTPUT_DIR: Final[Path] = PROCESSED_DATA_DIR / "auctions"
AUCTION_DATA_FILENAME: Final[str] = "auction_data.parquet"
METADATA_FILENAME: Final[str] = "metadata.json"

# Progress tracking
PROGRESS_FILE: Final[Path] = RAW_OUTPUT_DIR / "scraper_progress.json"

# =============================================================================
# Parallel Processing Configuration
# =============================================================================

# Batch processing
DEFAULT_BATCH_SIZE: Final[int] = 100
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 1000

# Parallel workers
DEFAULT_MAX_WORKERS: Final[int] = 5
MIN_WORKERS: Final[int] = 1
MAX_WORKERS: Final[int] = 20

# =============================================================================
# Hugging Face Configuration
# =============================================================================

# Dataset metadata
HF_DATASET_NAME: Final[str] = "maxsold-auction-data"
HF_DATASET_DESCRIPTION: Final[str] = "MaxSold auction data with aggregated item metrics"
HF_DATASET_LICENSE: Final[str] = "cc-by-4.0"
HF_DATASET_TAGS: Final[list[str]] = [
    "auction",
    "price-prediction",
    "e-commerce",
    "maxsold",
]

# Files to upload to Hugging Face
HF_UPLOAD_FILES: Final[list[str]] = [
    AUCTION_DATA_FILENAME,
    METADATA_FILENAME,
]

# =============================================================================
# Validation Configuration
# =============================================================================

# Required fields in final dataset (after renaming and prefixing)
REQUIRED_OUTPUT_FIELDS: Final[list[str]] = [
    "auction_id",
    "auction_title",
    "auction_item_count",
    "auction_total_viewed",
    "auction_total_winning_price",
    "auction_total_bid_count",
    "auction_total_images",
]

# Data type validation
NUMERIC_FIELDS: Final[list[str]] = [
    "item_count",
    "total_viewed",
    "total_winning_price",
    "total_bid_count",
    "total_images",
    "extended_bidding_interval",
    "extended_bidding_threshold",
]

# =============================================================================
# Helper Functions
# =============================================================================


def get_prefixed_field_name(field_name: str) -> str:
    """
    Add auction_ prefix to field name if not excluded.
    
    Args:
        field_name: Original field name
        
    Returns:
        Prefixed field name
    """
    if field_name in EXCLUDE_FROM_PREFIX:
        return field_name
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
