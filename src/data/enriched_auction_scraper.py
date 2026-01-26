# =============================================================================
# Enriched Auction Data Scraper
# =============================================================================
"""
Scraper for enriched auction-level data with location and category information.

This module implements the requirements from the issue:
1. Scrapes enriched auction data from MaxSold API (https://api.maxsold.com/sales/am/{auction_id})
2. Loads auction IDs from Hugging Face dataset (jpearce610/auction_data)
3. Extracts specified fields (amAuctionId, type, category, displayRegion, approxLocation)
4. Renames fields (amAuctionId→auction_id, amLotId→item_id)
5. Adds 'enriched_auction_' prefix to all column names (except auction_id and item_id)
6. Uses parallel processing for faster scraping
7. Uploads to Hugging Face

Usage:
    # From command line
    python -m src.data.enriched_auction_scraper --limit 100

    # From Python
    from src.data.enriched_auction_scraper import scrape_enriched_auctions
    df = await scrape_enriched_auctions(auction_ids=[99941, 99942])
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data import scraper_config as config

# =============================================================================
# Enriched Auction Data Fetcher
# =============================================================================


class EnrichedAuctionDataFetcher:
    """
    Fetches and processes enriched auction data from MaxSold API.

    Features:
    - Async HTTP client with rate limiting
    - Automatic retries with exponential backoff
    - Parallel processing with controlled concurrency
    """

    def __init__(
        self,
        rate_limit: int = config.DEFAULT_RATE_LIMIT,
        max_concurrent: int = config.CONCURRENT_REQUESTS,
        timeout: float = config.REQUEST_TIMEOUT,
    ):
        """
        Initialize the enriched auction data fetcher.

        Args:
            rate_limit: Maximum requests per second
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.min_interval = 1.0 / rate_limit
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EnrichedAuctionDataFetcher":
        """Initialize async HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "AuctionPricePredictor/1.0 (Research Project)",
                "Accept": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with EnrichedAuctionDataFetcher():'"
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)

            self.last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=config.RETRY_DELAY, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def fetch_enriched_auction(self, auction_id: int) -> dict[str, Any]:
        """
        Fetch enriched auction data from MaxSold API.

        Args:
            auction_id: Auction ID to fetch

        Returns:
            Dictionary containing enriched auction data
        """
        await self._rate_limit()

        # Build URL with auction_id
        url = config.ENRICHED_AUCTION_ENDPOINT.replace("{auction_id}", str(auction_id))

        logger.debug(f"Fetching enriched data for auction {auction_id}")

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error for auction {auction_id}: {e.response.status_code}"
            )
            raise
        except Exception as e:
            logger.error(f"Error fetching enriched auction {auction_id}: {e}")
            raise

    def process_enriched_auction_data(
        self, auction_id: int, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process raw API response into enriched auction data.

        Extracts fields of interest and handles nested structures like approxLocation.

        Args:
            auction_id: Auction ID
            data: Raw API response

        Returns:
            Dictionary with enriched auction data (flattened)
        """
        result = {}

        # Add auction_id first (this will not get the prefix)
        result["auction_id"] = auction_id

        # Extract configured fields
        for field in config.ENRICHED_AUCTION_FIELDS:
            if field == "amAuctionId":
                # Already handled as auction_id above
                continue
            elif field == "approxLocation":
                # Handle nested location structure
                location = data.get("approxLocation", {})
                if isinstance(location, dict):
                    # Extract location subfields
                    result["approxLocation_city"] = location.get("city")
                    result["approxLocation_countryCode"] = location.get("countryCode")
                    result["approxLocation_regionCode"] = location.get("regionCode")
                    result["approxLocation_postalCode"] = location.get("postalCode")

                    # Extract lat/lng from latLng
                    lat_lng = location.get("latLng", {})
                    if isinstance(lat_lng, dict):
                        result["approxLocation_lat"] = lat_lng.get("lat")
                        result["approxLocation_lng"] = lat_lng.get("lng")
                else:
                    # If location is not a dict, set all to None
                    result["approxLocation_city"] = None
                    result["approxLocation_countryCode"] = None
                    result["approxLocation_regionCode"] = None
                    result["approxLocation_postalCode"] = None
                    result["approxLocation_lat"] = None
                    result["approxLocation_lng"] = None
            else:
                # Extract other fields directly
                result[field] = data.get(field)

        logger.debug(f"Processed enriched auction {auction_id}: {len(result)} fields")

        return result

    async def fetch_and_process_enriched_auction(
        self, auction_id: int
    ) -> dict[str, Any] | None:
        """
        Fetch and process a single enriched auction.

        Args:
            auction_id: Auction ID to fetch

        Returns:
            Processed enriched auction data or None if failed
        """
        try:
            data = await self.fetch_enriched_auction(auction_id)
            processed = self.process_enriched_auction_data(auction_id, data)
            return processed
        except Exception as e:
            logger.error(f"Failed to fetch enriched auction {auction_id}: {e}")
            return None

    async def fetch_multiple_enriched_auctions(
        self,
        auction_ids: list[int],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple enriched auctions with controlled concurrency.

        Args:
            auction_ids: List of auction IDs to fetch
            progress_callback: Optional callback function for progress updates

        Returns:
            List of processed enriched auction data
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_with_semaphore(auction_id: int) -> dict[str, Any] | None:
            async with semaphore:
                result = await self.fetch_and_process_enriched_auction(auction_id)
                if progress_callback:
                    progress_callback(auction_id, result is not None)
                return result

        tasks = [fetch_with_semaphore(aid) for aid in auction_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception for auction {auction_ids[i]}: {result}")
            elif result is not None:
                valid_results.append(result)

        logger.info(
            f"Successfully fetched {len(valid_results)}/{len(auction_ids)} enriched auctions"
        )

        return valid_results


# =============================================================================
# Data Transformation
# =============================================================================


def transform_enriched_auction_data(auctions: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform enriched auction data: rename fields and add prefix to column names.

    Exception: auction_id and item_id do not get the prefix and remain as-is.

    Args:
        auctions: List of enriched auction dictionaries

    Returns:
        DataFrame with transformed column names
    """
    if not auctions:
        logger.warning("No enriched auctions to transform")
        return pd.DataFrame()

    df = pd.DataFrame(auctions)

    # Rename fields according to mapping (amAuctionId→auction_id, amLotId→item_id)
    rename_map = {}
    for old_name in df.columns:
        new_name = config.get_renamed_enriched_auction_field_name(old_name)
        if new_name != old_name:
            rename_map[old_name] = new_name

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug(f"Renamed columns: {rename_map}")

    # Add enriched_auction_ prefix to all columns (except auction_id and item_id)
    prefix_map = {
        col: config.get_prefixed_enriched_auction_field_name(col) for col in df.columns
    }
    df = df.rename(columns=prefix_map)
    logger.debug(f"Added prefix to columns: {list(df.columns)}")

    return df


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, progress_file: Path = config.ENRICHED_AUCTION_PROGRESS_FILE):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to progress file
        """
        self.progress_file = progress_file
        self.completed_ids: set[int] = set()
        self.failed_ids: set[int] = set()
        self._load()

    def _load(self) -> None:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get("completed", []))
                    self.failed_ids = set(data.get("failed", []))
                    logger.info(
                        f"Loaded progress: {len(self.completed_ids)} completed, "
                        f"{len(self.failed_ids)} failed"
                    )
            except Exception as e:
                logger.error(f"Error loading progress file: {e}")

    def save(self) -> None:
        """Save progress to file."""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w") as f:
                json.dump(
                    {
                        "completed": list(self.completed_ids),
                        "failed": list(self.failed_ids),
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving progress file: {e}")

    def mark_completed(self, auction_id: int) -> None:
        """Mark auction as successfully scraped."""
        self.completed_ids.add(auction_id)
        self.failed_ids.discard(auction_id)

    def mark_failed(self, auction_id: int) -> None:
        """Mark auction as failed."""
        self.failed_ids.add(auction_id)

    def is_completed(self, auction_id: int) -> bool:
        """Check if auction was already scraped."""
        return auction_id in self.completed_ids

    def filter_pending(self, auction_ids: list[int]) -> list[int]:
        """Filter out already completed auctions."""
        return [aid for aid in auction_ids if aid not in self.completed_ids]


# =============================================================================
# Auction ID Loading from Hugging Face
# =============================================================================


def load_auction_ids_from_hf(limit: int | None = None) -> list[int]:
    """
    Load auction IDs from Hugging Face dataset.

    Args:
        limit: Optional limit on number of auction IDs to load

    Returns:
        List of auction IDs
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading auction IDs from Hugging Face: {config.HF_DATASET_REPO}")

        # Load dataset from Hugging Face
        dataset = load_dataset(config.HF_DATASET_REPO, split="train")

        # Extract auction IDs
        id_column = config.HF_AUCTION_ID_COLUMN
        if id_column not in dataset.column_names:
            logger.error(
                f"Column '{id_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
            # Try alternative column names
            possible_columns = ["auction_id", "amAuctionId", "id"]
            for col in possible_columns:
                if col in dataset.column_names:
                    logger.info(f"Using alternative column: {col}")
                    id_column = col
                    break
            else:
                raise ValueError("Could not find auction ID column in dataset")

        auction_ids = dataset[id_column]

        # Convert to list and ensure integers
        auction_ids = [int(aid) for aid in auction_ids]

        # Remove duplicates and sort
        auction_ids = sorted(set(auction_ids))

        logger.info(f"Loaded {len(auction_ids)} unique auction IDs from Hugging Face")

        # Apply limit if specified
        if limit:
            auction_ids = auction_ids[:limit]
            logger.info(f"Limited to {limit} auction IDs")

        return auction_ids

    except ImportError:
        logger.error(
            "datasets library not installed. Install with: pip install datasets"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load auction IDs from Hugging Face: {e}")
        raise


# =============================================================================
# Main Scraping Function
# =============================================================================


async def scrape_enriched_auctions(
    auction_ids: list[int] | None = None,
    limit: int | None = None,
    use_progress_tracking: bool = True,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    rate_limit: int = config.DEFAULT_RATE_LIMIT,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """
    Scrape enriched auction data from MaxSold API.

    Args:
        auction_ids: List of auction IDs to scrape (if None, loads from Hugging Face)
        limit: Maximum number of auctions to scrape
        use_progress_tracking: Whether to track and resume progress
        max_workers: Maximum parallel workers
        rate_limit: Maximum requests per second
        output_file: Output file path (if None, uses default from config)

    Returns:
        DataFrame with scraped enriched auction data
    """
    # Load auction IDs if not provided
    if auction_ids is None:
        logger.info("Loading auction IDs from Hugging Face")
        auction_ids = load_auction_ids_from_hf(limit=limit)
    else:
        # Apply limit if specified
        if limit:
            auction_ids = auction_ids[:limit]
            logger.info(f"Limited to {limit} auctions")

    # Filter out completed auctions if tracking progress
    if use_progress_tracking:
        tracker = ProgressTracker()
        original_count = len(auction_ids)
        auction_ids = tracker.filter_pending(auction_ids)
        logger.info(
            f"After filtering completed: {len(auction_ids)} pending "
            f"({original_count - len(auction_ids)} already completed)"
        )
    else:
        tracker = None

    if not auction_ids:
        logger.warning("No auctions to scrape")
        return pd.DataFrame()

    # Setup progress callback
    def progress_callback(auction_id: int, success: bool) -> None:
        if tracker:
            if success:
                tracker.mark_completed(auction_id)
            else:
                tracker.mark_failed(auction_id)
            tracker.save()

    # Fetch enriched auctions
    logger.info(f"Starting to scrape {len(auction_ids)} enriched auctions...")

    async with EnrichedAuctionDataFetcher(
        rate_limit=rate_limit,
        max_concurrent=max_workers,
    ) as fetcher:
        auctions = await fetcher.fetch_multiple_enriched_auctions(
            auction_ids,
            progress_callback=progress_callback if use_progress_tracking else None,
        )

    # Transform data
    logger.info(f"Transforming {len(auctions)} enriched auction records...")
    df = transform_enriched_auction_data(auctions)

    # Save to file
    if output_file is None:
        output_file = (
            config.ENRICHED_AUCTION_PROCESSED_OUTPUT_DIR
            / config.ENRICHED_AUCTION_DATA_FILENAME
        )

    if not df.empty:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")

    return df


# =============================================================================
# Hugging Face Upload
# =============================================================================


async def upload_to_huggingface(
    data_file: Path | None = None,
    repo_id: str | None = None,
    private: bool = False,
) -> None:
    """
    Upload scraped enriched auction data to Hugging Face Datasets.

    Args:
        data_file: Path to parquet file (uses default if None)
        repo_id: HuggingFace repository ID (uses default if None)
        private: Whether to make the dataset private
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Install with: pip install datasets huggingface-hub")
        return

    try:
        from src.config import settings
    except ImportError as e:
        logger.error(f"Could not import settings: {e}")
        return

    # Setup paths
    if data_file is None:
        data_file = (
            config.ENRICHED_AUCTION_PROCESSED_OUTPUT_DIR
            / config.ENRICHED_AUCTION_DATA_FILENAME
        )

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    # Setup repo
    if repo_id is None:
        repo_id = settings.huggingface.dataset_id
        if not repo_id:
            logger.error("Hugging Face repository ID not configured")
            logger.error("Set HF_DATASET_REPO in environment or .env file")
            return

    # Validate token
    if not settings.huggingface.token:
        logger.error("Hugging Face token not found")
        logger.error("Set HF_TOKEN in environment or .env file")
        return

    logger.info(f"Uploading {data_file} to {repo_id}")

    try:
        # Load data
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(df)} records from {data_file}")

        # Create dataset
        dataset = Dataset.from_pandas(df)

        # Create metadata
        metadata = {
            "name": config.HF_ENRICHED_AUCTION_DATASET_NAME,
            "description": config.HF_ENRICHED_AUCTION_DATASET_DESCRIPTION,
            "license": config.HF_ENRICHED_AUCTION_DATASET_LICENSE,
            "tags": config.HF_ENRICHED_AUCTION_DATASET_TAGS,
            "num_records": len(df),
            "columns": list(df.columns),
            "created_at": datetime.utcnow().isoformat(),
        }

        # Save metadata
        metadata_file = data_file.parent / config.METADATA_FILENAME
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Created metadata file: {metadata_file}")

        # Upload dataset
        logger.info("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            repo_id,
            private=private,
            token=settings.huggingface.token,
        )

        # Upload metadata file
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo=config.METADATA_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            token=settings.huggingface.token,
        )

        logger.info(f"Successfully uploaded to {repo_id}")
        logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        raise


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Command-line interface for enriched auction scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape enriched auction data from MaxSold API"
    )
    parser.add_argument(
        "--auction-ids",
        type=int,
        nargs="+",
        help="Specific auction IDs to scrape",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of auctions to scrape",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress tracking",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {config.DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=config.DEFAULT_RATE_LIMIT,
        help=f"Requests per second (default: {config.DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload to Hugging Face after scraping",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Make Hugging Face dataset private",
    )

    args = parser.parse_args()

    # Ensure directories exist
    config.ENRICHED_AUCTION_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ENRICHED_AUCTION_PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run scraper
    logger.info("Starting enriched auction data scraper...")

    df = asyncio.run(
        scrape_enriched_auctions(
            auction_ids=args.auction_ids,
            limit=args.limit,
            use_progress_tracking=not args.no_progress,
            max_workers=args.workers,
            rate_limit=args.rate_limit,
            output_file=Path(args.output) if args.output else None,
        )
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df)}")
    if not df.empty:
        logger.info(f"Columns: {', '.join(df.columns)}")
        logger.info("\nFirst few records:")
        print(df.head())

    # Upload to Hugging Face if requested
    if args.upload_hf:
        logger.info("\nUploading to Hugging Face...")
        asyncio.run(
            upload_to_huggingface(
                repo_id=args.hf_repo,
                private=args.hf_private,
            )
        )


if __name__ == "__main__":
    main()
