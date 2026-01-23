# =============================================================================
# Auction Data Scraper
# =============================================================================
"""
Scraper for auction-level data with aggregated item metrics.

This module implements the requirements from the issue:
1. Scrapes auction data from MaxSold API
2. Aggregates item-level metrics (viewed, current_bid, bid_count, image count)
3. Renames fields per specifications (catalog_lots -> item_count, current_bid -> winning_price)
4. Adds 'auction_' prefix to all column names
5. Uses parallel processing for faster scraping
6. Uploads to Hugging Face

Usage:
    # From command line
    python -m src.data.auction_scraper --limit 100
    
    # From Python
    from src.data.auction_scraper import scrape_auctions
    df = await scrape_auctions(auction_ids=[99941, 99942])
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
# Auction Data Fetcher
# =============================================================================


class AuctionDataFetcher:
    """
    Fetches and processes auction data from MaxSold API.
    
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
        Initialize the auction data fetcher.
        
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

    async def __aenter__(self) -> "AuctionDataFetcher":
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
            raise RuntimeError("Client not initialized. Use 'async with AuctionDataFetcher():'")
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
    async def fetch_auction_items(self, auction_id: int) -> dict[str, Any]:
        """
        Fetch auction items from MaxSold API.
        
        Args:
            auction_id: Auction ID to fetch
            
        Returns:
            Dictionary containing auction data and items
        """
        await self._rate_limit()

        url = config.AUCTION_ITEMS_ENDPOINT
        params = {
            "auctionid": auction_id,
            "limit": config.DEFAULT_ITEMS_LIMIT,
        }

        logger.debug(f"Fetching auction {auction_id}")
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for auction {auction_id}: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error fetching auction {auction_id}: {e}")
            raise

    def process_auction_data(self, auction_id: int, data: dict[str, Any] | list[Any]) -> dict[str, Any]:
        """
        Process raw API response into auction-level aggregated data.
        
        Args:
            auction_id: Auction ID
            data: Raw API response (can be dict with 'auction' key or list of items)
            
        Returns:
            Dictionary with auction-level data and aggregated metrics
        """
        # Handle different response structures
        auction_data = {}
        items = []
        
        if isinstance(data, dict):
            # Response has auction metadata and items
            auction_data = data.get("auction", {})
            items = data.get("items", [])
            
            # If auction_data is empty but we have items, extract from first item
            if not auction_data and items:
                # Some APIs include auction info in each item
                first_item = items[0] if items else {}
                if "auction" in first_item:
                    auction_data = first_item["auction"]
        elif isinstance(data, list):
            # Response is just a list of items
            items = data
        else:
            logger.warning(f"Unexpected response type for auction {auction_id}: {type(data)}")
            
        # Extract auction-level fields
        result = {"id": auction_id}
        
        for field in config.AUCTION_FIELDS:
            if field == "id":
                continue  # Already set
            result[field] = auction_data.get(field)
        
        # Aggregate item-level metrics
        result["total_viewed"] = sum(item.get("viewed", 0) for item in items)
        result["total_winning_price"] = sum(item.get("current_bid", 0) for item in items)
        result["total_bid_count"] = sum(item.get("bid_count", 0) for item in items)
        
        # Count total images across all items
        total_images = 0
        for item in items:
            images = item.get("images", [])
            if isinstance(images, list):
                total_images += len(images)
            elif isinstance(images, int):
                total_images += images
                
        result["total_images"] = total_images
        
        # If catalog_lots is not in auction data, use the count of items
        if result.get("catalog_lots") is None:
            result["catalog_lots"] = len(items)
            
        logger.debug(
            f"Processed auction {auction_id}: "
            f"{result.get('catalog_lots', 0)} items, "
            f"{result['total_bid_count']} bids"
        )
        
        return result

    async def fetch_and_process_auction(self, auction_id: int) -> dict[str, Any] | None:
        """
        Fetch and process a single auction.
        
        Args:
            auction_id: Auction ID to fetch
            
        Returns:
            Processed auction data or None if failed
        """
        try:
            data = await self.fetch_auction_items(auction_id)
            processed = self.process_auction_data(auction_id, data)
            return processed
        except Exception as e:
            logger.error(f"Failed to fetch auction {auction_id}: {e}")
            return None

    async def fetch_multiple_auctions(
        self,
        auction_ids: list[int],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple auctions with controlled concurrency.
        
        Args:
            auction_ids: List of auction IDs to fetch
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processed auction data
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_semaphore(auction_id: int) -> dict[str, Any] | None:
            async with semaphore:
                result = await self.fetch_and_process_auction(auction_id)
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
                
        logger.info(f"Successfully fetched {len(valid_results)}/{len(auction_ids)} auctions")
        
        return valid_results


# =============================================================================
# Data Transformation
# =============================================================================


def transform_auction_data(auctions: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform auction data: rename fields and add prefix to column names.
    
    Args:
        auctions: List of auction dictionaries
        
    Returns:
        DataFrame with transformed column names
    """
    if not auctions:
        logger.warning("No auctions to transform")
        return pd.DataFrame()
    
    df = pd.DataFrame(auctions)
    
    # Rename fields according to mapping
    rename_map = {}
    for old_name in df.columns:
        new_name = config.get_renamed_field_name(old_name)
        if new_name != old_name:
            rename_map[old_name] = new_name
            
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug(f"Renamed columns: {rename_map}")
    
    # Add auction_ prefix to all columns
    prefix_map = {col: config.get_prefixed_field_name(col) for col in df.columns}
    df = df.rename(columns=prefix_map)
    logger.debug(f"Added prefix to columns: {list(df.columns)}")
    
    return df


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, progress_file: Path = config.PROGRESS_FILE):
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
# Main Scraping Function
# =============================================================================


async def scrape_auctions(
    auction_ids: list[int] | None = None,
    limit: int | None = None,
    use_progress_tracking: bool = True,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    rate_limit: int = config.DEFAULT_RATE_LIMIT,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """
    Scrape auction data from MaxSold API.
    
    Args:
        auction_ids: List of auction IDs to scrape (if None, loads from config file)
        limit: Maximum number of auctions to scrape
        use_progress_tracking: Whether to track and resume progress
        max_workers: Maximum parallel workers
        rate_limit: Maximum requests per second
        output_file: Output file path (if None, uses default from config)
        
    Returns:
        DataFrame with scraped auction data
    """
    # Load auction IDs if not provided
    if auction_ids is None:
        logger.info(f"Loading auction IDs from {config.AUCTION_IDS_FILE}")
        try:
            df = pd.read_parquet(config.AUCTION_IDS_FILE)
            auction_ids = df[config.AUCTION_IDS_COLUMN].tolist()
            logger.info(f"Loaded {len(auction_ids)} auction IDs")
        except Exception as e:
            logger.error(f"Failed to load auction IDs: {e}")
            raise
    
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
    
    # Fetch auctions
    logger.info(f"Starting to scrape {len(auction_ids)} auctions...")
    
    async with AuctionDataFetcher(
        rate_limit=rate_limit,
        max_concurrent=max_workers,
    ) as fetcher:
        auctions = await fetcher.fetch_multiple_auctions(
            auction_ids,
            progress_callback=progress_callback if use_progress_tracking else None,
        )
    
    # Transform data
    logger.info(f"Transforming {len(auctions)} auction records...")
    df = transform_auction_data(auctions)
    
    # Save to file
    if output_file is None:
        output_file = config.PROCESSED_OUTPUT_DIR / config.AUCTION_DATA_FILENAME
    
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
    Upload scraped data to Hugging Face Datasets.
    
    Args:
        data_file: Path to parquet file (uses default if None)
        repo_id: HuggingFace repository ID (uses default if None)
        private: Whether to make the dataset private
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
        from src.config import settings
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Install with: pip install datasets huggingface-hub")
        return
    
    # Setup paths
    if data_file is None:
        data_file = config.PROCESSED_OUTPUT_DIR / config.AUCTION_DATA_FILENAME
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    # Setup repo
    if repo_id is None:
        repo_id = settings.huggingface.dataset_id
        if not repo_id or repo_id == "maxsold-auctions":
            logger.error("Hugging Face repository ID not configured")
            logger.error("Set HF_DATASET_REPO in environment or .env file")
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
            "name": config.HF_DATASET_NAME,
            "description": config.HF_DATASET_DESCRIPTION,
            "license": config.HF_DATASET_LICENSE,
            "tags": config.HF_DATASET_TAGS,
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
    """Command-line interface for auction scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape auction data from MaxSold API"
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
    config.ensure_directories()
    
    # Run scraper
    logger.info("Starting auction data scraper...")
    
    df = asyncio.run(
        scrape_auctions(
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
        logger.info(f"\nFirst few records:")
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
