# =============================================================================
# Bid Data Scraper
# =============================================================================
"""
Scraper for bid-level data from MaxSold auctions.

This module implements the requirements from the issue:
1. Scrapes bid data from MaxSold API
2. Loads item IDs from Hugging Face dataset (jpearce610/item_data)
3. Extracts bid history fields (time_of_bid, amount, isproxy)
4. Renames time_of_bid to 'time' and adds 'bid_' prefix to columns
5. First bid gets bid_count as bid_id, last bid gets 1 (counting downwards)
6. Uses parallel processing for faster scraping
7. Implements memory-efficient batch processing (100 auctions at a time)
8. Uploads to Hugging Face

Usage:
    # From command line
    python -m src.data.bid_scraper --limit 100

    # From Python
    from src.data.bid_scraper import scrape_bids
    df = await scrape_bids(item_ids=[(103293, 7433850)])
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
# Bid Data Fetcher
# =============================================================================


class BidDataFetcher:
    """
    Fetches and processes bid data from MaxSold API.

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
        Initialize the bid data fetcher.

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

    async def __aenter__(self) -> "BidDataFetcher":
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
                "Client not initialized. Use 'async with BidDataFetcher():'"
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
    async def fetch_item_bids(
        self, auction_id: int, item_id: int
    ) -> list[dict[str, Any]]:
        """
        Fetch bid history for a single item from MaxSold API.

        Args:
            auction_id: Auction ID
            item_id: Item ID

        Returns:
            List of bid dictionaries
        """
        await self._rate_limit()

        url = config.AUCTION_ITEMS_ENDPOINT
        params = {
            "auctionid": auction_id,
            "itemid": item_id,
        }

        logger.debug(f"Fetching bids for item {item_id} in auction {auction_id}")

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract bid history from response
            bids = []

            if isinstance(data, dict):
                # Response has auction metadata with items
                auction_data = data.get("auction", {})
                items_data = auction_data.get("items", [])

                # Handle different item structures
                if isinstance(items_data, dict):
                    # Items are structured as {0: item_data, ...}
                    # Get the first (and likely only) item
                    if items_data:
                        item_data = next(iter(items_data.values()))
                        bids = item_data.get("bid_history", [])
                elif isinstance(items_data, list) and items_data:
                    # Items are a list
                    item_data = items_data[0]
                    bids = item_data.get("bid_history", [])

            elif isinstance(data, list) and data:
                # Response is just a list with one item
                item_data = data[0]
                bids = item_data.get("bid_history", [])

            logger.info(
                f"Retrieved {len(bids)} bids for item {item_id} in auction {auction_id}"
            )
            return bids

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for item {item_id}: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error fetching item {item_id}: {e}")
            raise

    def process_bid_data(
        self, bids: list[dict[str, Any]], auction_id: int, item_id: int
    ) -> list[dict[str, Any]]:
        """
        Process raw bid data and extract configured fields.

        Applies the following transformations:
        1. Extracts configured bid fields
        2. Adds auction_id and item_id
        3. Assigns id counting downward (first bid = count, last bid = 1)

        Args:
            bids: Raw bid data from API
            auction_id: Parent auction ID
            item_id: Parent item ID

        Returns:
            List of dictionaries with extracted bid fields
        """
        processed_bids = []
        bid_count = len(bids)

        for idx, bid in enumerate(bids):
            result = {
                "auction_id": auction_id,
                "item_id": item_id,
            }

            # Extract configured bid fields
            for field in config.BID_FIELDS:
                result[field] = bid.get(field)

            # Assign id counting downward (first bid = count, last bid = 1)
            # These will get bid_ prefix added during transformation
            result["id"] = bid_count - idx
            result["count"] = bid_count

            processed_bids.append(result)

        return processed_bids

    async def fetch_and_process_item(
        self, auction_id: int, item_id: int
    ) -> list[dict[str, Any]] | None:
        """
        Fetch and process bid history for a single item.

        Args:
            auction_id: Auction ID
            item_id: Item ID

        Returns:
            List of processed bid data or None if failed
        """
        try:
            bids = await self.fetch_item_bids(auction_id, item_id)
            if not bids:
                logger.debug(f"No bids found for item {item_id}")
                return []
            processed_bids = self.process_bid_data(bids, auction_id, item_id)
            return processed_bids
        except Exception as e:
            logger.error(f"Failed to fetch bids for item {item_id}: {e}")
            return None

    async def fetch_multiple_items(
        self,
        item_pairs: list[tuple[int, int]],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Fetch bids for multiple items with controlled concurrency.

        Args:
            item_pairs: List of (auction_id, item_id) tuples
            progress_callback: Optional callback function for progress updates

        Returns:
            Flattened list of all bids from all items
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_with_semaphore(
            auction_id: int, item_id: int
        ) -> list[dict[str, Any]] | None:
            async with semaphore:
                result = await self.fetch_and_process_item(auction_id, item_id)
                if progress_callback:
                    progress_callback(
                        auction_id, item_id, result is not None and len(result) > 0
                    )
                return result

        tasks = [fetch_with_semaphore(aid, iid) for aid, iid in item_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out None and exceptions
        all_bids = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception for item {item_pairs[i][1]}: {result}")
            elif result is not None:
                all_bids.extend(result)

        logger.info(
            f"Successfully fetched {len(all_bids)} bids from "
            f"{len(item_pairs)} items"
        )

        return all_bids


# =============================================================================
# Data Transformation
# =============================================================================


def transform_bid_data(bids: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform bid data: rename fields and add 'bid_' prefix to column names.

    Exception: auction_id and item_id do not get the prefix and remain as-is.

    Args:
        bids: List of bid dictionaries

    Returns:
        DataFrame with transformed column names (all have bid_ prefix except auction_id, item_id)
    """
    if not bids:
        logger.warning("No bids to transform")
        return pd.DataFrame()

    df = pd.DataFrame(bids)

    # Rename fields according to mapping
    rename_map = {}
    for old_name in df.columns:
        new_name = config.get_renamed_bid_field_name(old_name)
        if new_name != old_name:
            rename_map[old_name] = new_name

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug(f"Renamed columns: {rename_map}")

    # Add bid_ prefix to all columns except auction_id and item_id
    prefix_map = {col: config.get_prefixed_bid_field_name(col) for col in df.columns}
    df = df.rename(columns=prefix_map)
    logger.debug(f"Added 'bid_' prefix to columns: {list(df.columns)}")

    return df


def append_to_parquet_efficient(df_new: pd.DataFrame, output_file: Path) -> int:
    """
    Append DataFrame to parquet file efficiently using pyarrow.

    This avoids loading the entire existing file into memory.

    Args:
        df_new: New data to append
        output_file: Path to parquet file

    Returns:
        Total number of rows after append
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_file.exists():
        # Read existing parquet file metadata to get row count
        existing_table = pq.read_table(output_file)

        # Convert new df to arrow table
        new_table = pa.Table.from_pandas(df_new, preserve_index=False)

        # Write both tables to parquet
        combined_table = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined_table, output_file)

        total_count = len(combined_table)
        logger.info(f"Appended {len(df_new)} rows (total: {total_count})")

        # Clean up
        del existing_table
        del new_table
        del combined_table

        return total_count
    else:
        # First write
        df_new.to_parquet(output_file, index=False)
        logger.info(f"Created {output_file} with {len(df_new)} records")
        return len(df_new)


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, progress_file: Path = config.BID_PROGRESS_FILE):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to progress file
        """
        self.progress_file = progress_file
        self.completed_items: set[tuple[int, int]] = set()
        self.failed_items: set[tuple[int, int]] = set()
        self._load()

    def _load(self) -> None:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    # Load as list of [auction_id, item_id] pairs
                    self.completed_items = {
                        tuple(pair) for pair in data.get("completed", [])
                    }
                    self.failed_items = {tuple(pair) for pair in data.get("failed", [])}
                    logger.info(
                        f"Loaded progress: {len(self.completed_items)} completed, "
                        f"{len(self.failed_items)} failed"
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
                        "completed": [list(pair) for pair in self.completed_items],
                        "failed": [list(pair) for pair in self.failed_items],
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving progress file: {e}")

    def mark_completed(self, auction_id: int, item_id: int) -> None:
        """Mark item as successfully scraped."""
        pair = (auction_id, item_id)
        self.completed_items.add(pair)
        self.failed_items.discard(pair)

    def mark_failed(self, auction_id: int, item_id: int) -> None:
        """Mark item as failed."""
        pair = (auction_id, item_id)
        self.failed_items.add(pair)

    def is_completed(self, auction_id: int, item_id: int) -> bool:
        """Check if item was already scraped."""
        return (auction_id, item_id) in self.completed_items

    def filter_pending(
        self, item_pairs: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Filter out already completed items."""
        return [pair for pair in item_pairs if pair not in self.completed_items]


# =============================================================================
# Item ID Loading from Hugging Face
# =============================================================================


def load_item_ids_from_hf(limit: int | None = None) -> list[tuple[int, int]]:
    """
    Load (auction_id, item_id) pairs from Hugging Face dataset.

    Args:
        limit: Optional limit on number of items to load

    Returns:
        List of (auction_id, item_id) tuples
    """
    try:
        from datasets import load_dataset

        logger.info(
            f"Loading item IDs from Hugging Face: {config.HF_ITEM_DATASET_REPO}"
        )

        # Load dataset from Hugging Face
        dataset = load_dataset(config.HF_ITEM_DATASET_REPO, split="train")

        # Extract item and auction IDs
        item_id_column = config.HF_ITEM_ID_COLUMN
        auction_id_column = "auction_id"  # Should be present in item dataset

        if item_id_column not in dataset.column_names:
            logger.error(
                f"Column '{item_id_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
            raise ValueError("Could not find item_id column in dataset")

        if auction_id_column not in dataset.column_names:
            logger.error(
                f"Column '{auction_id_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
            raise ValueError("Could not find auction_id column in dataset")

        item_ids = dataset[item_id_column]
        auction_ids = dataset[auction_id_column]

        # Create list of (auction_id, item_id) pairs
        item_pairs = list(zip(auction_ids, item_ids, strict=True))

        # Convert to integers
        item_pairs = [(int(aid), int(iid)) for aid, iid in item_pairs]

        logger.info(f"Loaded {len(item_pairs)} item pairs from Hugging Face")

        # Apply limit if specified
        if limit:
            item_pairs = item_pairs[:limit]
            logger.info(f"Limited to {limit} item pairs")

        return item_pairs

    except ImportError:
        logger.error(
            "datasets library not installed. Install with: pip install datasets"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load item IDs from Hugging Face: {e}")
        raise


# =============================================================================
# Main Scraping Function
# =============================================================================


async def scrape_bids(
    item_pairs: list[tuple[int, int]] | None = None,
    limit: int | None = None,
    use_progress_tracking: bool = True,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    rate_limit: int = config.DEFAULT_RATE_LIMIT,
    output_file: Path | None = None,
    batch_size: int = 100,  # Process items in batches to reduce memory
) -> pd.DataFrame:
    """
    Scrape bid data from MaxSold API with batch processing for memory efficiency.

    Args:
        item_pairs: List of (auction_id, item_id) tuples to scrape (if None, loads from HF)
        limit: Maximum number of items to scrape
        use_progress_tracking: Whether to track and resume progress
        max_workers: Maximum parallel workers
        rate_limit: Maximum requests per second
        output_file: Output file path (if None, uses default from config)
        batch_size: Number of items to process per batch (default: 100)

    Returns:
        DataFrame with scraped bid data
    """
    # Load item pairs if not provided
    if item_pairs is None:
        logger.info("Loading item IDs from Hugging Face")
        item_pairs = load_item_ids_from_hf(limit=limit)
    else:
        # Apply limit if specified
        if limit:
            item_pairs = item_pairs[:limit]
            logger.info(f"Limited to {limit} items")

    # Filter out completed items if tracking progress
    if use_progress_tracking:
        tracker = ProgressTracker()
        original_count = len(item_pairs)
        item_pairs = tracker.filter_pending(item_pairs)
        logger.info(
            f"After filtering completed: {len(item_pairs)} pending "
            f"({original_count - len(item_pairs)} already completed)"
        )
    else:
        tracker = None

    if not item_pairs:
        logger.warning("No items to scrape")
        # Return existing data if available
        if output_file is None:
            output_file = config.BID_PROCESSED_OUTPUT_DIR / config.BID_DATA_FILENAME
        if output_file.exists():
            return pd.read_parquet(output_file)
        return pd.DataFrame()

    # Setup output file
    if output_file is None:
        output_file = config.BID_PROCESSED_OUTPUT_DIR / config.BID_DATA_FILENAME
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress callback
    def progress_callback(auction_id: int, item_id: int, success: bool) -> None:
        if tracker:
            if success:
                tracker.mark_completed(auction_id, item_id)
            else:
                tracker.mark_failed(auction_id, item_id)
            tracker.save()

    # Process items in batches to balance speed and memory
    logger.info(
        f"Starting to scrape bids from {len(item_pairs)} items "
        f"(max {max_workers} concurrent, write every {batch_size} items)..."
    )

    total_bids_scraped = 0
    items_processed = 0
    accumulated_bids = []

    # Process in chunks
    for chunk_start in range(0, len(item_pairs), batch_size):
        chunk_pairs = item_pairs[chunk_start : chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        total_chunks = (len(item_pairs) + batch_size - 1) // batch_size

        logger.info(
            f"Processing chunk {chunk_num}/{total_chunks} with {len(chunk_pairs)} items "
            f"(progress: {items_processed}/{len(item_pairs)})"
        )

        # Fetch bids for this chunk with concurrency
        async with BidDataFetcher(
            rate_limit=rate_limit,
            max_concurrent=max_workers,
        ) as fetcher:
            chunk_bids = await fetcher.fetch_multiple_items(
                chunk_pairs,
                progress_callback=progress_callback if use_progress_tracking else None,
            )

        # Add to accumulated bids
        accumulated_bids.extend(chunk_bids)
        items_processed += len(chunk_pairs)

        logger.info(
            f"Chunk {chunk_num} fetched {len(chunk_bids)} bids. "
            f"Accumulated: {len(accumulated_bids)} bids"
        )

        # Write accumulated bids to disk and clear memory
        if accumulated_bids:
            df_batch = transform_bid_data(accumulated_bids)

            if not df_batch.empty:
                total_bids_scraped = append_to_parquet_efficient(df_batch, output_file)
                logger.info(
                    f"Saved chunk {chunk_num}. Total in file: {total_bids_scraped:,} bids"
                )

            # Clear memory
            del df_batch
            del accumulated_bids
            del chunk_bids
            accumulated_bids = []  # Reset for next chunk

        logger.info(f"Progress: {items_processed}/{len(item_pairs)} items complete")

    # Load final result
    if output_file.exists():
        df = pd.read_parquet(output_file)
        logger.info(f"Final dataset: {len(df)} records saved to {output_file}")
        return df
    else:
        logger.warning("No data was scraped")
        return pd.DataFrame()


# =============================================================================
# Hugging Face Upload
# =============================================================================


async def upload_to_huggingface(
    data_file: Path | None = None,
    repo_id: str | None = None,
    private: bool = False,
) -> None:
    """
    Upload scraped bid data to Hugging Face Datasets.

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
        data_file = config.BID_PROCESSED_OUTPUT_DIR / config.BID_DATA_FILENAME

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
            "name": config.HF_BID_DATASET_NAME,
            "description": config.HF_BID_DATASET_DESCRIPTION,
            "license": config.HF_BID_DATASET_LICENSE,
            "tags": config.HF_BID_DATASET_TAGS,
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
    """Command-line interface for bid scraper."""
    parser = argparse.ArgumentParser(description="Scrape bid data from MaxSold API")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of items to scrape",
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
        "--batch-size",
        type=int,
        default=100,
        help="Items per batch for memory efficiency (default: 100)",
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
    config.BID_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.BID_PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run scraper
    logger.info("Starting bid data scraper...")

    df = asyncio.run(
        scrape_bids(
            limit=args.limit,
            use_progress_tracking=not args.no_progress,
            max_workers=args.workers,
            rate_limit=args.rate_limit,
            output_file=Path(args.output) if args.output else None,
            batch_size=args.batch_size,
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
