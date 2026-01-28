# =============================================================================
# Enriched Item Data Scraper
# =============================================================================
"""
Scraper for enriched item-level data from MaxSold API.

This module implements the requirements from the issue:
1. Scrapes enriched item data from MaxSold enriched API endpoint
2. Loads item IDs from Hugging Face dataset (jpearce610/item_data)
3. Extracts specified enriched item fields (amLotId, amAuctionId, title, brand, etc.)
4. Adds 'enriched_item_' prefix to all column names (except item_id and auction_id)
5. Uses parallel processing for faster scraping
6. Uploads to Hugging Face

Usage:
    # From command line
    python -m src.data.enriched_item_scraper --limit 100

    # From Python
    from src.data.enriched_item_scraper import scrape_enriched_items
    df = await scrape_enriched_items(item_ids=[7433915, 7433916])
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Enriched Item Data Fetcher
# =============================================================================


class EnrichedItemDataFetcher:
    """
    Fetches and processes enriched item data from MaxSold API.

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
        Initialize the enriched item data fetcher.

        Args:
            rate_limit: Maximum requests per second
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        # Token bucket rate limiting (allows true concurrency)
        self.tokens = float(rate_limit)  # Start with full bucket
        self.max_tokens = float(rate_limit)
        self.refill_rate = float(rate_limit)  # Tokens per second
        self.last_refill_time = 0.0
        self._token_lock = asyncio.Lock()  # Only for token updates, not requests
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EnrichedItemDataFetcher":
        """Initialize async HTTP client with connection pooling."""
        # Configure connection pool for better performance with concurrent requests
        limits = httpx.Limits(
            max_connections=self.max_concurrent * 2,  # Allow some buffer
            max_keepalive_connections=self.max_concurrent * 2,
            keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
        )
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            limits=limits,
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
                "Client not initialized. Use 'async with EnrichedItemDataFetcher():'"
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Apply token bucket rate limiting to allow concurrent requests."""
        while True:
            async with self._token_lock:
                current_time = asyncio.get_event_loop().time()
                
                # Refill tokens based on time elapsed
                if self.last_refill_time > 0:
                    time_elapsed = current_time - self.last_refill_time
                    tokens_to_add = time_elapsed * self.refill_rate
                    self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
                
                self.last_refill_time = current_time
                
                # Check if we have a token available
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return  # Token acquired, proceed with request
                
                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.refill_rate
            
            # Wait outside the lock to allow other coroutines to run
            await asyncio.sleep(wait_time)

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if request should be retried based on exception type."""
        # Retry network errors and timeouts
        if isinstance(exception, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        
        # For HTTP errors, only retry 5xx server errors and 429 rate limits
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            # Don't retry 404s or other client errors
            return status_code >= 500 or status_code == 429
        
        return False

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=config.RETRY_DELAY, max=10),
        retry=lambda retry_state: retry_state.outcome.failed and 
              retry_state.args[0]._should_retry(retry_state.outcome.exception()),
    )
    async def fetch_enriched_item(self, item_id: int) -> dict[str, Any] | None:
        """
        Fetch enriched item data from MaxSold API.

        Args:
            item_id: Item ID to fetch enriched data for

        Returns:
            Dictionary with enriched item data or None if failed
        """
        await self._rate_limit()

        url = config.ENRICHED_ITEM_ENDPOINT.replace("{item_id}", str(item_id))

        logger.debug(f"Fetching enriched data for item {item_id}")

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Retrieved enriched data for item {item_id}")
            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Item {item_id} not found (404)")
                return None
            logger.error(
                f"HTTP error for item {item_id}: {e.response.status_code}"
            )
            raise
        except Exception as e:
            logger.error(f"Error fetching item {item_id}: {e}")
            raise

    def process_enriched_item_data(
        self, enriched_data: dict[str, Any], item_id: int
    ) -> dict[str, Any]:
        """
        Process raw enriched item data and extract configured fields.

        Args:
            enriched_data: Raw enriched data from API
            item_id: Item ID (used for validation)

        Returns:
            Dictionary with extracted enriched item fields
        """
        result = {}

        # Extract basic fields (amLotId, amAuctionId)
        result["amLotId"] = enriched_data.get("amLotId", item_id)
        result["amAuctionId"] = enriched_data.get("amAuctionId")

        # Extract fields from generatedDescription object
        generated_desc = enriched_data.get("generatedDescription", {})
        if generated_desc:
            for field in config.ENRICHED_ITEM_GENERATED_DESCRIPTION_FIELDS:
                result[field] = generated_desc.get(field)

        # Extract nested JSON fields (stored as JSON strings)
        for field in config.ENRICHED_ITEM_JSON_FIELDS:
            value = generated_desc.get(field) if generated_desc else None
            if value is not None:
                # Store as JSON string
                result[field] = json.dumps(value)
            else:
                result[field] = None

        return result

    async def fetch_and_process_item(
        self, item_id: int
    ) -> dict[str, Any] | None:
        """
        Fetch and process enriched data for a single item.

        Args:
            item_id: Item ID to fetch

        Returns:
            Processed enriched item data or None if failed
        """
        try:
            enriched_data = await self.fetch_enriched_item(item_id)
            if enriched_data is None:
                return None

            processed_data = self.process_enriched_item_data(enriched_data, item_id)
            return processed_data
        except Exception as e:
            logger.error(f"Failed to fetch enriched data for item {item_id}: {e}")
            return None

    async def fetch_multiple_items_streaming(
        self,
        item_ids: list[int],
        progress_callback=None,
    ):
        """
        Fetch enriched data for multiple items with streaming (one at a time) to minimize memory.

        Args:
            item_ids: List of item IDs to fetch
            progress_callback: Optional callback function for progress updates

        Yields:
            Tuple of (item_id, enriched_data) for each successfully fetched item
        """
        for item_id in item_ids:
            try:
                result = await self.fetch_and_process_item(item_id)
                if progress_callback:
                    progress_callback(item_id, result is not None)

                if result is not None:
                    logger.info(f"Fetched enriched data for item {item_id}")
                    yield item_id, result

            except Exception as e:
                logger.error(f"Exception for item {item_id}: {e}")
                if progress_callback:
                    progress_callback(item_id, False)

    async def fetch_multiple_items(
        self,
        item_ids: list[int],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Fetch enriched data for multiple items with controlled concurrency.

        Args:
            item_ids: List of item IDs to fetch
            progress_callback: Optional callback function for progress updates

        Returns:
            List of enriched item data dictionaries
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_with_semaphore(item_id: int) -> dict[str, Any] | None:
            async with semaphore:
                result = await self.fetch_and_process_item(item_id)
                if progress_callback:
                    progress_callback(item_id, result is not None)
                return result

        tasks = [fetch_with_semaphore(iid) for iid in item_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        enriched_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception for item {item_ids[i]}: {result}")
            elif result is not None:
                enriched_items.append(result)

        logger.info(
            f"Successfully fetched {len(enriched_items)} enriched items from "
            f"{len(item_ids)} requests"
        )

        return enriched_items


# =============================================================================
# Data Transformation
# =============================================================================


def transform_enriched_item_data(items: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform enriched item data: rename fields and add 'enriched_item_' prefix to column names.

    Exception: item_id and auction_id do not get the prefix and remain as-is.

    Args:
        items: List of enriched item dictionaries

    Returns:
        DataFrame with transformed column names (all have enriched_item_ prefix except item_id and auction_id)
    """
    if not items:
        logger.warning("No enriched items to transform")
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # Build complete column mapping: rename + prefix in one operation
    # This avoids creating an intermediate DataFrame copy
    final_rename_map = {}
    for old_name in df.columns:
        # First apply field renaming (e.g., amLotId -> item_id)
        renamed = config.get_renamed_enriched_item_field_name(old_name)
        # Then apply prefix (e.g., title -> enriched_item_title, but skip item_id/auction_id)
        final_name = config.get_prefixed_enriched_item_field_name(renamed)
        final_rename_map[old_name] = final_name
    
    # Single rename operation instead of two
    df = df.rename(columns=final_rename_map)
    logger.debug(f"Transformed columns: {list(final_rename_map.keys())} -> {list(df.columns)}")

    return df


def write_batch_parquet(df_batch: pd.DataFrame, batch_file: Path) -> int:
    """
    Write a single batch to a parquet file.
    
    This replaces the O(n²) append pattern with O(n) writes.
    All batch files are merged once at the end.

    Args:
        df_batch: Batch data to write
        batch_file: Path to batch parquet file

    Returns:
        Number of rows written
    """
    df_batch.to_parquet(batch_file, index=False)
    logger.info(f"Wrote batch file {batch_file.name} with {len(df_batch)} records")
    return len(df_batch)


def merge_batch_parquets(batch_dir: Path, output_file: Path) -> int:
    """
    Merge all batch parquet files into final output file.
    
    This is done once at the end, avoiding O(n²) complexity of
    repeated read-concat-write operations.

    Args:
        batch_dir: Directory containing batch parquet files
        output_file: Final output parquet file

    Returns:
        Total number of rows in merged file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Find all batch files
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    
    if not batch_files:
        logger.warning("No batch files found to merge")
        return 0
    
    logger.info(f"Merging {len(batch_files)} batch files into {output_file}")
    
    # Read all batch tables
    tables = []
    for batch_file in batch_files:
        table = pq.read_table(batch_file)
        tables.append(table)
        logger.debug(f"Loaded {batch_file.name}: {len(table)} rows")
    
    # Concatenate all tables
    combined_table = pa.concat_tables(tables)
    
    # Write final file
    pq.write_table(combined_table, output_file)
    total_rows = len(combined_table)
    
    logger.info(f"Merged {len(batch_files)} batches into {output_file} ({total_rows:,} total rows)")
    
    # Clean up batch files
    for batch_file in batch_files:
        batch_file.unlink()
        logger.debug(f"Deleted batch file {batch_file.name}")
    
    return total_rows


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, progress_file: Path = config.ENRICHED_ITEM_PROGRESS_FILE):
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

    def mark_completed(self, item_id: int) -> None:
        """Mark item as successfully scraped."""
        self.completed_ids.add(item_id)
        self.failed_ids.discard(item_id)

    def mark_failed(self, item_id: int) -> None:
        """Mark item as failed."""
        self.failed_ids.add(item_id)

    def is_completed(self, item_id: int) -> bool:
        """Check if item was already scraped."""
        return item_id in self.completed_ids

    def filter_pending(self, item_ids: list[int]) -> list[int]:
        """Filter out already completed items."""
        return [iid for iid in item_ids if iid not in self.completed_ids]


# =============================================================================
# Multi-Core Processing Worker
# =============================================================================


def _process_batch_worker(
    batch_num: int,
    chunk_ids: list[int],
    rate_limit: int,
    max_workers: int,
    batch_dir: Path,
    use_progress_tracking: bool,
    progress_file: Path,
) -> tuple[int, int, int]:
    """
    Worker function to process a single batch in a separate process.
    
    This function runs in its own process with its own event loop,
    allowing true multi-core parallelism for I/O-bound operations.
    
    Args:
        batch_num: Batch number for file naming
        chunk_ids: List of item IDs to process
        rate_limit: Rate limit per worker
        max_workers: Concurrent workers per process
        batch_dir: Directory to write batch file
        use_progress_tracking: Whether to track progress
        progress_file: Path to progress tracking file
    
    Returns:
        Tuple of (batch_num, num_items, num_items_processed)
    """
    # Each process needs its own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Setup progress tracker if needed
        tracker = ProgressTracker(progress_file) if use_progress_tracking else None
        
        def progress_callback(item_id: int, success: bool) -> None:
            if tracker:
                if success:
                    tracker.mark_completed(item_id)
                else:
                    tracker.mark_failed(item_id)
                tracker.save()
        
        # Fetch enriched items for this chunk
        async def fetch_chunk():
            async with EnrichedItemDataFetcher(
                rate_limit=rate_limit,
                max_concurrent=max_workers,
            ) as fetcher:
                return await fetcher.fetch_multiple_items(
                    chunk_ids,
                    progress_callback=progress_callback if use_progress_tracking else None,
                )
        
        chunk_items = loop.run_until_complete(fetch_chunk())
        
        # Transform and write batch file
        if chunk_items:
            df_batch = transform_enriched_item_data(chunk_items)
            if not df_batch.empty:
                batch_file = batch_dir / f"batch_{batch_num:05d}.parquet"
                batch_row_count = write_batch_parquet(df_batch, batch_file)
                logger.info(
                    f"[Process {os.getpid()}] Batch {batch_num}: {batch_row_count} items from {len(chunk_ids)} requests"
                )
                return (batch_num, batch_row_count, len(chunk_ids))
        
        return (batch_num, 0, len(chunk_ids))
        
    finally:
        loop.close()


# =============================================================================
# Item ID Loading from Hugging Face
# =============================================================================


def load_item_ids_from_hf(limit: int | None = None) -> list[int]:
    """
    Load item IDs from Hugging Face dataset.

    Args:
        limit: Optional limit on number of item IDs to load

    Returns:
        List of item IDs
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading item IDs from Hugging Face: {config.HF_ITEM_DATASET_REPO}")

        # Load dataset from Hugging Face
        dataset = load_dataset(config.HF_ITEM_DATASET_REPO, split="train")

        # Extract item IDs
        id_column = config.HF_ITEM_ID_COLUMN
        if id_column not in dataset.column_names:
            logger.error(
                f"Column '{id_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
            # Try alternative column names
            possible_columns = ["item_id", "id", "amLotId"]
            for col in possible_columns:
                if col in dataset.column_names:
                    logger.info(f"Using alternative column: {col}")
                    id_column = col
                    break
            else:
                raise ValueError("Could not find item ID column in dataset")

        item_ids = dataset[id_column]

        # Convert to list and ensure integers
        item_ids = [int(iid) for iid in item_ids]

        # Remove duplicates and sort
        item_ids = sorted(set(item_ids))

        logger.info(f"Loaded {len(item_ids)} unique item IDs from Hugging Face")

        # Apply limit if specified
        if limit:
            item_ids = item_ids[:limit]
            logger.info(f"Limited to {limit} item IDs")

        return item_ids

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


async def scrape_enriched_items(
    item_ids: list[int] | None = None,
    limit: int | None = None,
    use_progress_tracking: bool = True,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    rate_limit: int = config.DEFAULT_RATE_LIMIT,
    output_file: Path | None = None,
    batch_size: int = 100,  # Process items in batches to reduce memory
) -> pd.DataFrame:
    """
    Scrape enriched item data from MaxSold API with batch processing for memory efficiency.

    Args:
        item_ids: List of item IDs to scrape (if None, loads from Hugging Face)
        limit: Maximum number of items to scrape
        use_progress_tracking: Whether to track and resume progress
        max_workers: Maximum parallel workers
        rate_limit: Maximum requests per second
        output_file: Output file path (if None, uses default from config)
        batch_size: Number of items to process per batch (default: 100)

    Returns:
        DataFrame with scraped enriched item data
    """
    # Load item IDs if not provided
    if item_ids is None:
        logger.info("Loading item IDs from Hugging Face")
        item_ids = load_item_ids_from_hf(limit=limit)
    else:
        # Apply limit if specified
        if limit:
            item_ids = item_ids[:limit]
            logger.info(f"Limited to {limit} items")

    # Filter out completed items if tracking progress
    if use_progress_tracking:
        tracker = ProgressTracker()
        original_count = len(item_ids)
        item_ids = tracker.filter_pending(item_ids)
        logger.info(
            f"After filtering completed: {len(item_ids)} pending "
            f"({original_count - len(item_ids)} already completed)"
        )
    else:
        tracker = None

    if not item_ids:
        logger.warning("No items to scrape")
        # Return existing data if available
        if output_file is None:
            output_file = config.ENRICHED_ITEM_PROCESSED_OUTPUT_DIR / config.ENRICHED_ITEM_DATA_FILENAME
        if output_file.exists():
            return pd.read_parquet(output_file)
        return pd.DataFrame()

    # Setup output file and batch directory
    if output_file is None:
        output_file = config.ENRICHED_ITEM_PROCESSED_OUTPUT_DIR / config.ENRICHED_ITEM_DATA_FILENAME
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for batch files
    batch_dir = output_file.parent / ".batches"
    batch_dir.mkdir(exist_ok=True)

    # Determine number of CPU cores to use
    num_cores = max(1, os.cpu_count() - 2) if os.cpu_count() else 1
    logger.info(f"Using {num_cores} CPU cores for parallel processing")

    # Split work into chunks
    chunks = []
    for chunk_start in range(0, len(item_ids), batch_size):
        chunk_ids = item_ids[chunk_start : chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        chunks.append((chunk_num, chunk_ids))
    
    total_chunks = len(chunks)
    logger.info(
        f"Starting to scrape enriched data from {len(item_ids)} items "
        f"({total_chunks} batches, {max_workers} workers per batch, "
        f"{num_cores} parallel processes)..."
    )

    total_items_scraped = 0
    items_processed = 0
    batch_files_written = []

    # Get progress file path for workers
    progress_file = config.ENRICHED_ITEM_PROGRESS_FILE if use_progress_tracking else None

    # Process batches in parallel across CPU cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all batch jobs to the pool
        future_to_batch = {}
        for chunk_num, chunk_ids in chunks:
            future = executor.submit(
                _process_batch_worker,
                chunk_num,
                chunk_ids,
                rate_limit,
                max_workers,
                batch_dir,
                use_progress_tracking,
                progress_file,
            )
            future_to_batch[future] = (chunk_num, len(chunk_ids))
        
        # Process results as they complete
        for future in as_completed(future_to_batch):
            chunk_num, num_items = future_to_batch[future]
            try:
                batch_num, num_rows, items_in_batch = future.result()
                total_items_scraped += num_rows
                items_processed += items_in_batch
                
                if num_rows > 0:
                    batch_files_written.append(batch_dir / f"batch_{batch_num:05d}.parquet")
                
                logger.info(
                    f"Completed batch {batch_num}/{total_chunks}: "
                    f"{num_rows} items, Progress: {items_processed}/{len(item_ids)} items "
                    f"(Total: {total_items_scraped:,} items)"
                )
            except Exception as e:
                logger.error(f"Batch {chunk_num} failed with error: {e}")

    # Merge all batch files into final output file
    if batch_files_written:
        logger.info(f"Merging {len(batch_files_written)} batch files...")
        total_rows = merge_batch_parquets(batch_dir, output_file)
        
        # Clean up batch directory
        try:
            batch_dir.rmdir()
            logger.debug(f"Removed batch directory {batch_dir}")
        except Exception as e:
            logger.warning(f"Could not remove batch directory: {e}")
        
        # Load and return final result
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
    Upload scraped enriched item data to Hugging Face Datasets.

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
        data_file = config.ENRICHED_ITEM_PROCESSED_OUTPUT_DIR / config.ENRICHED_ITEM_DATA_FILENAME

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    # Setup repo - use enriched item dataset repo if available
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
            "name": config.HF_ENRICHED_ITEM_DATASET_NAME,
            "description": config.HF_ENRICHED_ITEM_DATASET_DESCRIPTION,
            "license": config.HF_ENRICHED_ITEM_DATASET_LICENSE,
            "tags": config.HF_ENRICHED_ITEM_DATASET_TAGS,
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
    """Command-line interface for enriched item scraper."""
    parser = argparse.ArgumentParser(description="Scrape enriched item data from MaxSold API")
    parser.add_argument(
        "--item-ids",
        type=int,
        nargs="+",
        help="Specific item IDs to scrape",
    )
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
    config.ensure_directories()
    config.ENRICHED_ITEM_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ENRICHED_ITEM_PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run scraper
    logger.info("Starting enriched item data scraper...")

    df = asyncio.run(
        scrape_enriched_items(
            item_ids=args.item_ids,
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
