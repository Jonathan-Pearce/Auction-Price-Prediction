# =============================================================================
# Image Data Scraper
# =============================================================================
"""
Scraper for image data from MaxSold auctions.

This module implements the requirements from the issue:
1. Scrapes image URLs from MaxSold API
2. Downloads images with parallel processing
3. Processes images (resize to 224 pixels, maintain aspect ratio)
4. Uses MobileNetV3 ONNX model with OpenCV for feature extraction
5. Extracts 576-dim embeddings from the second-to-last layer
6. Uploads to Hugging Face

Usage:
    # From command line
    python -m src.data.image_scraper --limit 100

    # From Python
    from src.data.image_scraper import scrape_images
    df = await scrape_images(auction_ids=[99941, 99942])
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from loguru import logger
from onnx import helper as onnx_helper
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import PROJECT_ROOT
from src.data import scraper_config as config

# =============================================================================
# Image Embedding Extractor
# =============================================================================


class ImageEmbeddingExtractor:
    """
    Extracts image embeddings using MobileNetV3 ONNX model with OpenCV.

    The model extracts 576-dimensional feature embeddings from the
    second-to-last layer (after GlobalAveragePool and Flatten).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        target_size: tuple[int, int] = (224, 224),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to ONNX model file
            target_size: Target image size (width, height)
            mean: Normalization mean (default: ImageNet mean)
            std: Normalization std (default: ImageNet std)
        """
        # Get configuration
        model_config = config.IMAGE_MODEL_CONFIG
        preprocess_config = config.IMAGE_PREPROCESSING_CONFIG

        if model_path is None:
            model_path = PROJECT_ROOT / model_config["onnx_file"]
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.target_size = target_size or tuple(preprocess_config["target_size"])
        self.mean = np.array(mean or preprocess_config["mean"], dtype=np.float32)
        self.std = np.array(std or preprocess_config["std"], dtype=np.float32)

        # Expected embedding dimension
        self.embedding_dim = model_config["embedding_dim"]

        # ONNX session will be initialized lazily
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def _ensure_session(self) -> None:
        """Initialize ONNX runtime session with intermediate output."""
        if self._session is not None:
            return

        logger.info(f"Loading ONNX model from {self.model_path}")

        # Load and modify model to output intermediate layer
        model = onnx.load(str(self.model_path))

        # Add the flatten output as an additional output (576-dim embedding)
        flatten_output_name = "/Flatten_output_0"
        output_node = onnx_helper.make_tensor_value_info(
            flatten_output_name, onnx.TensorProto.FLOAT, [1, self.embedding_dim]
        )
        model.graph.output.append(output_node)

        # Create session from modified model with optimized settings
        model_bytes = model.SerializeToString()
        
        # Configure session options for balanced performance/CPU usage
        sess_options = ort.SessionOptions()
        # Limit to 4 cores max for CPU efficiency (reduce from 8 to lower CPU usage)
        max_threads = min(4, os.cpu_count() or 4)
        sess_options.intra_op_num_threads = max_threads  # Threads per operation
        sess_options.inter_op_num_threads = max_threads  # Threads between operations
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Sequential mode uses less CPU
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        self._session = ort.InferenceSession(
            model_bytes,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        logger.info(
            f"ONNX model loaded successfully. "
            f"Input: {self._input_name}, Embedding dim: {self.embedding_dim}"
        )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Preprocess image for model input using OpenCV.

        Args:
            image_bytes: Raw image bytes from HTTP response

        Returns:
            Preprocessed image as numpy array (1, 3, 224, 224) or None if failed
        """
        try:
            # Decode image from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Failed to decode image")
                return None

            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            max_dim = max(h, w)
            target_max = self.target_size[0]  # 224

            if max_dim > target_max:
                scale = target_max / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                # Use INTER_LINEAR instead of INTER_AREA for lower CPU usage
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad to target size (center padding)
            h, w = img.shape[:2]
            target_h, target_w = self.target_size

            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left

            img = cv2.copyMakeBorder(
                img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            # Normalize to [0, 1] and apply ImageNet normalization
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std

            # Convert to CHW format (channels first) and add batch dimension
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            return img.astype(np.float32)

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray | None:
        """
        Extract embedding from image bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            576-dimensional embedding as numpy array or None if failed
        """
        self._ensure_session()

        # Preprocess image
        preprocessed = self.preprocess_image(image_bytes)
        if preprocessed is None:
            return None

        try:
            # Run inference - outputs are [final_output, embedding]
            outputs = self._session.run(None, {self._input_name: preprocessed})

            # Return the embedding (second output, 576-dim)
            embedding = outputs[1].flatten()

            if len(embedding) != self.embedding_dim:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)}, "
                    f"expected {self.embedding_dim}"
                )
            else:
                logger.debug(f"Extracted {self.embedding_dim}-dim embedding successfully")

            return embedding

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None


# =============================================================================
# Image Data Fetcher
# =============================================================================


class ImageDataFetcher:
    """
    Fetches image data from MaxSold API and extracts embeddings.

    Features:
    - Async HTTP client with rate limiting
    - Automatic retries with exponential backoff
    - Parallel processing with controlled concurrency
    - Image embedding extraction using MobileNetV3
    """

    def __init__(
        self,
        rate_limit: int = config.DEFAULT_RATE_LIMIT,
        max_concurrent: int = config.CONCURRENT_REQUESTS,
        timeout: float | None = None,
    ):
        """
        Initialize the image data fetcher.

        Args:
            rate_limit: Maximum requests per second
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        http_config = config.IMAGE_HTTP_CONFIG

        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout or http_config["timeout"]
        self.min_interval = 1.0 / rate_limit
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

        # Embedding extractor (lazy initialization)
        self._extractor: ImageEmbeddingExtractor | None = None

    async def __aenter__(self) -> "ImageDataFetcher":
        """Initialize async HTTP client with connection pooling."""
        # Configure connection pooling for better performance
        limits = httpx.Limits(
            max_connections=100,  # Total connection pool
            max_keepalive_connections=50,  # Reuse connections
            keepalive_expiry=30.0,  # Keep connections alive
        )
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better multiplexing
            headers={
                "User-Agent": "AuctionPricePredictor/1.0 (Research Project)",
                "Accept": "image/*,application/json",
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
                "Client not initialized. Use 'async with ImageDataFetcher():'"
            )
        return self._client

    @property
    def extractor(self) -> ImageEmbeddingExtractor:
        """Get embedding extractor (lazy initialization)."""
        if self._extractor is None:
            self._extractor = ImageEmbeddingExtractor()
        return self._extractor

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
    async def fetch_auction_items(self, auction_id: int) -> list[dict[str, Any]]:
        """
        Fetch all items with image URLs for an auction.

        Args:
            auction_id: Auction ID to fetch items for

        Returns:
            List of item dictionaries with image information
        """
        await self._rate_limit()

        url = config.AUCTION_ITEMS_ENDPOINT
        params = {
            "auctionid": auction_id,
            "limit": config.DEFAULT_ITEMS_LIMIT,
        }

        logger.debug(f"Fetching items for auction {auction_id}")

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract items from response
            items = []

            if isinstance(data, dict):
                auction_data = data.get("auction", {})
                items_data = auction_data.get("items", [])

                if isinstance(items_data, dict):
                    items = list(items_data.values())
                elif isinstance(items_data, list):
                    items = items_data

            elif isinstance(data, list):
                items = data

            logger.info(f"Retrieved {len(items)} items for auction {auction_id}")
            return items

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error for auction {auction_id}: {e.response.status_code}"
            )
            raise
        except Exception as e:
            logger.error(f"Error fetching auction {auction_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=config.RETRY_DELAY, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def download_image(self, image_url: str) -> bytes | None:
        """
        Download image from URL.

        Args:
            image_url: URL of the image

        Returns:
            Image bytes or None if failed
        """
        try:
            response = await self.client.get(image_url)
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.debug(f"Failed to download image {image_url}: {e}")
            return None

    def extract_image_records(
        self, item: dict[str, Any], auction_id: int
    ) -> list[dict[str, Any]]:
        """
        Extract image records from item data.

        Args:
            item: Item data from API
            auction_id: Parent auction ID

        Returns:
            List of image record dictionaries
        """
        records = []
        item_id = item.get("id")
        images = item.get("images", [])

        if isinstance(images, list) and len(images) > 0:
            # Only process the first image from each item for speed
            img = images[0]
            idx = 0
            
            if isinstance(img, dict):
                # Image is an object with url property
                image_url = img.get("url") or img.get("src")
            elif isinstance(img, str):
                # Image is a direct URL string
                image_url = img
            else:
                image_url = None

            if image_url:
                records.append(
                    {
                        "auction_id": auction_id,
                        "item_id": item_id,
                        "image_index": idx,
                        "image_url": image_url,
                    }
                )

        return records

    async def process_image(
        self, record: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Download and process a single image.

        Args:
            record: Image record with URL

        Returns:
            Record with embedding or None if failed
        """
        image_url = record.get("image_url")
        if not image_url:
            return None

        try:
            logger.debug(f"Processing image: item {record.get('item_id')}, index {record.get('image_index')}")
            
            # Download image
            image_bytes = await self.download_image(image_url)
            if image_bytes is None:
                return None

            # Extract embedding
            embedding = self.extractor.extract_embedding(image_bytes)
            if embedding is None:
                return None

            logger.debug(f"Successfully processed image: item {record.get('item_id')}, index {record.get('image_index')}")
            
            # Return record with embedding (convert to float32 for space efficiency)
            return {
                "auction_id": record["auction_id"],
                "item_id": record["item_id"],
                "image_index": record["image_index"],
                "image_url": image_url,
                "embedding": embedding.astype('float32'),
            }

        except Exception as e:
            logger.debug(f"Error processing image {image_url}: {e}")
            return None

    async def process_auction_images(
        self, auction_id: int
    ) -> list[dict[str, Any]]:
        """
        Process all images for an auction.

        Args:
            auction_id: Auction ID

        Returns:
            List of processed image records with embeddings
        """
        try:
            # Fetch items for auction
            items = await self.fetch_auction_items(auction_id)

            # Extract image records from all items
            all_records = []
            for item in items:
                records = self.extract_image_records(item, auction_id)
                all_records.extend(records)

            if not all_records:
                logger.debug(f"No images found for auction {auction_id}")
                return []

            logger.info(
                f"Processing {len(all_records)} images for auction {auction_id}"
            )

            # Process images with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            processed_count = 0
            report_interval = max(1, len(all_records) // 10)  # Report every 10%

            async def process_with_semaphore(record):
                nonlocal processed_count
                async with semaphore:
                    result = await self.process_image(record)
                    processed_count += 1
                    
                    # Log progress every report_interval images
                    if processed_count % report_interval == 0:
                        pct = (processed_count / len(all_records)) * 100
                        logger.info(
                            f"Progress: {processed_count}/{len(all_records)} images processed ({pct:.1f}%)"
                        )
                    
                    return result

            tasks = [process_with_semaphore(r) for r in all_records]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out failures
            processed = []
            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"Image processing exception: {result}")
                elif result is not None:
                    processed.append(result)

            logger.info(
                f"Successfully processed {len(processed)}/{len(all_records)} "
                f"images for auction {auction_id}"
            )

            return processed

        except Exception as e:
            logger.error(f"Failed to process auction {auction_id}: {e}")
            return []

    async def fetch_multiple_auctions(
        self,
        auction_ids: list[int],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Process images for multiple auctions.

        Args:
            auction_ids: List of auction IDs
            progress_callback: Optional callback for progress updates

        Returns:
            Flattened list of all processed image records
        """
        all_records = []

        for idx, auction_id in enumerate(auction_ids, 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing auction {auction_id} ({idx}/{len(auction_ids)})")
                logger.info(f"{'='*60}")
                
                records = await self.process_auction_images(auction_id)
                all_records.extend(records)

                if progress_callback:
                    progress_callback(auction_id, len(records) > 0)

            except Exception as e:
                logger.error(f"Exception for auction {auction_id}: {e}")
                if progress_callback:
                    progress_callback(auction_id, False)

        logger.info(
            f"Processed {len(all_records)} images from {len(auction_ids)} auctions"
        )

        return all_records


def write_batch_parquet(df: pd.DataFrame, output_file: Path) -> int:
    """
    Write DataFrame to parquet file.
    
    Args:
        df: DataFrame to write
        output_file: Path to output file
    
    Returns:
        Number of rows written
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_file)
    return len(df)


def merge_batch_parquets(batch_dir: Path, output_file: Path) -> int:
    """
    Merge all batch parquet files into a single file.
    
    Args:
        batch_dir: Directory containing batch files
        output_file: Path to final output file
    
    Returns:
        Total number of rows in merged file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    
    if not batch_files:
        logger.warning("No batch files found to merge")
        return 0
    
    # Read and concatenate all batch files
    tables = [pq.read_table(f) for f in batch_files]
    combined_table = pa.concat_tables(tables)
    
    # Write combined table
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined_table, output_file)
    
    # Clean up batch files
    for batch_file in batch_files:
        batch_file.unlink()
    
    return len(combined_table)


# =============================================================================
# Data Transformation
# =============================================================================


def transform_image_data(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Transform image records to DataFrame format.

    Args:
        records: List of image records with embeddings

    Returns:
        DataFrame with image data and embeddings
    """
    if not records:
        logger.warning("No image records to transform")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Rename columns with prefix
    column_rename = {
        "auction_id": "auction_id",  # Keep as-is
        "item_id": "item_id",  # Keep as-is
        "image_index": "image_index",
        "embedding": "image_embedding",
    }

    df = df.rename(columns=column_rename)
    
    # Remove image_url column to save space (URLs not needed in storage)
    if 'image_url' in df.columns:
        df = df.drop(columns=['image_url'])
    
    # Ensure embeddings are float32 numpy arrays (not lists or float64)
    if 'image_embedding' in df.columns:
        df['image_embedding'] = df['image_embedding'].apply(
            lambda x: np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) or x.dtype != np.float32 else x
        )

    logger.info(f"Transformed {len(df)} image records")
    return df


def append_to_parquet_efficient(df_new: pd.DataFrame, output_file: Path) -> int:
    """
    Append DataFrame to parquet file efficiently.

    Args:
        df_new: New data to append
        output_file: Path to parquet file

    Returns:
        Total number of rows after append
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_file.exists():
        existing_table = pq.read_table(output_file)
        new_table = pa.Table.from_pandas(df_new, preserve_index=False)
        combined_table = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined_table, output_file)

        total_count = len(combined_table)
        logger.info(f"Appended {len(df_new)} rows (total: {total_count})")

        del existing_table
        del new_table
        del combined_table

        return total_count
    else:
        df_new.to_parquet(output_file, index=False)
        logger.info(f"Created {output_file} with {len(df_new)} records")
        return len(df_new)


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, progress_file: Path = config.IMAGE_PROGRESS_FILE):
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
        limit: Optional limit on number of auction IDs

    Returns:
        List of auction IDs
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading auction IDs from Hugging Face: {config.HF_DATASET_REPO}")

        dataset = load_dataset(config.HF_DATASET_REPO, split="train")

        id_column = config.HF_AUCTION_ID_COLUMN
        if id_column not in dataset.column_names:
            logger.error(f"Column '{id_column}' not found in dataset")
            possible_columns = ["auction_id", "amAuctionId", "id"]
            for col in possible_columns:
                if col in dataset.column_names:
                    logger.info(f"Using alternative column: {col}")
                    id_column = col
                    break
            else:
                raise ValueError("Could not find auction ID column")

        auction_ids = [int(aid) for aid in dataset[id_column]]
        auction_ids = sorted(set(auction_ids))

        logger.info(f"Loaded {len(auction_ids)} unique auction IDs")

        if limit:
            auction_ids = auction_ids[:limit]
            logger.info(f"Limited to {limit} auction IDs")

        return auction_ids

    except ImportError:
        logger.error("datasets library not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load auction IDs: {e}")
        raise


# =============================================================================
# Batch Processing Worker (for multiprocessing)
# =============================================================================


def _process_batch_worker(
    batch_num: int,
    chunk_ids: list[int],
    rate_limit: int,
    max_workers: int,
    batch_dir: Path,
    use_progress_tracking: bool,
    progress_file: Path | None,
) -> tuple[int, int, int]:
    """
    Worker function to process a batch of auctions in a separate process.
    
    This runs in its own process with its own event loop and ONNX session.
    Each process gets its own memory space, allowing true parallel CPU utilization.
    
    Args:
        batch_num: Batch number for tracking
        chunk_ids: List of auction IDs to process in this batch
        rate_limit: API rate limit
        max_workers: Max concurrent workers within this process
        batch_dir: Directory to write batch results
        use_progress_tracking: Whether to track progress
        progress_file: Path to progress file
    
    Returns:
        Tuple of (batch_num, num_rows_written, num_auctions_processed)
    """
    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Progress callback
        def progress_callback(auction_id: int, success: bool) -> None:
            if use_progress_tracking and progress_file:
                tracker = ProgressTracker(progress_file=progress_file)
                if success:
                    tracker.mark_completed(auction_id)
                else:
                    tracker.mark_failed(auction_id)
                tracker.save()
        
        # Fetch images for this batch
        async def fetch_chunk():
            async with ImageDataFetcher(
                rate_limit=rate_limit,
                max_concurrent=max_workers,
            ) as fetcher:
                return await fetcher.fetch_multiple_auctions(
                    chunk_ids,
                    progress_callback=progress_callback if use_progress_tracking else None,
                )
        
        chunk_records = loop.run_until_complete(fetch_chunk())
        
        # Transform and write batch file
        if chunk_records:
            df_batch = transform_image_data(chunk_records)
            if not df_batch.empty:
                batch_file = batch_dir / f"batch_{batch_num:05d}.parquet"
                batch_row_count = write_batch_parquet(df_batch, batch_file)
                logger.info(
                    f"[Process {os.getpid()}] Batch {batch_num}: {batch_row_count} images from {len(chunk_ids)} auctions"
                )
                return (batch_num, batch_row_count, len(chunk_ids))
        
        return (batch_num, 0, len(chunk_ids))
        
    finally:
        loop.close()


# =============================================================================
# Main Scraping Function
# =============================================================================


async def scrape_images(
    auction_ids: list[int] | None = None,
    limit: int | None = None,
    use_progress_tracking: bool = True,
    max_workers: int = config.DEFAULT_MAX_WORKERS,
    rate_limit: int = config.DEFAULT_RATE_LIMIT,
    output_file: Path | None = None,
    batch_size: int = 50,
    num_processes: int | None = None,
) -> pd.DataFrame:
    """
    Scrape image embeddings from MaxSold API.

    Args:
        auction_ids: List of auction IDs to scrape (if None, loads from HF)
        limit: Maximum number of auctions to scrape
        use_progress_tracking: Whether to track and resume progress
        max_workers: Maximum async workers per process for I/O concurrency
        rate_limit: Maximum requests per second
        output_file: Output file path
        batch_size: Auctions per batch for memory efficiency
        num_processes: Number of parallel processes (default: min(4, cpu_count))

    Returns:
        DataFrame with image embeddings
    """
    # Load auction IDs if not provided
    if auction_ids is None:
        logger.info("Loading auction IDs from Hugging Face")
        auction_ids = load_auction_ids_from_hf(limit=limit)
    else:
        if limit:
            auction_ids = auction_ids[:limit]
            logger.info(f"Limited to {limit} auctions")

    # Filter out completed auctions
    if use_progress_tracking:
        tracker = ProgressTracker()
        original_count = len(auction_ids)
        auction_ids = tracker.filter_pending(auction_ids)
        logger.info(
            f"After filtering: {len(auction_ids)} pending "
            f"({original_count - len(auction_ids)} already completed)"
        )
    else:
        tracker = None

    if not auction_ids:
        logger.warning("No auctions to scrape")
        if output_file is None:
            output_file = (
                config.IMAGE_PROCESSED_OUTPUT_DIR / config.IMAGE_DATA_FILENAME
            )
        if output_file.exists():
            return pd.read_parquet(output_file)
        return pd.DataFrame()

    # Setup output file
    if output_file is None:
        output_file = config.IMAGE_PROCESSED_OUTPUT_DIR / config.IMAGE_DATA_FILENAME
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Progress callback
    def progress_callback(auction_id: int, success: bool) -> None:
        if tracker:
            if success:
                tracker.mark_completed(auction_id)
            else:
                tracker.mark_failed(auction_id)
            tracker.save()

    # Determine optimal process count for balanced CPU usage
    # Default to 2 processes to reduce peak CPU load
    if num_processes is None:
        num_processes = min(2, os.cpu_count() or 2)
    
    # Adjust workers per process to balance total concurrency
    # Target: num_processes × max_workers ≈ 4-6 (for moderate CPU usage)
    target_concurrency = 6
    if max_workers * num_processes > target_concurrency:
        max_workers = max(2, target_concurrency // num_processes)
        logger.info(f"Adjusted workers to {max_workers} per process for moderate CPU usage")
    
    logger.info(
        f"Starting to scrape images from {len(auction_ids)} auctions "
        f"(batch size: {batch_size}, {num_processes} processes, {max_workers} workers/process)..."
    )

    # Split auctions into batches
    batches = []
    for chunk_start in range(0, len(auction_ids), batch_size):
        chunk_ids = auction_ids[chunk_start : chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        batches.append((chunk_num, chunk_ids))
    
    total_batches = len(batches)
    logger.info(f"Split into {total_batches} batches for parallel processing")

    # Use multiprocessing for parallel batch processing
    if num_processes > 1 and len(batches) > 1:
        # Create temporary directory for batch files
        batch_dir = output_file.parent / "_batches"
        batch_dir.mkdir(exist_ok=True)
        
        # Process batches in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for batch_num, chunk_ids in batches:
                future = executor.submit(
                    _process_batch_worker,
                    batch_num,
                    chunk_ids,
                    rate_limit,
                    max_workers,
                    batch_dir,
                    use_progress_tracking,
                    tracker.progress_file if tracker else None,
                )
                futures.append(future)
            
            # Wait for all batches to complete with progress tracking
            completed = 0
            for future in as_completed(futures):
                try:
                    batch_num, num_rows, num_auctions = future.result()
                    completed += 1
                    logger.info(
                        f"Batch {batch_num} complete: {num_rows} images from "
                        f"{num_auctions} auctions ({completed}/{total_batches} batches)"
                    )
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
        
        # Merge all batch files into final output
        logger.info("Merging batch files...")
        total_images = merge_batch_parquets(batch_dir, output_file)
        
        # Clean up batch directory
        try:
            batch_dir.rmdir()
        except:
            pass
    else:
        # Single-process fallback for small jobs
        logger.info("Using single-process mode")
        total_images = 0
        auctions_processed = 0

        for chunk_num, chunk_ids in batches:
            logger.info(
                f"Processing batch {chunk_num}/{total_batches} "
                f"({len(chunk_ids)} auctions)"
            )

            async with ImageDataFetcher(
                rate_limit=rate_limit,
                max_concurrent=max_workers,
            ) as fetcher:
                chunk_records = await fetcher.fetch_multiple_auctions(
                    chunk_ids,
                    progress_callback=progress_callback if use_progress_tracking else None,
                )

            auctions_processed += len(chunk_ids)

            if chunk_records:
                df_batch = transform_image_data(chunk_records)

                if not df_batch.empty:
                    total_images = append_to_parquet_efficient(df_batch, output_file)
                    logger.info(f"Saved batch {chunk_num}. Total: {total_images:,} images")

                del df_batch
                del chunk_records

            logger.info(f"Progress: {auctions_processed}/{len(auction_ids)} auctions")

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
    Upload image embeddings to Hugging Face Datasets.

    Args:
        data_file: Path to parquet file
        repo_id: HuggingFace repository ID
        private: Whether to make the dataset private
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi

        from src.config import settings
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        return

    # Setup paths
    if data_file is None:
        data_file = config.IMAGE_PROCESSED_OUTPUT_DIR / config.IMAGE_DATA_FILENAME

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    # Setup repo
    if repo_id is None:
        repo_id = settings.huggingface.dataset_id
        if not repo_id:
            logger.error("Hugging Face repository ID not configured")
            return

    # Validate token
    if not settings.huggingface.token:
        logger.error("Hugging Face token not found")
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
            "name": config.HF_IMAGE_DATASET_NAME,
            "description": config.HF_IMAGE_DATASET_DESCRIPTION,
            "license": config.HF_IMAGE_DATASET_LICENSE,
            "tags": config.HF_IMAGE_DATASET_TAGS,
            "num_records": len(df),
            "columns": list(df.columns),
            "embedding_dim": config.IMAGE_MODEL_CONFIG["embedding_dim"],
            "model": "MobileNetV3-small",
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

        # Upload metadata
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
    """Command-line interface for image scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape image embeddings from MaxSold API"
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
        "--batch-size",
        type=int,
        default=50,
        help="Auctions per batch (default: 50)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=f"Number of parallel processes (default: min(2, CPU count) = {min(2, os.cpu_count() or 2)}). Use 1-2 for low CPU, 3-4 for high performance",
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
    config.IMAGE_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGE_PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run scraper
    logger.info("Starting image data scraper...")

    df = asyncio.run(
        scrape_images(
            auction_ids=args.auction_ids,
            limit=args.limit,
            use_progress_tracking=not args.no_progress,
            max_workers=args.workers,
            rate_limit=args.rate_limit,
            output_file=Path(args.output) if args.output else None,
            batch_size=args.batch_size,
            num_processes=args.processes,
        )
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df)}")
    if not df.empty:
        logger.info(f"Columns: {', '.join(df.columns)}")
        if "image_embedding" in df.columns:
            embedding_dim = len(df["image_embedding"].iloc[0])
            logger.info(f"Embedding dimension: {embedding_dim}")

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
