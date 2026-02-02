# =============================================================================
# Auction Price Prediction - Data Scraper
# =============================================================================
"""
Orchestrates data collection from MaxSold API.

Features:
- Batch collection of multiple auctions
- Progress tracking and resumption
- Data validation and storage
- Export to Parquet/DuckDB
- Upload to Hugging Face Datasets

Usage:
    # Command line
    python -m src.data.scraper --limit 100

    # Python
    from src.data.scraper import run_scraper
    await run_scraper(auction_ids=[99941, 99942])
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, settings
from src.data.maxsold_client import MaxSoldClient
from src.data.schemas import Auction, AuctionRecord, BidRecord, ItemRecord

# =============================================================================
# Scraper Configuration
# =============================================================================


class ScraperConfig:
    """Configuration for scraper run."""

    def __init__(
        self,
        auction_ids: list[int] | None = None,
        start_id: int | None = None,
        end_id: int | None = None,
        limit: int | None = None,
        include_bids: bool = True,
        include_enriched: bool = False,
        output_dir: Path | None = None,
        batch_size: int = 100,
        max_concurrent: int = 5,
        save_raw: bool = True,
        save_processed: bool = True,
    ):
        self.auction_ids = auction_ids
        self.start_id = start_id
        self.end_id = end_id
        self.limit = limit
        self.include_bids = include_bids
        self.include_enriched = include_enriched
        self.output_dir = output_dir or RAW_DATA_DIR
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.save_raw = save_raw
        self.save_processed = save_processed


# =============================================================================
# Data Storage
# =============================================================================


class DataStorage:
    """Handles saving scraped data to disk and database."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_auction_json(self, auction: Auction) -> Path:
        """Save auction data as JSON."""
        import json

        filepath = self.output_dir / f"auction_{auction.auction_id}.json"
        with open(filepath, "w") as f:
            json.dump(auction.model_dump(mode="json"), f, indent=2, default=str)
        return filepath

    def save_batch_parquet(
        self,
        auctions: list[Auction],
        batch_name: str,
    ) -> dict[str, Path]:
        """
        Save batch of auctions as Parquet files.

        Creates three files:
        - auctions_{batch_name}.parquet
        - items_{batch_name}.parquet
        - bids_{batch_name}.parquet
        """
        import pandas as pd

        # Flatten to records
        auction_records = []
        item_records = []
        bid_records = []

        for auction in auctions:
            # Auction record
            auction_records.append(
                AuctionRecord(
                    auction_id=auction.auction_id,
                    title=auction.title,
                    auction_type=auction.auction_type,
                    city=auction.location.city if auction.location else None,
                    state=auction.location.state if auction.location else None,
                    country=auction.location.country if auction.location else None,
                    start_time=auction.start_time,
                    end_time=auction.end_time,
                    num_items=auction.num_items,
                    status=auction.status,
                ).model_dump()
            )

            # Item records
            for item in auction.items:
                item_records.append(
                    ItemRecord(
                        item_id=item.item_id,
                        auction_id=auction.auction_id,
                        lot_number=item.lot_number,
                        title=item.title,
                        description=item.description,
                        category=item.category,
                        starting_bid=item.starting_bid,
                        winning_price=item.winning_price,
                        num_bids=item.num_bids,
                        num_bidders=item.num_bidders,
                        primary_image_url=str(item.primary_image_url)
                        if item.primary_image_url
                        else None,
                        end_time=item.end_time,
                        has_bids=item.has_bids,
                    ).model_dump()
                )

                # Bid records
                for bid in item.bids:
                    bid_records.append(
                        BidRecord(
                            bid_id=bid.bid_id,
                            item_id=item.item_id,
                            auction_id=auction.auction_id,
                            bidder_id=bid.bidder_id,
                            amount=bid.amount,
                            bid_time=bid.bid_time,
                            is_winning=bid.is_winning,
                            extended_auction=bid.extended_auction,
                        ).model_dump()
                    )

        # Save to Parquet
        paths = {}

        if auction_records:
            df = pd.DataFrame(auction_records)
            path = self.output_dir / f"auctions_{batch_name}.parquet"
            df.to_parquet(path, index=False)
            paths["auctions"] = path

        if item_records:
            df = pd.DataFrame(item_records)
            path = self.output_dir / f"items_{batch_name}.parquet"
            df.to_parquet(path, index=False)
            paths["items"] = path

        if bid_records:
            df = pd.DataFrame(bid_records)
            path = self.output_dir / f"bids_{batch_name}.parquet"
            df.to_parquet(path, index=False)
            paths["bids"] = path

        logger.info(
            f"Saved batch {batch_name}: "
            f"{len(auction_records)} auctions, "
            f"{len(item_records)} items, "
            f"{len(bid_records)} bids"
        )

        return paths

    def save_to_duckdb(self, auctions: list[Auction]) -> None:
        """Insert auction data into DuckDB database."""

        # TODO: Implement DuckDB insertion
        # Use the schema from references/schema.sql
        logger.info(f"Would save {len(auctions)} auctions to DuckDB")


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Tracks scraping progress for resumption."""

    def __init__(self, output_dir: Path):
        self.progress_file = output_dir / "scraper_progress.json"
        self.completed_ids: set[int] = set()
        self.failed_ids: set[int] = set()
        self._load()

    def _load(self) -> None:
        """Load progress from file."""
        import json

        if self.progress_file.exists():
            with open(self.progress_file) as f:
                data = json.load(f)
                self.completed_ids = set(data.get("completed", []))
                self.failed_ids = set(data.get("failed", []))

    def save(self) -> None:
        """Save progress to file."""
        import json

        with open(self.progress_file, "w") as f:
            json.dump(
                {
                    "completed": list(self.completed_ids),
                    "failed": list(self.failed_ids),
                    "last_updated": datetime.utcnow().isoformat(),
                },
                f,
            )

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
# Main Scraper
# =============================================================================


async def run_scraper(
    config: ScraperConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run the data scraper.

    Args:
        config: Scraper configuration
        **kwargs: Override config options

    Returns:
        Dictionary with scraping results
    """
    config = config or ScraperConfig(**kwargs)

    # Initialize components
    storage = DataStorage(config.output_dir)
    progress = ProgressTracker(config.output_dir)

    # Determine which auctions to scrape
    if config.auction_ids:
        auction_ids = config.auction_ids
    elif config.start_id and config.end_id:
        auction_ids = list(range(config.start_id, config.end_id + 1))
    else:
        # TODO: Implement auction discovery
        # For now, require explicit auction IDs
        logger.error("No auction IDs specified. Provide auction_ids or start_id/end_id.")
        return {"error": "No auction IDs specified"}

    # Filter out completed
    pending_ids = progress.filter_pending(auction_ids)
    if config.limit:
        pending_ids = pending_ids[: config.limit]

    logger.info(
        f"Scraping {len(pending_ids)} auctions "
        f"({len(auction_ids) - len(pending_ids)} already completed)"
    )

    # Scrape in batches
    results = {
        "total": len(pending_ids),
        "completed": 0,
        "failed": 0,
        "items_collected": 0,
        "bids_collected": 0,
    }

    async with MaxSoldClient() as client:
        for i in range(0, len(pending_ids), config.batch_size):
            batch_ids = pending_ids[i : i + config.batch_size]
            batch_name = f"batch_{i // config.batch_size:04d}"

            logger.info(f"Processing {batch_name}: auctions {batch_ids[0]}-{batch_ids[-1]}")

            # Fetch batch
            auctions = await client.get_multiple_auctions(
                auction_ids=batch_ids,
                include_bids=config.include_bids,
                include_enriched=config.include_enriched,
                max_concurrent=config.max_concurrent,
            )

            # Update progress
            for auction in auctions:
                progress.mark_completed(auction.auction_id)
                results["completed"] += 1
                results["items_collected"] += len(auction.items)
                results["bids_collected"] += sum(
                    len(item.bids) for item in auction.items
                )

            # Mark failed
            fetched_ids = {a.auction_id for a in auctions}
            for aid in batch_ids:
                if aid not in fetched_ids:
                    progress.mark_failed(aid)
                    results["failed"] += 1

            # Save batch
            if config.save_raw and auctions:
                storage.save_batch_parquet(auctions, batch_name)

            # Save progress
            progress.save()

            logger.info(
                f"Batch complete: {results['completed']}/{results['total']} auctions, "
                f"{results['items_collected']} items, {results['bids_collected']} bids"
            )

    return results


# =============================================================================
# Hugging Face Upload
# =============================================================================


async def upload_to_huggingface(
    data_dir: Path | None = None,
    repo_id: str | None = None,
) -> None:
    """
    Upload scraped data to Hugging Face Datasets.

    Args:
        data_dir: Directory with Parquet files
        repo_id: Hugging Face dataset repository ID
    """

    data_dir = data_dir or PROCESSED_DATA_DIR
    repo_id = repo_id or settings.huggingface.dataset_id

    logger.info(f"Uploading data from {data_dir} to {repo_id}")

    # Load all parquet files
    # TODO: Implement proper dataset creation and upload
    # - Combine all batch files
    # - Create train/val/test splits
    # - Upload to HF Hub

    raise NotImplementedError("HF upload not yet implemented")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for scraper CLI."""
    parser = argparse.ArgumentParser(description="Scrape MaxSold auction data")
    parser.add_argument(
        "--auction-ids",
        type=int,
        nargs="+",
        help="Specific auction IDs to scrape",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        help="Start of auction ID range",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        help="End of auction ID range",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of auctions to scrape",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in sample mode (small subset for testing)",
    )
    parser.add_argument(
        "--no-bids",
        action="store_true",
        help="Skip fetching bid history",
    )
    parser.add_argument(
        "--include-enriched",
        action="store_true",
        help="Fetch enriched item data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Auctions per batch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for scraped data",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload to Hugging Face after scraping",
    )

    args = parser.parse_args()

    # Configure
    config = ScraperConfig(
        auction_ids=args.auction_ids,
        start_id=args.start_id,
        end_id=args.end_id,
        limit=args.limit or (10 if args.sample else None),
        include_bids=not args.no_bids,
        include_enriched=args.include_enriched,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        batch_size=args.batch_size,
    )

    # Run scraper
    logger.info("Starting MaxSold scraper...")
    results = asyncio.run(run_scraper(config))

    # Print results
    logger.info("Scraping complete!")
    logger.info(f"  Auctions completed: {results.get('completed', 0)}")
    logger.info(f"  Auctions failed: {results.get('failed', 0)}")
    logger.info(f"  Items collected: {results.get('items_collected', 0)}")
    logger.info(f"  Bids collected: {results.get('bids_collected', 0)}")

    # Upload to HF if requested
    if args.upload_hf:
        asyncio.run(upload_to_huggingface())


if __name__ == "__main__":
    main()
