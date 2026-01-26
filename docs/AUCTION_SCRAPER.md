# Auction Data Scraper

This module provides functionality to scrape auction-level data from the MaxSold API with aggregated item metrics.

## Overview

The auction scraper is designed to:
1. Fetch auction data from MaxSold API
2. Aggregate item-level metrics (viewed counts, bid totals, image counts)
3. Transform field names according to specifications
4. Support parallel processing for efficient scraping
5. Upload results to Hugging Face Datasets

## Architecture

### Components

1. **scraper_config.yaml**: YAML configuration file with all constants and settings
2. **scraper_config.py**: Configuration loader that reads from YAML file
3. **auction_scraper.py**: Main scraper implementation with data fetching and processing
4. **test_scraper.py**: Validation tests for scraper functionality

### Data Flow

```
Auction IDs → API Fetcher → Data Processor → DataFrame Transform → Output File → Hugging Face
```

## Configuration

All configuration values are defined in `src/data/scraper_config.yaml`:

### API Settings
- `api.base_url`: MaxSold API base URL
- `api.parameters.items_limit`: Maximum items per request (default: 2500)
- `rate_limiting.requests_per_second`: Requests per second (default: 10)
- `rate_limiting.concurrent_requests`: Max concurrent requests (default: 5)

### Field Mappings
- `catalog_lots` → `item_count`
- `current_bid` → `winning_price`
- All fields get `auction_` prefix (configured in `fields.column_prefix`)

### Data Files
- Input: `data/raw/auction_location_data.parquet` (column: `amAuctionId`)
- Output: `data/processed/auctions/auction_data.parquet`

To modify configuration values, edit the `scraper_config.yaml` file. The Python module will automatically load the updated values.

## Usage

### Command Line

Basic usage:
```bash
python -m src.data.auction_scraper --limit 10
```

Scrape specific auctions:
```bash
python -m src.data.auction_scraper --auction-ids 99941 99942 99943
```

With parallel processing:
```bash
python -m src.data.auction_scraper --workers 10 --rate-limit 20
```

Upload to Hugging Face:
```bash
python -m src.data.auction_scraper --limit 100 --upload-hf --hf-repo username/dataset-name
```

### Python API

```python
import asyncio
from src.data.auction_scraper import scrape_auctions

# Scrape specific auctions
df = asyncio.run(scrape_auctions(
    auction_ids=[99941, 99942, 99943],
    max_workers=5,
    rate_limit=10
))

# Scrape from file with limit
df = asyncio.run(scrape_auctions(
    limit=100,
    use_progress_tracking=True
))
```

### Uploading to Hugging Face

```python
import asyncio
from src.data.auction_scraper import upload_to_huggingface

asyncio.run(upload_to_huggingface(
    repo_id="username/maxsold-auctions",
    private=False
))
```

## Features

### Parallel Processing

The scraper uses asyncio with controlled concurrency:
- Semaphore limits concurrent requests
- Rate limiting prevents API overload
- Progress tracking for resumption

### Data Aggregation

For each auction, the scraper aggregates:
- `total_viewed`: Sum of views across all items
- `total_winning_price`: Sum of current_bid (winning prices)
- `total_bid_count`: Sum of bid counts
- `total_images`: Count of all images across items

### Field Transformations

1. **Renaming**: `catalog_lots` → `item_count`, `current_bid` → `winning_price`
2. **Prefixing**: All columns get `auction_` prefix

Example:
```
id → auction_id
title → auction_title
catalog_lots → auction_item_count
total_winning_price → auction_total_winning_price
```

### Progress Tracking

The scraper maintains progress in `data/raw/auctions/scraper_progress.json`:
- Tracks completed auction IDs
- Tracks failed auction IDs
- Enables resumption after interruption

## Output Format

The output DataFrame contains columns:
- `auction_id`: Auction identifier
- `auction_title`: Auction title
- `auction_starts`: Start time
- `auction_ends`: End time
- `auction_item_count`: Number of items in auction
- `auction_total_viewed`: Total views across all items
- `auction_total_winning_price`: Sum of winning prices
- `auction_total_bid_count`: Total number of bids
- `auction_total_images`: Total number of images
- Additional auction metadata fields

## Error Handling

The scraper includes:
- Automatic retries with exponential backoff
- Rate limit protection
- Exception logging
- Graceful failure handling (returns partial results)

## Performance

### Rate Limiting
Default: 10 requests/second (configurable)

### Parallel Processing
Default: 5 concurrent requests (configurable)

### Example Timing
- Single auction: ~1-2 seconds
- 100 auctions: ~1-2 minutes (with 5 workers)
- 1000 auctions: ~15-20 minutes (with 10 workers)

## Testing

Run validation tests:
```bash
python scripts/test_scraper.py
```

Tests validate:
- Configuration loading
- Field transformations
- Data aggregation logic
- DataFrame creation
- Auction ID file loading

## Environment Variables

Set in `.env` file:
```bash
# Hugging Face (optional, for upload)
HF_TOKEN=your_token_here
HF_DATASET_REPO=your-username/dataset-name
HF_ORGANIZATION=your-org  # optional
```

## Integration with Existing Code

This scraper complements the existing scraper (`src/data/scraper.py`):
- **Existing scraper**: Item-level data with bid history
- **New scraper**: Auction-level aggregated data

Both can be used depending on the analysis needs:
- Use auction scraper for high-level auction analysis
- Use item scraper for detailed item and bid analysis

## YAML Configuration Structure

The scraper uses a YAML configuration file (`src/data/scraper_config.yaml`) with the following structure:

```yaml
api:
  base_url: "https://maxsold.maxsold.com/msapi"
  endpoints:
    auction_items: "/auctions/items"
  parameters:
    items_limit: 2500
    timeout: 30

rate_limiting:
  requests_per_second: 10
  concurrent_requests: 5

fields:
  auction_fields:
    - id
    - title
    - catalog_lots
    # ... more fields
  field_rename_map:
    catalog_lots: item_count
    current_bid: winning_price
  column_prefix: "auction_"

storage:
  input:
    file: "auction_location_data.parquet"
    directory: "data/raw"
  output:
    processed_directory: "data/processed/auctions"
```

To customize the scraper behavior:
1. Edit values in `scraper_config.yaml`
2. The Python module automatically loads the updated configuration
3. No code changes required for most configuration updates

## API Response Structure

Expected MaxSold API response:
```json
{
  "auction": {
    "id": 99941,
    "title": "Estate Sale",
    "starts": "2024-01-01T00:00:00",
    "ends": "2024-01-02T00:00:00",
    "catalog_lots": 150,
    ...
  },
  "items": [
    {
      "item_id": 1,
      "viewed": 100,
      "current_bid": 25.50,
      "bid_count": 5,
      "images": ["url1.jpg", "url2.jpg"]
    },
    ...
  ]
}
```

Or as a simple list of items:
```json
[
  {
    "item_id": 1,
    "viewed": 100,
    "current_bid": 25.50,
    "bid_count": 5,
    "images": ["url1.jpg", "url2.jpg"]
  },
  ...
]
```

## Troubleshooting

### No auction IDs loaded
Ensure `data/raw/auction_location_data.parquet` exists with `amAuctionId` column.

### Rate limit errors (429)
Reduce `--rate-limit` parameter or increase delays between requests.

### Connection errors
Check network connectivity and API availability.

### Memory issues with large datasets
Process in smaller batches using `--limit` parameter.

## Future Enhancements

Potential improvements:
- [ ] Incremental updates (only scrape new/changed auctions)
- [ ] Support for historical snapshots
- [ ] Enhanced error recovery
- [ ] Database integration (DuckDB)
- [ ] Data quality metrics and validation
- [ ] Automatic scheduling/cron support

## Contributing

When modifying the scraper:
1. Update configuration in `scraper_config.py`
2. Add tests to `test_scraper.py`
3. Run linting: `ruff check src/data/auction_scraper.py`
4. Format code: `black src/data/auction_scraper.py`
5. Update this documentation

## License

MIT License - See project LICENSE file for details.
