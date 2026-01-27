# Enriched Item Data Scraper Documentation

## Overview

The enriched item data scraper collects detailed item-level metadata from MaxSold's enriched API endpoint. This includes item attributes, brands, categories, condition information, and other structured data that enhances the basic item information.

## API Endpoint

**URL**: `https://api.maxsold.com/listings/am/{item_id}/enriched`

Example:
```
https://api.maxsold.com/listings/am/7433915/enriched
```

## Features

- **Memory Efficient**: Processes items in batches (default 100) with periodic writes to disk
- **Parallel Processing**: Configurable concurrency (10-20 workers) for faster scraping
- **Progress Tracking**: Resumable scraping with automatic progress saves
- **Rate Limiting**: Respects API rate limits with exponential backoff
- **PyArrow Integration**: Efficient parquet appending without loading entire files
- **Hugging Face Integration**: Automatic upload to Hugging Face Datasets

## Data Source

Item IDs are loaded from the existing Hugging Face dataset:
- **Repository**: `jpearce610/item_data`
- **Column**: `item_id`

## Fields Extracted

### Basic Fields
- `amLotId` → renamed to `item_id`
- `amAuctionId` → renamed to `auction_id`

### Fields from generatedDescription Object
Extracted directly (without intermediate prefix):
- `title`: Item title
- `slug`: URL-friendly slug
- `description`: Item description
- `qualitativeDescription`: Quality assessment
- `brand`: Brand name
- `seriesLine`: Product series
- `condition`: Condition rating
- `working`: Working status (boolean)
- `singleKeyItem`: Single key item flag (boolean)
- `numItems`: Number of items

### Nested JSON Fields
These are stored as JSON strings:
- `brands`: Array of brand objects with confidence scores
- `categories`: Array of category objects with confidence scores
- `items`: Array of item objects with quantities
- `attributes`: Array of attribute objects (name-value pairs)
- `photosTaken`: Array of photo filenames

## Column Naming

All columns (except `item_id` and `auction_id`) receive the `enriched_item_` prefix:

- `item_id` → `item_id` (no prefix)
- `auction_id` → `auction_id` (no prefix)
- `title` → `enriched_item_title`
- `brand` → `enriched_item_brand`
- `brands` → `enriched_item_brands`
- etc.

## Usage

### Command Line

Basic usage:
```bash
# Scrape all items from Hugging Face dataset
python -m src.data.enriched_item_scraper

# Or use Makefile
make scrape-enriched
```

With options:
```bash
# Scrape first 100 items (for testing)
python -m src.data.enriched_item_scraper --limit 100
make scrape-enriched-sample

# Custom batch size and workers
python -m src.data.enriched_item_scraper \
    --batch-size 50 \
    --workers 15 \
    --rate-limit 8

# Upload to Hugging Face after scraping
python -m src.data.enriched_item_scraper \
    --upload-hf \
    --hf-repo jpearce610/enriched_item_data

# Scrape specific item IDs
python -m src.data.enriched_item_scraper \
    --item-ids 7433915 7433916 7433917

# Disable progress tracking
python -m src.data.enriched_item_scraper --no-progress
```

### Python API

```python
import asyncio
from src.data.enriched_item_scraper import scrape_enriched_items

# Scrape all items
df = asyncio.run(scrape_enriched_items())

# Scrape specific items
df = asyncio.run(scrape_enriched_items(
    item_ids=[7433915, 7433916],
    max_workers=10,
    rate_limit=5
))

# Scrape with custom settings
df = asyncio.run(scrape_enriched_items(
    limit=1000,
    batch_size=100,
    max_workers=15,
    rate_limit=10,
    use_progress_tracking=True
))
```

### Upload to Hugging Face

```python
from src.data.enriched_item_scraper import upload_to_huggingface

# Upload dataset
await upload_to_huggingface(
    repo_id="jpearce610/enriched_item_data",
    private=False
)
```

## Output

### File Location
- **Raw data**: `data/raw/enriched_items/`
- **Processed data**: `data/processed/enriched_items/enriched_item_data.parquet`
- **Progress file**: `data/raw/enriched_items/enriched_item_scraper_progress.json`

### Schema

The output parquet file contains:
- `item_id`: Integer
- `auction_id`: Integer
- `enriched_item_title`: String
- `enriched_item_description`: String
- `enriched_item_brand`: String
- `enriched_item_condition`: String
- `enriched_item_working`: Boolean
- `enriched_item_numItems`: Integer
- `enriched_item_brands`: JSON String
- `enriched_item_categories`: JSON String
- `enriched_item_items`: JSON String
- `enriched_item_attributes`: JSON String
- `enriched_item_photosTaken`: JSON String

## Memory Efficiency

The scraper implements several memory optimization techniques:

1. **Batch Processing**: Items are processed in batches (default 100)
2. **Periodic Writes**: Data is written to disk after each batch
3. **Memory Cleanup**: Accumulated data is deleted after each write
4. **PyArrow Streaming**: Parquet appending without loading entire file
5. **Controlled Concurrency**: Limits parallel requests to avoid memory spikes

## Error Handling

- **404 Errors**: Items not found are logged and skipped (not marked as failures)
- **Rate Limiting**: Exponential backoff on rate limit errors
- **Network Errors**: Automatic retries with exponential backoff (max 3 attempts)
- **Validation Errors**: Logged but don't crash the scraper
- **Progress Recovery**: Failed items can be retried on next run

## Progress Tracking

The scraper maintains a progress file that tracks:
- **Completed items**: Successfully scraped item IDs
- **Failed items**: Items that encountered errors
- **Last updated**: Timestamp of last progress save

This enables:
- Resuming interrupted scraping sessions
- Skipping already-scraped items
- Tracking overall progress
- Identifying problematic items

## Configuration

All configuration is stored in `src/data/scraper_config.yaml`:

```yaml
# Enriched item fields configuration
fields:
  enriched_item_fields:
    - amLotId
    - amAuctionId
    - generatedDescription
  
  enriched_item_json_fields:
    - brands
    - categories
    - items
    - attributes
    - photosTaken
  
  enriched_item_generated_description_fields:
    - title
    - slug
    - description
    # ... etc

# Storage paths
storage:
  output:
    enriched_item_raw_directory: "data/raw/enriched_items"
    enriched_item_processed_directory: "data/processed/enriched_items"
    enriched_item_data_filename: "enriched_item_data.parquet"

# Hugging Face metadata
huggingface:
  enriched_item_dataset:
    name: "maxsold-enriched-item-data"
    description: "MaxSold enriched item data with detailed attributes"
    license: "cc-by-4.0"
    tags:
      - auction
      - price-prediction
      - enriched-items
```

## Testing

Run tests:
```bash
# All enriched item scraper tests
pytest tests/test_enriched_item_scraper.py -v

# Specific test
pytest tests/test_enriched_item_scraper.py::test_transform_enriched_item_data -v

# With coverage
pytest tests/test_enriched_item_scraper.py --cov=src.data.enriched_item_scraper
```

## Performance

Typical performance metrics:
- **Throughput**: 10-20 items/second (with rate limit of 10 req/s)
- **Memory**: ~100-200 MB per batch of 100 items
- **Batch write**: ~1-2 seconds per 100 items
- **Network**: ~100 KB per item (varies by data richness)

## Common Issues

### 404 Errors
Some items may not have enriched data available. These are logged but don't count as failures.

### Rate Limiting
If you see many rate limit errors, reduce the `--rate-limit` parameter:
```bash
python -m src.data.enriched_item_scraper --rate-limit 5
```

### Memory Issues
If running out of memory, reduce batch size:
```bash
python -m src.data.enriched_item_scraper --batch-size 50
```

### Slow Performance
Increase workers for faster scraping:
```bash
python -m src.data.enriched_item_scraper --workers 20
```

## Integration

The enriched item data complements the basic item data:

1. **Item Scraper** → Basic item info (price, bids, images)
2. **Enriched Item Scraper** → Detailed attributes (brand, condition, categories)
3. **Join on**: `item_id` and `auction_id`

Example join:
```python
import pandas as pd

# Load datasets
items_df = pd.read_parquet("data/processed/items/item_data.parquet")
enriched_df = pd.read_parquet("data/processed/enriched_items/enriched_item_data.parquet")

# Join on item_id and auction_id
combined_df = items_df.merge(
    enriched_df,
    on=["item_id", "auction_id"],
    how="left"  # Keep all items, even without enriched data
)
```

## Future Enhancements

Potential improvements:
- [ ] Add image download functionality for `photosTaken`
- [ ] Extract more structured data from nested JSON fields
- [ ] Add data quality validation and cleaning
- [ ] Implement deduplication logic
- [ ] Add support for incremental updates
- [ ] Create summary statistics dashboard

## Related Documentation

- [Item Scraper](ITEM_SCRAPER.md) - Basic item data scraping
- [Auction Scraper](AUCTION_SCRAPER.md) - Auction-level data scraping
- [Data Documentation](DATA.md) - Overall data collection strategy
