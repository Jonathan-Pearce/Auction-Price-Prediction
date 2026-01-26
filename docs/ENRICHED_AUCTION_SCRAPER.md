# Enriched Auction Data Scraper

## Overview

The enriched auction data scraper collects additional metadata about MaxSold auctions, including:
- Auction type and category
- Geographic location (city, region, postal code, coordinates)
- Display region information

This data complements the main auction data by providing contextual information useful for price prediction models.

## API Endpoint

**URL**: `https://api.maxsold.com/sales/am/{auction_id}`

**Method**: GET

**Example**: `https://api.maxsold.com/sales/am/103293`

## Data Fields Extracted

### Raw API Fields
- `amAuctionId` - Auction identifier
- `type` - Type of auction (e.g., estate_sale, moving_sale)
- `category` - Primary category (e.g., Home & Garden, Furniture)
- `displayRegion` - Human-readable location string
- `approxLocation` - Nested location object containing:
  - `city` - City name
  - `countryCode` - ISO country code (e.g., CA, US)
  - `regionCode` - State/province code (e.g., ON, BC)
  - `postalCode` - Postal/zip code prefix
  - `latLng` - Coordinates object with `lat` and `lng`

### Transformed Output Fields

All fields are prefixed with `enriched_auction_` except `auction_id`:

- `auction_id` - Renamed from `amAuctionId` (no prefix)
- `enriched_auction_type`
- `enriched_auction_category`
- `enriched_auction_displayRegion`
- `enriched_auction_approxLocation_city`
- `enriched_auction_approxLocation_countryCode`
- `enriched_auction_approxLocation_regionCode`
- `enriched_auction_approxLocation_postalCode`
- `enriched_auction_approxLocation_lat`
- `enriched_auction_approxLocation_lng`

## Usage

### Command Line

```bash
# Basic usage - loads auction IDs from Hugging Face
python -m src.data.enriched_auction_scraper

# With specific auction IDs
python -m src.data.enriched_auction_scraper --auction-ids 103293 99941 99942

# With limit
python -m src.data.enriched_auction_scraper --limit 100

# With custom workers and rate limiting
python -m src.data.enriched_auction_scraper --workers 10 --rate-limit 5

# Upload to Hugging Face after scraping
python -m src.data.enriched_auction_scraper --upload-hf --hf-repo username/repo

# Full example
python -m src.data.enriched_auction_scraper \
    --limit 500 \
    --workers 5 \
    --rate-limit 10 \
    --output data/processed/my_enriched_data.parquet \
    --upload-hf \
    --hf-repo jpearce610/enriched-auction-data
```

### Python API

```python
import asyncio
from src.data.enriched_auction_scraper import scrape_enriched_auctions

# Basic usage
df = asyncio.run(scrape_enriched_auctions())

# With specific auction IDs
df = asyncio.run(scrape_enriched_auctions(
    auction_ids=[103293, 99941, 99942]
))

# With custom parameters
df = asyncio.run(scrape_enriched_auctions(
    limit=100,
    max_workers=10,
    rate_limit=5,
    use_progress_tracking=True
))

print(df.head())
```

## Configuration

All configuration is managed in `src/data/scraper_config.yaml`:

```yaml
api:
  enriched_base_url: "https://api.maxsold.com"
  endpoints:
    enriched_auction: "/sales/am/{auction_id}"

fields:
  enriched_auction_fields:
    - amAuctionId
    - type
    - category
    - displayRegion
    - approxLocation

  enriched_auction_field_rename_map:
    amAuctionId: auction_id
    amLotId: item_id

  enriched_auction_column_prefix: "enriched_auction_"

storage:
  input:
    hf_dataset_repo: "jpearce610/auction_data"
    hf_auction_id_column: "auction_id"

  output:
    enriched_auction_raw_directory: "data/raw/enriched_auctions"
    enriched_auction_processed_directory: "data/processed/enriched_auctions"
    enriched_auction_data_filename: "enriched_auction_data.parquet"
```

## Features

### Parallel Processing
- Configurable concurrent requests (default: 5)
- Rate limiting to respect API constraints (default: 10 req/sec)
- Batch processing for memory efficiency

### Automatic Retries
- Exponential backoff on failures
- Configurable retry attempts (default: 3)
- Handles transient network errors

### Progress Tracking
- Saves progress to JSON file
- Resumes from last completed auction on restart
- Tracks both completed and failed auctions

### Data Transformation
- Automatic field renaming
- Prefix application (except auction_id and item_id)
- Nested structure flattening (location data)
- Parquet output format

### Hugging Face Integration
- Direct upload to Hugging Face Datasets
- Automatic metadata generation
- Token-based authentication

## Output

The scraper produces:

1. **Parquet file**: `data/processed/enriched_auctions/enriched_auction_data.parquet`
   - Columnar format for efficient storage and querying
   - Contains all transformed enriched auction records

2. **Metadata file**: `data/processed/enriched_auctions/metadata.json`
   - Dataset statistics (row count, columns)
   - Timestamp of creation
   - License and description

3. **Progress file**: `data/raw/enriched_auctions/enriched_auction_scraper_progress.json`
   - Tracking of completed and failed auctions
   - Used for resumption

## Example Output

```python
import pandas as pd

df = pd.read_parquet('data/processed/enriched_auctions/enriched_auction_data.parquet')
print(df.head())

# Output:
#   auction_id  enriched_auction_type  enriched_auction_category  ...
#       103293         estate_sale            Home & Garden      ...
#        99941          moving_sale                Furniture      ...
```

## Error Handling

- **Network errors**: Automatic retry with exponential backoff
- **Missing data**: Fields set to None if not present in API response
- **Invalid responses**: Logged but don't stop scraper
- **Rate limiting**: Enforced via async semaphore and sleep

## Dependencies

- `httpx` - Async HTTP client
- `tenacity` - Retry logic
- `pandas` - Data manipulation
- `pyarrow` - Parquet file format
- `loguru` - Logging
- `datasets` - Hugging Face Datasets (optional)
- `huggingface-hub` - Hugging Face upload (optional)

## Testing

```bash
# Run tests
pytest tests/test_enriched_auction_scraper.py -v

# With coverage
pytest tests/test_enriched_auction_scraper.py --cov=src.data.enriched_auction_scraper
```

## Notes

- The API endpoint may not be publicly documented - use respectfully
- Consider rate limiting to avoid being blocked
- Auction IDs are loaded from the main auction dataset on Hugging Face
- Geographic coordinates are approximate and may not be precise
- Some auctions may not have complete location data

## Related Files

- `src/data/enriched_auction_scraper.py` - Main scraper implementation
- `src/data/scraper_config.yaml` - Configuration file
- `src/data/scraper_config.py` - Configuration helper functions
- `tests/test_enriched_auction_scraper.py` - Unit tests
