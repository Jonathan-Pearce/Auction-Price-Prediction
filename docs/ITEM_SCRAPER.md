# Item Data Scraper

## Overview

The item data scraper collects detailed information about auction items from MaxSold's API. It loads auction IDs from the Hugging Face dataset and scrapes item-level data for each auction.

## Features

- **Hugging Face Integration**: Automatically loads auction IDs from the `jpearce610/auction_data` dataset
- **Parallel Processing**: Concurrent scraping with configurable workers and rate limiting
- **Progress Tracking**: Resumes from where it left off if interrupted
- **Data Validation**: Handles missing fields and different response structures gracefully
- **Field Transformation**: Automatically adds `item_` prefix to all column names

## Usage

### Command Line

Basic usage (loads auction IDs from Hugging Face):
```bash
python -m src.data.item_scraper
```

With Makefile:
```bash
make scrape-items
```

Scrape a small sample (10 auctions):
```bash
python -m src.data.item_scraper --limit 10
# or
make scrape-items-sample
```

Scrape specific auction IDs:
```bash
python -m src.data.item_scraper --auction-ids 99941 99942 99943
```

Upload to Hugging Face after scraping:
```bash
python -m src.data.item_scraper --upload-hf --hf-repo your-username/item-data
```

### Options

- `--auction-ids`: Specific auction IDs to scrape (space-separated)
- `--limit`: Maximum number of auctions to scrape
- `--no-progress`: Disable progress tracking
- `--workers`: Number of parallel workers (default: 5)
- `--rate-limit`: Requests per second (default: 10)
- `--output`: Custom output file path
- `--upload-hf`: Upload to Hugging Face after scraping
- `--hf-repo`: Hugging Face repository ID for upload
- `--hf-private`: Make the Hugging Face dataset private

### Python API

```python
import asyncio
from src.data.item_scraper import scrape_items

# Scrape items from specific auctions
df = asyncio.run(scrape_items(auction_ids=[99941, 99942]))

# Load auction IDs from Hugging Face and scrape
df = asyncio.run(scrape_items(limit=100))

# Customize scraping parameters
df = asyncio.run(
    scrape_items(
        limit=50,
        max_workers=10,
        rate_limit=5,
        use_progress_tracking=True
    )
)
```

## Data Fields

The scraper extracts the following fields for each item (all prefixed with `item_`):

| Original Field    | Output Field                | Description                         |
|-------------------|----------------------------|-------------------------------------|
| `id`              | `item_id`                  | Unique item identifier              |
| `auction_id`      | `item_auction_id`          | Parent auction ID                   |
| `title`           | `item_title`               | Item title/name                     |
| `description`     | `item_description`         | Item description                    |
| `viewed`          | `item_viewed`              | Number of times viewed              |
| `starting_bid`    | `item_starting_bid`        | Starting bid amount                 |
| `current_bid`     | `item_current_bid`         | Current highest bid                 |
| `proxy_bid`       | `item_proxy_bid`           | Proxy bid amount                    |
| `start_time`      | `item_start_time`          | Item bidding start time             |
| `end_time`        | `item_end_time`            | Item bidding end time               |
| `bid_count`       | `item_bid_count`           | Total number of bids                |
| `bidding_extended`| `item_bidding_extended`    | Whether bidding was extended        |
| `images`          | `item_images`              | List of image URLs                  |
| (computed)        | `item_number_of_images`    | Count of images for the item        |

## Output

### File Structure

```
data/
├── raw/
│   └── items/
│       └── item_scraper_progress.json    # Progress tracking
└── processed/
    └── items/
        ├── item_data.parquet             # Main output
        └── metadata.json                 # Dataset metadata
```

### Output Format

- **Format**: Parquet (efficient columnar storage)
- **Columns**: All fields prefixed with `item_`
- **Rows**: One row per item

Example output:
```
   item_id  item_auction_id      item_title  item_viewed  item_starting_bid  item_current_bid  item_bid_count  item_number_of_images
0     1001            99941  Antique Chair          150               10.0              45.0               8                      3
1     1002            99941  Vintage Table          200               25.0               0.0               0                      1
2     1003            99941       Lamp Set           75                5.0              12.0               3                      2
```

## Configuration

All configuration is stored in `src/data/scraper_config.yaml`:

```yaml
# Item-specific configuration
fields:
  item_fields:
    - id
    - auction_id
    - title
    - description
    - viewed
    - starting_bid
    - current_bid
    - proxy_bid
    - start_time
    - end_time
    - bid_count
    - bidding_extended
    - images
  
  item_column_prefix: "item_"

storage:
  input:
    hf_dataset_repo: "jpearce610/auction_data"
    hf_auction_id_column: "auction_id"
  
  output:
    item_processed_directory: "data/processed/items"
    item_data_filename: "item_data.parquet"
```

## Progress Tracking

The scraper maintains progress in `data/raw/items/item_scraper_progress.json`:

```json
{
  "completed": [99941, 99942, 99943],
  "failed": [99944],
  "last_updated": "2024-01-26T00:00:00"
}
```

This allows the scraper to:
- Resume from where it left off if interrupted
- Skip already-scraped auctions
- Track failed auctions for retry

Disable with `--no-progress` flag.

## API Endpoint

The scraper uses MaxSold's auction items endpoint:

```
GET https://maxsold.maxsold.com/msapi/auctions/items
Parameters:
  - auctionid: {auction_id}
  - limit: 2500
```

## Rate Limiting

Default rate limiting:
- **10 requests/second** (configurable with `--rate-limit`)
- **5 concurrent requests** (configurable with `--workers`)
- **Automatic retries** with exponential backoff (up to 3 attempts)

## Special Cases

### Zero-Bid Items
Items with no bids are included with:
- `item_bid_count = 0`
- `item_current_bid = 0.0`

These are valid data points for training models to predict "no bid" scenarios.

### Missing Fields
If a field is missing from the API response, it's set to `None` (null) in the output.

### Image Counting
The `item_number_of_images` field is computed by:
- Counting items in the `images` list (if it's a list)
- Using the value directly (if it's an integer)
- Defaulting to 0 (if missing or invalid)

## Uploading to Hugging Face

After scraping, upload the dataset:

```bash
# Set your HF token
export HF_TOKEN=your_token_here

# Upload to default repo
python -m src.data.item_scraper --upload-hf

# Upload to custom repo
python -m src.data.item_scraper --upload-hf --hf-repo username/repo-name
```

Or programmatically:

```python
from src.data.item_scraper import upload_to_huggingface
import asyncio

asyncio.run(upload_to_huggingface(
    repo_id="username/item-data",
    private=False
))
```

## Testing

Run the test suite:

```bash
pytest tests/test_item_scraper.py -v
```

Run integration test with mock data:

```bash
python scripts/test_item_scraper_integration.py
```

## Troubleshooting

### Network Errors
If you encounter network errors, the scraper will:
1. Retry up to 3 times with exponential backoff
2. Mark the auction as failed in progress file
3. Continue with other auctions

### Memory Issues
If scraping many auctions causes memory issues:
- Reduce `--workers` (default: 5)
- Scrape in smaller batches with `--limit`

### Progress Not Saved
Ensure the output directory exists and is writable:
```bash
mkdir -p data/raw/items
mkdir -p data/processed/items
```

### Hugging Face Upload Fails
Verify:
- `HF_TOKEN` environment variable is set
- Token has write access to the repository
- Repository exists (or will be created automatically)

## Performance

Typical performance (depends on network speed and MaxSold's API):
- **~10 auctions/minute** with default settings
- **~100-1000 items/minute** (depending on auction size)
- **Progress saved** after each auction

For 1000 auctions with average 100 items each:
- Estimated time: ~2 hours
- Output size: ~10-20 MB (Parquet format)
