# Combined Engineered Datasets

This document describes the pipeline for combining engineered auction, item, and bid datasets.

## Overview

The `combine_engineered_datasets.py` module provides a pipeline to download, merge, and upload engineered feature datasets from Hugging Face.

## Datasets Combined

The pipeline combines three engineered datasets:

1. **Auction Features** (`jpearce610/engineered_auction_data`)
   - Auction-level features (duration, location, totals, etc.)
   - One row per auction
   - Key: `auction_id`

2. **Item Features** (`jpearce610/engineered_item_data`)
   - Item-level features (title length, images, price, etc.)
   - One row per item
   - Keys: `auction_id`, `item_id`

3. **Bid Features** (`jpearce610/engineered_bid_data`)
   - Bid-level features (time series, velocity, increments, etc.)
   - One row per bid
   - Keys: `auction_id`, `item_id`

## Merge Strategy

The pipeline performs a **full outer merge** in two steps:

1. **Step 1**: Merge auction + item data on `auction_id`
2. **Step 2**: Merge result with bid data on `auction_id` and `item_id`

### Key Features

- **Full Outer Join**: Keeps all records from all datasets
- **Column Ordering**: Ensures `auction_id` and `item_id` are the first two columns
- **Column Preservation**: Keeps all columns from all datasets (with suffixes for conflicts)
- **Memory Efficiency**: Loads bid data efficiently to handle large datasets

## Usage

### Command Line

```bash
# Combine and save locally only
python -m src.combine_engineered_datasets

# Combine and upload to Hugging Face
python -m src.combine_engineered_datasets --upload

# Specify custom output location
python -m src.combine_engineered_datasets --output data/my_combined_data.parquet

# Specify custom HF repository
python -m src.combine_engineered_datasets --upload --repo-id myuser/my-combined-data
```

### Makefile Targets

```bash
# Combine datasets and save locally
make combine-datasets

# Combine and upload to Hugging Face
make combine-datasets-upload
```

### Python API

```python
from src.combine_engineered_datasets import run_pipeline

# Run the full pipeline
merged_df = run_pipeline(
    upload=True,
    output_file=Path("data/combined.parquet"),
    repo_id="jpearce610/combined_engineered_data"
)
```

## Output

The merged dataset contains:
- All columns from auction, item, and bid datasets
- `auction_id` and `item_id` as first two columns
- Suffix `_bid` for conflicting column names from bid data
- Suffix `_auction` or `_item` for conflicts between auction and item data

### Example Column Order

```
auction_id | item_id | auction_length_hours | item_title_length | bid_amount | bid_number | ...
```

## Environment Variables

The pipeline uses the `HF_TOKEN` environment variable for authentication:

```bash
# In .env file
HF_TOKEN=your_huggingface_token_here
```

## Output Repository

By default, the combined dataset is uploaded to:
- **Repository**: `jpearce610/combined_engineered_data`
- **Visibility**: Public

## Error Handling

The pipeline includes comprehensive error handling:

- **Missing Keys**: Raises `ValueError` if no common merge keys found
- **Missing Token**: Raises `ValueError` if HF_TOKEN not set and upload requested
- **Empty Datasets**: Handles empty datasets gracefully with outer join

## Performance Considerations

- The bid dataset can be very large (millions of rows)
- Dataset loading uses standard HuggingFace `load_dataset()` API
- Merge operations are performed in-memory using pandas
- Final dataset is saved as Parquet for efficient storage

## Testing

Run the test suite to validate merge logic:

```bash
pytest tests/test_combine_engineered_datasets.py -v
```

Tests cover:
- Basic merge functionality
- Column preservation
- Column ordering
- Outer join behavior
- Duplicate key handling
- Empty dataset handling
- Error cases

## Related Documentation

- [Feature Engineering Pipeline](BATCH_FEATURE_ENGINEERING.md)
- [Auction Features](docs/AUCTION_SCRAPER.md)
- [Item Features](docs/ITEM_SCRAPER.md)
- [Bid Features](docs/BID_SCRAPER_SUMMARY.md)
