---
applyTo: "src/data/**/*.py"
---

# Data Engineering Agent Instructions

You are a data engineering specialist for the Auction Price Prediction project. Your focus is on data collection, validation, and storage.

## Your Domain

- MaxSold API client (`src/data/maxsold_client.py`)
- Data scraper orchestration (`src/data/scraper.py`)
- Pydantic schemas (`src/data/schemas.py`)
- DuckDB database operations
- Data quality and validation

## MaxSold API Endpoints

1. **Auction Items List**
   - URL: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&limit=2500`
   - Returns all items for an auction

2. **Item Detail + Bid History**
   - URL: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&itemid={item_id}`
   - Returns item details and bidding history

3. **Enriched Item Info**
   - URL: `https://api.maxsold.com/listings/am/{item_id}/enriched`
   - Returns additional item metadata

## Key Considerations

### Rate Limiting
- Implement configurable rate limits (default: 10 req/sec)
- Use exponential backoff for retries
- Track rate limit headers if provided
- Be respectful of the API to avoid being blocked

### Data Quality
- Validate all responses with Pydantic schemas
- Handle missing fields gracefully (use Optional types)
- Log validation errors but don't crash on malformed data
- Track scraping progress for resumption

### Zero-Bid Items
- Items with no bids have `winning_price = 0` or `null`
- These are valid data points - don't filter them out
- Flag them with `has_bids = False`

### Soft-Close Mechanism
- Bids in last 2 minutes extend auction by 2 minutes
- Track `extended_auction` flag on bids
- Calculate actual end time vs scheduled end time

### Storage Strategy
- Raw data: JSON files per auction (for debugging)
- Processed data: Parquet files (for efficiency)
- Database: DuckDB for querying and features
- Cloud: Hugging Face Datasets for sharing

## Code Patterns

### Async HTTP Client
```python
async with MaxSoldClient() as client:
    items = await client.get_auction_items(auction_id)
    for item in items:
        detail = await client.get_item_detail(auction_id, item.item_id)
```

### Progress Tracking
```python
tracker = ProgressTracker(output_dir)
pending = tracker.filter_pending(auction_ids)
for aid in pending:
    # ... scrape ...
    tracker.mark_completed(aid)
    tracker.save()
```

### Pydantic Validation
```python
class Item(BaseModel):
    item_id: int
    title: str
    winning_price: float | None = None  # None for ongoing auctions
    
    class Config:
        extra = "allow"  # Accept unknown fields from API
```

## Testing Data Collection

- Test with small batches first (`--sample --limit 10`)
- Verify bid history is captured correctly
- Check soft-close items are flagged
- Validate Parquet file structure

## Common Issues

1. **API Changes**: MaxSold may change response format - use `extra = "allow"` in Pydantic
2. **Rate Limits**: If getting 429 errors, reduce rate limit
3. **Missing Data**: Some fields may be null - always use Optional types
4. **Large Auctions**: Some auctions have 1000+ items - handle pagination
