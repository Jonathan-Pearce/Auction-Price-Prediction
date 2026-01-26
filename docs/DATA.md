# Data Documentation

> **Data collection, storage, and processing for MaxSold auction data**

## Overview

This document covers the data pipeline from collection through feature engineering, including schemas, storage strategy, and Hugging Face integration.

---

## Data Sources

### Primary Data: MaxSold API

MaxSold auction data is collected via their public API endpoints:

| Endpoint | Purpose | Data Retrieved |
|----------|---------|----------------|
| Auction Items | List all items in auction | Item IDs, titles, categories, current bids |
| Item Details | Full item with bid history | Description, images, all bids with timestamps |
| Enriched Info | Additional metadata | Category hierarchy, estimates, keywords |

See [references/maxsold-api-reference.md](../references/maxsold-api-reference.md) for complete API documentation.

### Enriched Data: Canadian Demographics & Economics

In addition to MaxSold data, we incorporate **enriched datasets** based on Forward Sortation Areas (FSA - first 3 characters of Canadian postal codes):

| Dataset | Source | Purpose |
|---------|--------|---------|
| Population & Dwelling Counts | Statistics Canada (2021 Census) | Demographic characteristics, urbanization proxy |
| Individual Tax Statistics | Canada Revenue Agency (2021 tax year) | Income levels, economic indicators |

These datasets enable geographic and economic features for auction predictions. See [ENRICHED_DATA.md](ENRICHED_DATA.md) for complete documentation on enriched data sources, schemas, and feature engineering.

---

## Data Schema

### Core Entities

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Auction   │──────<│    Item     │──────<│    Bid      │
└─────────────┘       └─────────────┘       └─────────────┘
                            │
                            │
                      ┌─────┴─────┐
                      │           │
               ┌──────┴───┐ ┌─────┴──────┐
               │  Image   │ │  Enriched  │
               └──────────┘ └────────────┘
```

### Table Schemas

See [references/schema.sql](../references/schema.sql) for complete DuckDB schema definitions.

**Key Tables:**
- `auctions` - Auction events (~10,000 records)
- `items` - Individual lots (~1,000,000 records)
- `bids` - Bidding history (~5,000,000+ records)
- `item_images` - Image URLs and metadata
- `item_enriched` - Additional item data

**Feature Tables:**
- `features_tabular` - Pre-computed tabular features
- `features_text` - Text embeddings
- `features_image` - Image embeddings
- `features_sequential` - Bid sequence features

---

## Data Collection Strategy

### Collection Scope

| Metric | Target |
|--------|--------|
| Auctions | ~10,000 completed auctions |
| Items | ~1,000,000 items |
| Bids | Full history for all items |
| Images | Primary image for each item |
| Time Range | Rolling 2-year history |

### Collection Process

```python
# Pseudocode for scraping workflow
async def collect_auctions():
    # 1. Discover auction IDs (from sitemap/search)
    auction_ids = await discover_auctions()
    
    # 2. For each auction
    for auction_id in auction_ids:
        # Fetch items list
        items = await client.get_auction_items(auction_id)
        
        # For each item, get details + bids
        for item in items:
            details = await client.get_item_details(
                auction_id, item.id
            )
            enriched = await client.get_enriched_info(item.id)
            
            # Store to DuckDB
            db.insert_item(details, enriched)
        
        # Rate limiting
        await asyncio.sleep(RATE_LIMIT_DELAY)
```

### Rate Limiting

| Setting | Value | Notes |
|---------|-------|-------|
| Requests per second | 2 | Conservative default |
| Retry attempts | 3 | With exponential backoff |
| Backoff base | 2 seconds | Doubles each retry |
| Concurrent requests | 5 | Async semaphore limit |

---

## Storage Architecture

### Two-Tier Storage

```
┌────────────────────────────┐
│   Hugging Face Datasets    │  ← Long-term storage, sharing
│   (Parquet format)         │
│   • Raw auction data       │
│   • Immutable snapshots    │
│   • Version controlled     │
└────────────┬───────────────┘
             │ Load/Sync
             ↓
┌────────────────────────────┐
│        DuckDB              │  ← Working database, queries
│   (Local file: data.db)    │
│   • Feature engineering    │
│   • Analytical queries     │
│   • Model serving          │
└────────────────────────────┘
```

### Hugging Face Dataset Structure

```
jonathan-pearce/maxsold-auctions/
├── data/
│   ├── auctions.parquet
│   ├── items.parquet
│   ├── bids.parquet
│   └── enriched.parquet
├── images/
│   └── {item_id}/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
└── README.md (dataset card)
```

### DuckDB Organization

**Single database file**: `data/auction_data.duckdb`

Rationale:
- Simplifies backups and portability
- DuckDB handles large tables efficiently
- Easy to embed in HF Space

---

## Data Quality

### Validation Rules

```python
# Item validation
class ItemValidator:
    def validate(self, item: Item) -> List[str]:
        errors = []
        
        if not item.title:
            errors.append("Missing title")
        
        if item.winning_bid < 0:
            errors.append("Negative winning bid")
        
        if item.bid_count > 0 and not item.bids:
            errors.append("Bid count mismatch")
        
        return errors
```

### Data Completeness Tracking

| Field | Required | Completeness Target |
|-------|----------|---------------------|
| title | Yes | 100% |
| description | No | 95%+ |
| category | No | 90%+ |
| images | No | 98%+ |
| bid_history | Yes (if bids) | 100% |
| enriched_data | No | 85%+ |

### Handling Missing Data

| Field | Strategy |
|-------|----------|
| Description | Empty string, flag in features |
| Category | "Unknown" category |
| Images | Use placeholder embedding |
| Enriched | Skip enriched features |

---

## Feature Engineering

### Tabular Features

```python
# Example tabular features
features = {
    # Item features
    "title_word_count": len(title.split()),
    "description_length": len(description),
    "has_dimensions": bool(dimensions),
    "image_count": len(images),
    
    # Category features
    "category_encoded": category_encoder[category],
    "category_avg_price": historical_avg[category],
    
    # Auction context
    "auction_item_count": auction.total_items,
    "lot_position_normalized": lot_num / total_lots,
    
    # Time features
    "day_of_week": end_time.weekday(),
    "hour_of_day": end_time.hour,
    
    # Historical features
    "seller_avg_price": seller_history.mean(),
    "category_sell_rate": category_sold / category_total,
}
```

### Text Features

```python
# Text embedding pipeline
from transformers import AutoTokenizer, AutoModel

def compute_text_embedding(title: str, description: str) -> np.ndarray:
    combined = f"{title} [SEP] {description}"
    
    inputs = tokenizer(combined, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding
```

### Image Features

```python
# Image embedding pipeline
from torchvision import models, transforms

def compute_image_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    
    # Preprocess
    tensor = transform(image).unsqueeze(0)
    
    # Extract features (before classification head)
    with torch.no_grad():
        embedding = model.features(tensor)
        embedding = model.avgpool(embedding)
        embedding = embedding.flatten().numpy()
    
    return embedding
```

### Sequential Features

```python
# Bid sequence features
def compute_sequence_features(bids: List[Bid]) -> dict:
    if not bids:
        return {"bid_count": 0, ...}
    
    amounts = [b.amount for b in bids]
    times = [b.timestamp for b in bids]
    
    return {
        "bid_count": len(bids),
        "unique_bidders": len(set(b.bidder_id for b in bids)),
        "time_to_first_bid": (times[0] - auction_start).seconds,
        "avg_bid_increment": np.mean(np.diff(amounts)),
        "final_bid_velocity": ...,  # Bids in last hour
        "soft_close_count": sum(1 for b in bids if is_soft_close(b)),
    }
```

---

## Uploading to Hugging Face

### Initial Upload

```python
from datasets import Dataset
import pandas as pd

# Load from DuckDB
df_items = duckdb.query("SELECT * FROM items").df()
df_bids = duckdb.query("SELECT * FROM bids").df()

# Create HF Dataset
dataset = Dataset.from_pandas(df_items)

# Push to Hub
dataset.push_to_hub(
    "jonathan-pearce/maxsold-auctions",
    private=False,
    token=HF_TOKEN
)
```

### Incremental Updates

```python
# Append new data
from datasets import load_dataset

# Load existing
dataset = load_dataset("jonathan-pearce/maxsold-auctions")

# Add new rows
new_data = Dataset.from_pandas(new_df)
updated = concatenate_datasets([dataset["train"], new_data])

# Push update
updated.push_to_hub("jonathan-pearce/maxsold-auctions")
```

### Dataset Card

Create `README.md` for the dataset:

```markdown
---
license: cc-by-4.0
task_categories:
  - tabular-regression
language:
  - en
size_categories:
  - 1M<n<10M
---

# MaxSold Auction Dataset

Auction data from MaxSold.com for price prediction research.

## Dataset Description
- ~10,000 auctions
- ~1,000,000 items with bidding history
- Categories: Furniture, Electronics, Collectibles, etc.

## Features
- Item metadata (title, description, category)
- Bidding history (all bids with timestamps)
- Auction context (location, timing)
- Images (primary item photos)

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("jonathan-pearce/maxsold-auctions")
```
```

---

## Data Privacy

- **Bidder IDs**: Anonymized/hashed, no PII
- **Seller Info**: Only public auction metadata
- **Images**: Publicly available from MaxSold
- **Compliance**: Respect robots.txt and ToS

---

*Last updated: January 2026*
