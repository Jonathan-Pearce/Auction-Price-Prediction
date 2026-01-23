# MaxSold API Reference

Documentation for the MaxSold public API endpoints used for auction data collection.

## Base URLs

- **Primary API**: `https://maxsold.maxsold.com/msapi/`
- **Listings API**: `https://api.maxsold.com/`

## Endpoints

### 1. Auction Items List

Retrieves all items for a specific auction.

**URL**: `GET /auctions/items`

**Full URL**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&limit={limit}`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `auctionid` | int | Yes | Unique auction identifier |
| `limit` | int | No | Maximum items to return (default: 2500) |

**Example Request**:
```
https://maxsold.maxsold.com/msapi/auctions/items?auctionid=99941&limit=2500
```

**Example Response**:
```json
{
  "items": [
    {
      "item_id": 7433850,
      "title": "Vintage Oak Dresser",
      "description": "...",
      "current_bid": 45.00,
      "bid_count": 5,
      "image_urls": ["..."],
      "category": "Furniture",
      "lot_number": 1
    }
  ],
  "auction_info": {
    "auction_id": 99941,
    "title": "Estate Sale - Toronto",
    "end_time": "2026-01-25T18:00:00Z",
    "status": "active"
  },
  "total_items": 150
}
```

---

### 2. Item Details with Bidding History

Retrieves detailed item information including full bidding history.

**URL**: `GET /auctions/items`

**Full URL**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&itemid={item_id}`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `auctionid` | int | Yes | Unique auction identifier |
| `itemid` | int | Yes | Unique item identifier |

**Example Request**:
```
https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293&itemid=7433850
```

**Example Response**:
```json
{
  "item": {
    "item_id": 7433850,
    "title": "Vintage Oak Dresser",
    "description": "Beautiful vintage oak dresser with brass handles...",
    "starting_bid": 1.00,
    "current_bid": 85.00,
    "winning_bid": null,
    "bid_count": 12,
    "image_urls": [
      "https://images.maxsold.com/...",
      "https://images.maxsold.com/..."
    ],
    "category": "Furniture",
    "subcategory": "Bedroom",
    "condition": "Good",
    "dimensions": "36\" x 18\" x 32\"",
    "lot_number": 1,
    "pickup_location": "Toronto, ON"
  },
  "bids": [
    {
      "bid_id": 123456,
      "amount": 85.00,
      "timestamp": "2026-01-25T17:58:30Z",
      "bidder_id": "user_abc123"
    },
    {
      "bid_id": 123455,
      "amount": 75.00,
      "timestamp": "2026-01-25T17:55:00Z",
      "bidder_id": "user_def456"
    }
  ],
  "auction_info": {
    "auction_id": 103293,
    "end_time": "2026-01-25T18:00:00Z"
  }
}
```

---

### 3. Enriched Item Information

Retrieves additional enriched metadata for an item.

**URL**: `GET /listings/am/{item_id}/enriched`

**Full URL**: `https://api.maxsold.com/listings/am/{item_id}/enriched`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `item_id` | int | Yes (path) | Unique item identifier |

**Example Request**:
```
https://api.maxsold.com/listings/am/7433915/enriched
```

**Example Response**:
```json
{
  "item_id": 7433915,
  "enriched_data": {
    "category_hierarchy": ["Home", "Furniture", "Bedroom", "Dressers"],
    "estimated_value": {
      "low": 50.00,
      "high": 150.00
    },
    "similar_items_sold": [
      {
        "item_id": 7400123,
        "winning_price": 95.00,
        "similarity_score": 0.87
      }
    ],
    "keywords": ["vintage", "oak", "dresser", "brass", "handles"],
    "condition_score": 7.5,
    "popularity_score": 0.65
  }
}
```

---

## Auction Mechanisms

### Soft Close Rule

MaxSold uses a **soft close** mechanism to prevent bid sniping:

- **Rule**: If a bid is placed within the **last 2 minutes** of the auction end time, the auction is extended by **2 minutes**
- **Repeatable**: This extension can happen multiple times if bidding continues
- **Impact on data**: The `end_time` field may differ from the original scheduled end time

**Example**:
```
Original end time: 18:00:00
Bid at 17:59:30 → New end time: 18:01:30
Bid at 18:01:00 → New end time: 18:03:00
No more bids → Auction ends at 18:03:00
```

### Zero-Bid Items

Some items receive **no bids** during the auction:

- **Winning price**: $0.00
- **Status**: Usually marked as "unsold" or "no bids"
- **Handling**: These items should be included in training data as valid outcomes
- **Use case**: Important for predicting which items may not sell

---

## Rate Limiting

> ⚠️ **TODO**: Document observed rate limits

**Recommendations**:
- Implement exponential backoff on 429 errors
- Add configurable delay between requests (default: 0.5s)
- Respect `Retry-After` headers if present
- Consider time-of-day for scraping (off-peak hours)

---

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Auction/Item doesn't exist |
| 429 | Too Many Requests - Rate limited |
| 500 | Server Error |

---

## Data Freshness

- **Active auctions**: Data updates in real-time with bids
- **Completed auctions**: Final data available after auction closes
- **Historical data**: Remains accessible (retention period TBD)

---

## Notes

- All timestamps are in **UTC** (ISO 8601 format)
- Prices are in **CAD** (Canadian Dollars)
- Image URLs are CDN-hosted and publicly accessible
- Bidder IDs are anonymized for privacy

---

*Last updated: January 2026*
*This documentation is based on observed API behavior and may be incomplete.*
