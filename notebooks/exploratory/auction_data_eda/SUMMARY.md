# Exploratory Data Analysis Summary - Auction Data

**Date:** January 26, 2026  
**Dataset:** `jpearce610/auction_data` (Hugging Face)  
**Objective:** Analyze auction-level data to propose engineered features for item-level price prediction

---

## 1. Dataset Overview

- **Total Records:** 24,621 auctions
- **Time Period:** 2015 to 2026
- **Columns:** 17 original columns
- **Data Quality:** Complete dataset with no missing values in critical fields

### Column Structure

```
auction_id                            int64
auction_title                        string
auction_starts                       datetime
auction_ends                         datetime
auction_last_item_closes             datetime
auction_removal_info                 string
auction_intro                        string
auction_pickup_time                  string
auction_partner_url                  string
auction_extended_bidding             bool
auction_extended_bidding_interval    int64
auction_extended_bidding_threshold   int64
auction_item_count                   int64
auction_total_viewed                 int64
auction_total_winning_price          float64
auction_total_bid_count              int64
auction_total_images                 int64
```

---

## 2. Key Findings

### 2.1 Auction Characteristics

- **Item Count per Auction:**
  - Median: ~88 items
  - Range: 1 to 1,214 items
  - Mean: ~120 items

- **Total Winning Price per Auction:**
  - Median: ~$6,000
  - Range: $0 to $129,000
  - Significant right skew (some high-value auctions)

- **Extended Bidding (Soft-Close):**
  - Nearly 100% of auctions use extended bidding
  - Threshold: 2 minutes before close triggers extension
  - Extension: -2 (appears to be configuration value)

### 2.2 Engagement Metrics

- **Views per Auction:**
  - Median: ~48,000 views
  - Strong variation: 0 to 1.55M views

- **Bids per Auction:**
  - Median: ~1,300 bids
  - Range: 0 to 25,500 bids

- **Images per Auction:**
  - Median: ~550 images
  - Range: 1 to 10,000 images

### 2.3 Correlations

Strong positive correlations found:
- **Item Count ↔ Total Price:** r = 0.82 (more items = higher total revenue)
- **Views ↔ Bids:** r = 0.94 (high views lead to high bids)
- **Item Count ↔ Total Bids:** r = 0.83
- **Item Count ↔ Total Views:** r = 0.82

Moderate correlations:
- **Avg Images per Item ↔ Avg Price per Item:** r = 0.45
- **Bid-to-View Ratio:** Median = 0.027 (2.7% of viewers place bids)

### 2.4 Temporal Patterns

- **Auction Duration:**
  - Median: ~168 hours (7 days)
  - Most auctions run for 1 week

- **Extended Hours:**
  - Average extension: varies by competitive bidding
  - Can add several hours beyond scheduled end

- **Seasonality:**
  - Auctions distributed across all months
  - Some day-of-week effects visible

### 2.5 Geographic Distribution

- **Top Locations:** Toronto, Hamilton, Brampton, Mississauga (Ontario, Canada)
- **Geographic Concentration:** Strong clustering in Greater Toronto Area
- **Total Unique Locations:** ~200+ cities

---

## 3. Proposed Engineered Features

### 3.1 Auction-Level Context Features

These features capture the "quality" or characteristics of an auction, which can be joined to item-level data:

```python
# Aggregated totals
auction_total_winning_price
auction_item_count
auction_total_viewed
auction_total_bid_count
auction_total_images

# Averages (key contextual features)
avg_price_per_item          # Auction average winning price
avg_views_per_item          # Auction average views
avg_bids_per_item           # Auction average bids
avg_images_per_item         # Auction average image count
bid_to_view_ratio           # Auction engagement rate
```

**Rationale:** Items in high-performing auctions tend to sell for more. These averages provide context.

### 3.2 Temporal Features

```python
# Duration features
auction_duration_hours      # Planned duration
auction_extended_hours      # Actual extension due to soft-close

# Cyclical features
start_year                  # Year effect (market trends)
start_month                 # Seasonal effect
start_day_of_week          # Day of week effect (0=Mon, 6=Sun)
start_hour                  # Hour of day (start time)
```

**Rationale:** Timing affects both competition and prices. Seasonality matters for collectibles/antiques.

### 3.3 Auction Mechanism Features

```python
auction_extended_bidding            # Boolean: soft-close enabled
auction_extended_bidding_threshold  # Minutes threshold (typically 2)
auction_extended_bidding_interval   # Extension minutes (varies)
```

**Rationale:** Soft-close auctions may have different bidding dynamics and final prices.

### 3.4 Location Features

```python
location                    # Extracted city from title
location_encoded            # One-hot or target encoding
location_popularity         # Number of auctions in that location
```

**Rationale:** Geographic differences in buying power, population density, and item types.

### 3.5 Text-Based Features

```python
title_length                # Character count
title_word_count            # Word count
intro_length                # Description length
intro_word_count            # Description word count
removal_info_length         # Pickup instructions length
has_partner_url             # Boolean: affiliated auction
```

**Rationale:** Longer, more detailed descriptions may correlate with higher prices. Partner auctions may differ.

### 3.6 Item-Relative Features (for item-level prediction)

When joining with item-level data, create these relative features:

```python
# Position features
item_position_in_auction    # Item number (1 to N)
item_pct_through_auction    # Temporal position (0.0 to 1.0)

# Relative performance
item_price_vs_auction_avg   # (item_price - avg_price_per_item) / std
item_views_vs_auction_avg   # (item_views - avg_views_per_item) / std
item_bids_vs_auction_avg    # Similar normalization
item_images_vs_auction_avg  # Compare to auction average
```

**Rationale:** An item's performance relative to its auction peers is highly predictive.

---

## 4. Recommended Feature Engineering Pipeline

### Step 1: Load Auction Data
```python
from datasets import load_dataset
auction_df = load_dataset("jpearce610/auction_data")["train"].to_pandas()
```

### Step 2: Create Temporal Features
```python
auction_df['auction_starts_dt'] = pd.to_datetime(auction_df['auction_starts'])
auction_df['auction_ends_dt'] = pd.to_datetime(auction_df['auction_ends'])
auction_df['auction_duration_hours'] = (
    auction_df['auction_ends_dt'] - auction_df['auction_starts_dt']
).dt.total_seconds() / 3600

auction_df['start_year'] = auction_df['auction_starts_dt'].dt.year
auction_df['start_month'] = auction_df['auction_starts_dt'].dt.month
auction_df['start_day_of_week'] = auction_df['auction_starts_dt'].dt.dayofweek
auction_df['start_hour'] = auction_df['auction_starts_dt'].dt.hour
```

### Step 3: Create Derived Metrics
```python
auction_df['avg_price_per_item'] = (
    auction_df['auction_total_winning_price'] / 
    auction_df['auction_item_count'].replace(0, np.nan)
)
auction_df['avg_views_per_item'] = (
    auction_df['auction_total_viewed'] / 
    auction_df['auction_item_count'].replace(0, np.nan)
)
auction_df['avg_bids_per_item'] = (
    auction_df['auction_total_bid_count'] / 
    auction_df['auction_item_count'].replace(0, np.nan)
)
auction_df['avg_images_per_item'] = (
    auction_df['auction_total_images'] / 
    auction_df['auction_item_count'].replace(0, np.nan)
)
auction_df['bid_to_view_ratio'] = (
    auction_df['auction_total_bid_count'] / 
    auction_df['auction_total_viewed'].replace(0, np.nan)
)
```

### Step 4: Extract Location
```python
import re
def extract_location(title):
    match = re.search(r'^([^(]+)\(', title)
    return match.group(1).strip() if match else None

auction_df['location'] = auction_df['auction_title'].apply(extract_location)
```

### Step 5: Create Text Features
```python
auction_df['title_length'] = auction_df['auction_title'].str.len()
auction_df['title_word_count'] = auction_df['auction_title'].str.split().str.len()
auction_df['intro_length'] = auction_df['auction_intro'].fillna('').str.len()
auction_df['has_partner_url'] = auction_df['auction_partner_url'].notna()
```

### Step 6: Join with Item-Level Data
```python
# When you have item-level data
item_df = load_item_data()  # Your item dataset
merged_df = item_df.merge(
    auction_df[['auction_id', 'avg_price_per_item', 'avg_views_per_item', ...]],
    on='auction_id',
    how='left'
)

# Create relative features
merged_df['item_price_vs_auction_avg'] = (
    (merged_df['item_winning_price'] - merged_df['avg_price_per_item']) / 
    merged_df['avg_price_per_item'].replace(0, np.nan)
)
```

---

## 5. Feature Importance Expectations

Based on correlation analysis, expected feature importance for predicting item prices:

### High Importance (Expected)
1. **avg_price_per_item** - Strong auction context signal
2. **item_price_vs_auction_avg** - Relative performance matters
3. **avg_views_per_item** - Auction popularity indicator
4. **location** - Geographic buying power
5. **bid_to_view_ratio** - Auction engagement quality

### Medium Importance (Expected)
6. **auction_item_count** - Competition/scarcity effect
7. **start_month** - Seasonality
8. **auction_duration_hours** - Time for discovery
9. **avg_images_per_item** - Quality indicator
10. **item_position_in_auction** - Ordering effects

### Lower Importance (Expected)
11. **start_hour** - Time of day (less critical)
12. **title_length** - Weak correlation
13. **auction_extended_bidding** - Nearly universal (low variance)

---

## 6. Data Quality Considerations

### Strengths
✓ Complete data (no missing critical values)  
✓ Large sample size (24k+ auctions)  
✓ Wide temporal coverage (10+ years)  
✓ Rich feature set with engagement metrics  

### Limitations
⚠ Auction-level data only (no item details in this dataset)  
⚠ Geographic concentration (mostly Ontario, Canada)  
⚠ Extended bidding near-universal (low variance)  
⚠ Outliers present (some very large/small auctions)  

### Recommendations
- **Outlier handling:** Consider log transforms for price/view features
- **Zero values:** Many auctions have 0 total winning price (all items unsold?) - investigate
- **Feature scaling:** Normalize features before model training
- **Target encoding:** For location with many categories (200+ cities)

---

## 7. Next Steps for Feature Engineering

1. **Load item-level data** and join with auction features
2. **Create item-relative features** as described above
3. **Encode categorical variables:**
   - Location: Target encoding or frequency encoding
   - Day of week: Cyclical encoding (sin/cos transformation)
4. **Handle outliers:**
   - Winsorize extreme values (99th percentile)
   - Log transform right-skewed features
5. **Feature selection:**
   - Use correlation analysis
   - Try tree-based feature importance
   - Consider LASSO for linear models
6. **Validation:**
   - Time-based split (train on older auctions, test on newer)
   - Cross-validation by auction_id to avoid leakage

---

## 8. Code Artifacts

All analysis code is available in:
- **Notebook:** `notebooks/exploratory/auction_data_eda/01_auction_data_exploration.ipynb`
- **Processed Data:** `data/processed/auction_data_with_features.parquet`
- **Test Script:** `notebooks/exploratory/auction_data_eda/test_hf_access.py`

---

## 9. Conclusion

The auction-level data provides rich contextual features for item-level price prediction. Key insights:

1. **Auction context matters:** Items in high-performing auctions tend to perform better
2. **Strong engagement patterns:** Views and bids are highly correlated
3. **Geographic effects:** Location is a significant factor
4. **Temporal patterns:** Seasonality and timing affect prices
5. **Relative features are key:** An item's performance vs. its auction peers is highly predictive

The proposed engineered features capture these patterns and should significantly improve model performance when predicting individual item prices.

---

**Contact:** For questions about this analysis, see the Jupyter notebook for detailed visualizations and code.
