# Enriched Data Documentation

> **Demographic and economic data enrichment for auction price prediction**

## Overview

This document describes the enriched datasets used to provide demographic and economic context for MaxSold auction predictions. These datasets are based on Canadian Forward Sortation Areas (FSA) - the first 3 characters of postal codes - and enable features that capture regional economic characteristics.

---

## What are Forward Sortation Areas (FSA)?

In Canada, postal codes consist of 6 characters in the format `A1A 1A1`. The Forward Sortation Area (FSA) is the first 3 characters (e.g., `M5V`, `K1A`), which represents a geographic region. FSAs typically cover:

- **Urban areas**: A few neighborhoods or city blocks
- **Rural areas**: Larger geographic regions

FSAs enable geographic analysis without requiring exact addresses, providing appropriate granularity for regional economic and demographic trends.

---

## Enriched Datasets

### 1. Population and Dwelling Counts (Statistics Canada, 2021 Census)

**File:** `data/enriched/population_dwelling_2021_fsa.csv`  
**Source:** Statistics Canada Table 9810001901  
**Reference:** 2021 Census of Population

**Description:**  
Population and private dwelling counts by Forward Sortation Area from the 2021 Canadian Census.

**Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `GEO` | string | Forward Sortation Area (e.g., "M5V") |
| `Population, 2021` | integer | Total population count |
| `Total private dwellings, 2021` | integer | Total number of private dwellings |
| `Private dwellings occupied by usual residents, 2021` | integer | Number of occupied private dwellings |

**Download:**
- Official source: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810001901
- Dataset ID: 9810001901-eng.csv

**Coverage:**  
All Canadian FSAs (~1,600 FSAs)

---

### 2. Individual Tax Statistics by FSA (CRA, 2021 Tax Year)

**File:** `data/enriched/tax_stats_2021_fsa.csv`  
**Source:** Canada Revenue Agency (CRA)  
**Reference:** Table 1a - FSA for All Returns, 2021 tax year

**Description:**  
Aggregate income statistics from individual tax returns, grouped by Forward Sortation Area.

**Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `FSA` | string | Forward Sortation Area (first 3 chars of postal code) |
| `Number of Returns` | integer | Total tax returns filed in this FSA |
| `Total Income` | integer | Aggregate total income (in dollars) |
| `Median Total Income` | integer | Median income for this FSA (in dollars) |
| `Average Total Income` | integer | Mean income for this FSA (in dollars) |

**Download:**
- Official page: https://www.canada.ca/en/revenue-agency/programs/about-canada-revenue-agency-cra/individual-income-tax-statistics-by-forward-sortation-area.html
- Direct CSV: https://www.canada.ca/content/dam/cra-arc/prog-policy/stats/individual-tax-stats-fsa/2021-tax-year/tbl1a-en.csv

**Coverage:**  
All Canadian FSAs with sufficient tax returns to maintain privacy (~1,600 FSAs)

---

## Integration with Auction Data

### Joining on FSA

Auction data from MaxSold includes postal codes in the `AuctionLocation` schema:

```python
class AuctionLocation(BaseModel):
    city: str | None
    state: str | None  # Province
    country: str | None
    postal_code: str | None  # e.g., "M5V 3A8"
```

Extract the FSA (first 3 characters) and join with enriched data:

```python
import pandas as pd

# Load enriched datasets
population = pd.read_csv('data/enriched/population_dwelling_2021_fsa.csv')
tax_stats = pd.read_csv('data/enriched/tax_stats_2021_fsa.csv')

# Extract FSA from postal code
auction_data['fsa'] = auction_data['postal_code'].str[:3].str.upper()

# Join population data
auction_enriched = auction_data.merge(
    population,
    left_on='fsa',
    right_on='GEO',
    how='left'
)

# Join tax statistics
auction_enriched = auction_enriched.merge(
    tax_stats,
    left_on='fsa',
    right_on='FSA',
    how='left'
)
```

---

## Feature Engineering

### Demographic Features

From the Census data, you can derive:

1. **Persons per Dwelling** (urbanization proxy)
   ```python
   features['persons_per_dwelling'] = (
       df['Population, 2021'] / 
       df['Private dwellings occupied by usual residents, 2021']
   )
   ```
   Interpretation: Higher values suggest multi-unit buildings (urban), lower suggests single-family homes (suburban/rural)

2. **Occupancy Rate**
   ```python
   features['occupancy_rate'] = (
       df['Private dwellings occupied by usual residents, 2021'] /
       df['Total private dwellings, 2021']
   )
   ```
   Interpretation: High occupancy suggests desirable neighborhoods

### Economic Features

From the Tax Statistics, you can derive:

1. **Income Level**
   ```python
   features['median_income'] = df['Median Total Income']
   features['average_income'] = df['Average Total Income']
   ```
   Interpretation: Proxy for purchasing power and auction bidding behavior

2. **Income Inequality**
   ```python
   features['income_skew'] = (
       df['Average Total Income'] - df['Median Total Income']
   ) / df['Median Total Income']
   ```
   Interpretation: Positive skew indicates income inequality

3. **Tax Returns per Capita**
   ```python
   features['returns_per_capita'] = (
       df['Number of Returns'] / df['Population, 2021']
   )
   ```
   Interpretation: Participation in formal economy (working-age population proxy)

4. **Income Bins** (for categorical encoding)
   ```python
   features['income_tier'] = pd.cut(
       df['Median Total Income'],
       bins=[0, 30000, 50000, 75000, 100000, float('inf')],
       labels=['low', 'lower_mid', 'mid', 'upper_mid', 'high']
   )
   ```

### Combined Features

1. **Economic Capacity Index**
   ```python
   features['economic_capacity'] = (
       df['Median Total Income'] * df['Population, 2021']
   ) / 1e9  # Scale to billions
   ```
   Interpretation: Total purchasing power in the region

2. **Urbanization Index**
   ```python
   features['urbanization'] = (
       df['persons_per_dwelling'] * np.log1p(df['Population, 2021'])
   )
   ```
   Interpretation: Combined measure of urban character

---

## Missing Data Handling

Not all auction postal codes will have corresponding FSA data:

1. **Invalid/Missing Postal Codes**: Some auctions may not have postal codes
2. **Non-Canadian Auctions**: FSA data is Canada-only
3. **New FSAs**: Newly created FSAs may not be in 2021 census data

**Recommended Handling:**

```python
# Flag missing enrichment
auction_data['has_enriched_data'] = ~auction_data['fsa'].isna()

# Fill numeric features with defaults
numeric_features = ['median_income', 'population', 'persons_per_dwelling']
auction_data[numeric_features] = auction_data[numeric_features].fillna(
    auction_data[numeric_features].median()
)

# Or use a separate "missing" category
auction_data['income_tier'] = auction_data['income_tier'].fillna('unknown')
```

---

## Example Use Case: Auction Price Prediction

### Hypothesis

Auctions in wealthier neighborhoods (higher median income) may have:
- Higher winning prices for luxury items
- More competitive bidding (more bidders per item)
- Different item category preferences

### Implementation

```python
from src.features import add_enriched_features

# Load auction data
auctions = load_auctions_from_db()

# Add enriched features
auctions_enriched = add_enriched_features(auctions)

# Use in model training
features = [
    'item_category',
    'starting_bid',
    'num_images',
    'median_income',        # From enriched data
    'population',           # From enriched data
    'persons_per_dwelling', # Derived from enriched data
]

X = auctions_enriched[features]
y = auctions_enriched['winning_price']

model.fit(X, y)
```

---

## Data Updates

### Current Version
- **Census data**: 2021 Census of Population (released 2022)
- **Tax data**: 2021 tax year (released 2023)

### Update Schedule
- **Census**: Every 5 years (next: 2026 census, released ~2027)
- **Tax Statistics**: Annually (typically released 2 years after tax year)

### How to Update

1. Download new files from official sources
2. Replace CSV files in `data/enriched/`
3. Verify schema compatibility
4. Update this documentation with new dates
5. Re-run feature engineering pipeline
6. Retrain models with new features

---

## Data Quality Notes

### Census Data
- **Completeness**: 100% coverage of Canadian FSAs
- **Accuracy**: Official government statistics
- **Granularity**: FSA level (sufficient for auction analysis)

### Tax Statistics
- **Privacy Protection**: FSAs with <10 returns are suppressed
- **Completeness**: ~95% of FSAs (urban areas nearly complete)
- **Lag**: 2-year delay (using 2021 for 2024 auctions)

### Limitations

1. **Temporal Mismatch**: Using 2021 data for 2024+ auctions
   - Acceptable for stable demographic trends
   - Income statistics change slowly over time
   
2. **Geographic Aggregation**: FSA covers multiple neighborhoods
   - Trade-off between granularity and privacy
   - Sufficient for regional trends
   
3. **Non-Canadian Auctions**: No enrichment available
   - Small percentage of MaxSold auctions
   - Can use model without enriched features

---

## Storage and Version Control

### In Repository
These datasets are **committed to the repository** because:
- Small size (~100KB - 1MB each)
- Static reference data (updated infrequently)
- Enables reproducibility without external dependencies

### `.gitignore` Exception
The `.gitignore` file explicitly allows these files:
```
!data/enriched/*.csv
!data/enriched/*.md
```

---

## References

### Official Sources

1. **Statistics Canada - 2021 Census**
   - URL: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810001901
   - Table ID: 9810001901
   - Release Date: February 2022

2. **CRA - Individual Tax Statistics by FSA**
   - URL: https://www.canada.ca/en/revenue-agency/programs/about-canada-revenue-agency-cra/individual-income-tax-statistics-by-forward-sortation-area.html
   - Edition: 2023 (2021 tax year)
   - Release Date: January 2023

### Related Documentation

- [DATA.md](DATA.md) - Main data documentation
- [DESIGN.md](DESIGN.md) - System architecture
- [data/enriched/README.md](../data/enriched/README.md) - Quick reference for enriched data

---

*Last updated: January 2026*
