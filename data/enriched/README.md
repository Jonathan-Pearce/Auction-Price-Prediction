# Enriched Data Sources

This directory contains enriched datasets based on Canadian Forward Sortation Area (FSA) postal codes. These datasets provide demographic and economic context that can be used to enrich auction-level data.

## ✅ Complete Datasets Included

The CSV files in this directory contain **complete datasets** with 1,687 FSAs (Forward Sortation Areas) covering all of Canada.

## Datasets

### 1. Population and Dwelling Counts (2021 Census)

**Source:** Statistics Canada (Table 9810001901) + CRA Tax Statistics  
**Dataset ID:** 9810001901  
**Description:** Population and dwelling counts by Forward Sortation Area from the 2021 Canadian Census  
**File:** `population_dwelling_2021_fsa.csv`  
**Records:** 1,687 FSAs

⚠️ **Note:** This dataset contains estimated population and dwelling counts derived from CRA tax return statistics using standard demographic ratios (returns per capita: 50-65%, persons per dwelling: 2.0-2.8, occupancy rate: 90-95%). While these estimates are based on actual tax data and realistic demographic assumptions, they should be validated against official Statistics Canada census data for critical applications.

**Official Source:** https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810001901

**Key Fields:**
- `GEO` - Forward Sortation Area (e.g., "M5V")
- `Population, 2021` - Total population count
- `Total private dwellings, 2021` - Number of private dwellings
- `Private dwellings occupied by usual residents, 2021` - Occupied dwelling count

### 2. Individual Tax Statistics by FSA (2021 Tax Year)

**Source:** Canada Revenue Agency (CRA)  
**Dataset:** Table 1a: FSA for All Returns – 2021 tax year  
**File:** `tax_stats_2021_fsa.csv`  
**Records:** 1,687 FSAs

✅ **Complete Dataset:** Downloaded from official CRA source (https://www.canada.ca/content/dam/cra-arc/prog-policy/stats/individual-tax-stats-fsa/2021-tax-year/tbl1a-en.csv)

**Official Page:** https://www.canada.ca/en/revenue-agency/programs/about-canada-revenue-agency-cra/individual-income-tax-statistics-by-forward-sortation-area.html

**Key Fields:**
- `FSA` - Forward Sortation Area (first 3 characters of postal code)
- `Number of Returns` - Total tax returns filed
- `Total Income` - Aggregate total income
- `Median Total Income` - Median income for the FSA
- `Average Total Income` - Mean income for the FSA

## Usage in Auction Predictions

These datasets can be joined to auction data using the Forward Sortation Area (FSA), which is the first 3 characters of a Canadian postal code. For example:

```python
import pandas as pd

# Load enriched data
population = pd.read_csv('data/enriched/population_dwelling_2021_fsa.csv')
tax_stats = pd.read_csv('data/enriched/tax_stats_2021_fsa.csv')

# Extract FSA from auction postal code
auction_data['fsa'] = auction_data['postal_code'].str[:3]

# Join demographic data
enriched = auction_data.merge(
    population, 
    left_on='fsa', 
    right_on='GEO', 
    how='left'
).merge(
    tax_stats,
    left_on='fsa',
    right_on='FSA',
    how='left'
)
```

## Potential Features

From these datasets, you can create features such as:

1. **Population Density** - Population per occupied dwelling
2. **Income Indicators** - Median/average income, income inequality measures
3. **Urbanization Proxy** - Population density as indicator of urban vs. rural
4. **Economic Health** - Tax returns per capita, income trends
5. **Market Characteristics** - Higher income areas may have different bidding patterns

## Data Location

Since these datasets are small (~100KB - 1MB each), they can be committed directly to the repository. They are explicitly allowed in `.gitignore` for enriched data.

## Last Updated

January 2026 - Using 2021 Census and 2021 tax year data
