# Exploratory Data Analysis - Auction Data Transformations

This folder contains the exploratory data analysis for the auction dataset from Hugging Face.

## Contents

### 📊 Main Analysis
- **`01_auction_data_exploration.ipynb`** - Comprehensive Jupyter notebook with visualizations and detailed analysis
- **`quick_eda.py`** - Python script for rapid analysis and feature engineering
- **`SUMMARY.md`** - Executive summary with key findings and feature engineering recommendations

### 🧪 Utilities
- **`test_hf_access.py`** - Script to test Hugging Face dataset access

## Quick Start

### Option 1: Run the Quick Analysis Script

```bash
cd /home/runner/work/Auction-Price-Prediction/Auction-Price-Prediction
python notebooks/exploratory/auction_data_eda/quick_eda.py
```

This will:
- Load the auction data from Hugging Face
- Create engineered features
- Print summary statistics
- Save processed data to `data/processed/auction_data_with_features.parquet`

### Option 2: Open the Jupyter Notebook

```bash
jupyter notebook notebooks/exploratory/auction_data_eda/01_auction_data_exploration.ipynb
```

This provides:
- Interactive visualizations
- Detailed correlation analysis
- Step-by-step feature engineering
- Distribution plots and outlier detection

## Key Findings

### Dataset Overview
- **24,621 auctions** spanning 2015-2026
- **17 original columns** with complete data (no missing values)
- **Geographic focus:** Primarily Ontario, Canada (Toronto region)

### Important Metrics
- **Median items per auction:** 103
- **Median total price:** $2,442
- **Median views:** 12,826
- **Median bids:** 1,109
- **Bid-to-view ratio:** ~9.3%

### Top Correlations with Price
1. Total bid count (r=0.84)
2. Total views (r=0.54)
3. Item count (r=0.42)

## Engineered Features

The analysis creates **26 engineered features** including:

### Auction Context Features
- `avg_price_per_item` - Average winning price per item in auction
- `avg_views_per_item` - Average views per item
- `avg_bids_per_item` - Average bids per item
- `avg_images_per_item` - Average images per item
- `bid_to_view_ratio` - Engagement rate

### Temporal Features
- `auction_duration_hours` - Duration of auction
- `auction_extended_hours` - Extension due to soft-close
- `start_year`, `start_month`, `start_day_of_week`, `start_hour`

### Location Features
- `location` - Extracted city from title

### Text Features
- `title_length`, `title_word_count` - Title characteristics
- `intro_length` - Description length
- `has_partner_url` - Affiliation indicator

## Data Files

### Input
- **Source:** `jpearce610/auction_data` (Hugging Face)
- **Local cache:** `data/external/auction_data.parquet`

### Output
- **Processed data:** `data/processed/auction_data_with_features.parquet`
  - Contains all original columns plus 18 engineered features
  - Ready for joining with item-level data

## Next Steps

1. **Join with item-level data** to create a complete dataset
2. **Create item-relative features** (e.g., `item_price_vs_auction_avg`)
3. **Build predictive models** using the engineered features
4. **Evaluate feature importance** to refine the feature set

## Feature Engineering Recommendations

See **`SUMMARY.md`** for detailed recommendations including:
- Recommended feature engineering pipeline
- Expected feature importance
- Data quality considerations
- Handling strategies for outliers and categorical variables

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=14.0.0
datasets>=2.15.0
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn pyarrow datasets
```

## Notes

- All analysis is **read-only** - does not modify source code
- Exploratory work is **isolated** in this folder
- Processed data is saved for downstream use
- No impact on existing codebase

---

**Last Updated:** January 26, 2026  
**Dataset Version:** jpearce610/auction_data (latest)  
**Analysis Focus:** Feature engineering for item price prediction
