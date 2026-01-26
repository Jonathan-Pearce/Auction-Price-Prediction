# Exploratory Data Analysis Summary

**Generated:** 2026-01-26 05:53:00

## Overview

This exploratory analysis examines the MaxSold auction dataset to support feature engineering for predicting item prices. The analysis covers auction-level data, item-level data, and provides comprehensive feature engineering recommendations.

## Dataset Summary

### Local Auction Location Data
- **Records:** 28816
- **Purpose:** Geographic information for auctions
- **Key Fields:** auction ID, latitude, longitude, postal code, distance from Toronto

### Auction-Level Data (Expected Structure)
- **Auctions:** 1000 auctions analyzed
- **Average Items per Auction:** 254.4
- **Average Total Revenue:** $25807.88
- **Average Duration:** 35.9 hours

### Item-Level Data (Expected Structure)
- **Items:** 5000 items analyzed
- **Categories:** 10 distinct categories
- **Average Winning Price:** $75.02
- **Zero-Bid Rate:** 1.8%

## Key Findings

### 1. Target Variable (Item Winning Price)

**Distribution Characteristics:**
- Highly right-skewed distribution
- Significant proportion of zero-bid items (1.8%)
- Median price: $58.93
- Mean price: $75.02

**Recommendation:** Use log transformation or two-stage modeling approach (classification + regression)

### 2. Category Analysis

**Price Variation by Category:**
- Categories show significant price differences
- Some categories (e.g., Jewelry, Electronics) have higher average prices
- Category-specific features are important predictors

### 3. Bidding Activity

**Strong Predictors:**
- `bid_count`: Number of bids is likely the strongest predictor
- `viewed`: View count indicates interest level
- `unique_bidders`: Competition drives prices

**Zero-Bid Items:**
- Items with no bids need special handling
- Consider separate classification model for "will sell" vs "won't sell"

### 4. Temporal Patterns

**Timing Matters:**
- Day of week and hour of day affect prices
- Weekend vs weekday auctions may differ
- Closing time is important (evening closings may have more bidders)

### 5. Image and Text Features

**Presentation Quality:**
- Number of images correlates with engagement
- Professional photos likely increase prices
- Detailed descriptions attract more bidders

## Feature Engineering Priorities

### High Priority Features (Implement First)
1. **bid_count** - Number of bids (strongest signal)
2. **starting_bid** - Baseline price
3. **category** - Item type (with historical pricing)
4. **condition** - Item quality
5. **viewed** - Interest metric
6. **image_count** - Presentation quality

### Medium Priority Features
- Temporal features (day of week, hour)
- Auction context (total items, auction type)
- Text features (word counts, keywords)
- Historical category statistics

### Advanced Features (Multi-modal Models)
- Text embeddings (BERT/DistilBERT)
- Image embeddings (ResNet/EfficientNet)
- Bid sequence patterns (LSTM/GRU)

## Data Quality Considerations

1. **Missing Values:** Plan imputation strategy for each feature type
2. **Outliers:** Consider capping extreme prices at 99th percentile
3. **Imbalanced Data:** Many zero-bid items require balanced sampling
4. **Temporal Validation:** Use time-based train/test split

## Recommended Modeling Approach

### Option 1: Two-Stage Model
1. **Stage 1:** Binary classifier for has_bids (will item sell?)
2. **Stage 2:** Regression for price (given item will sell)

### Option 2: Zero-Inflated Model
- Use zero-inflated regression models (e.g., ZIP, ZINB)
- Handles mixture of zero and continuous outcomes

### Option 3: Ensemble of Specialists
- **Tabular Model:** XGBoost on engineered features
- **Text Model:** Transformer on title + description
- **Image Model:** CNN on item photos
- **Sequential Model:** LSTM on bid history
- **Fusion Model:** Meta-learner combining all models

## Next Steps

1. **Implement Core Features:**
   - Start with high-priority tabular features
   - Create feature engineering pipeline in `src/features.py`
   - Add feature computation functions to schema

2. **Collect Historical Data:**
   - Calculate category-level statistics
   - Build seller performance metrics
   - Create similar-item lookup

3. **Build Baseline Model:**
   - Train simple XGBoost model on tabular features
   - Establish baseline performance metrics
   - Iterate with additional features

4. **Expand to Multi-Modal:**
   - Add text embeddings
   - Add image embeddings
   - Train fusion model

## Visualizations

The following visualizations have been generated in `exploratory/figures/`:

- `target_price_distribution.png` - Distribution of winning prices
- `price_by_category.png` - Price variation across categories
- `correlation_matrix.png` - Feature correlations
- `dist_*.png` - Individual feature distributions

## References

- Schema: `references/schema.sql`
- Scraper Config: `src/data/scraper_config.yaml`
- API Docs: `references/maxsold-api-reference.md`
