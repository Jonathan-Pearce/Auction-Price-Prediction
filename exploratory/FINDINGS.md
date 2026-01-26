# Key Findings from Exploratory Data Analysis

## Executive Summary

This exploratory data analysis examined the MaxSold auction dataset to identify patterns and propose feature engineering strategies for predicting item prices. The analysis reveals strong predictive signals from bidding activity, item characteristics, and temporal patterns, with recommendations for a multi-modal modeling approach.

---

## Critical Insights

### 1. Target Variable Characteristics

**Item Winning Price Distribution:**
- **Right-skewed**: Mean ($73.67) > Median ($58.25)
- **Zero-bid rate**: 1.8% of items receive no bids
- **Price range**: $0 - $460 (in synthetic data; real data likely higher)
- **Concentration**: 80% of items sell for under $100

**Implication**: Log transformation or two-stage modeling (classification → regression) recommended.

### 2. Strongest Predictors Identified

| Feature | Correlation | Type | Priority |
|---------|-------------|------|----------|
| `starting_bid` | 0.79 | Baseline | **HIGH** |
| `bid_count` | 0.37 | Activity | **HIGH** |
| `category` | Varies | Categorical | **HIGH** |
| `condition` | Varies | Categorical | **HIGH** |
| `viewed` | ~0.00 | Engagement | MEDIUM |

**Key Finding**: Starting bid is the strongest single predictor, but bid activity provides the most valuable signal about final price.

### 3. Category-Specific Patterns

**Price Variation by Category** (from synthetic data):
- Electronics: $78.88 average (highest)
- Art: $77.28 average
- Collectibles: $70.34 average (lowest)
- Standard deviation: $50-66 across categories

**Implication**: Category-specific features (historical averages, sell rates) are critical.

### 4. Zero-Bid Problem

**Challenge**: 1.8% of items receive zero bids
- Cannot train single regression model on all items
- Need to predict both "will sell" and "at what price"

**Solutions**:
1. **Two-stage model** (recommended):
   - Stage 1: Binary classifier (has_bids: yes/no)
   - Stage 2: Regressor (price | has_bids=True)
2. **Zero-inflated regression**: Single model handling both aspects
3. **Ensemble with weighting**: Combine models with confidence scores

---

## Feature Engineering Strategy

### Phase 1: Core Tabular Features (Weeks 1-2)

**Implement immediately** in `src/features.py`:

```python
# Item characteristics
- title_word_count
- description_length
- image_count
- condition_encoded (ordinal: Poor=0, Fair=1, Good=2, Excellent=3)
- category_encoded (target encoding with smoothing)

# Pricing baseline
- starting_bid
- starting_bid_log (log transformation)

# Bidding activity (if available)
- bid_count
- unique_bidders (if bid history available)

# Temporal features
- auction_day_of_week (0=Monday, 6=Sunday)
- auction_start_hour (0-23)
- is_weekend (boolean)
- closing_hour (0-23)

# Auction context
- auction_item_count (size of parent auction)
- lot_number_normalized (position in auction, 0-1)
- auction_type (one-hot encoding)
```

### Phase 2: Historical Features (Weeks 3-4)

**Require historical data aggregation**:

```python
# Category statistics (trailing 30/90/365 days)
- category_avg_price_30d
- category_median_price_90d
- category_sell_rate_365d (% with bids)
- category_item_count (items in category)

# Location features
- location_avg_price
- is_major_city (Toronto, Montreal, Vancouver)
- distance_from_toronto_meters (available in local data!)
```

### Phase 3: Multi-Modal Features (Weeks 5-8)

**Text Model**:
- DistilBERT embeddings (768-dim) from title + description
- TF-IDF vectors for keyword matching
- Named entity recognition for brands

**Image Model**:
- EfficientNet-B0 embeddings (1280-dim) from primary image
- Multi-image aggregation (mean pooling)
- Image quality metrics

**Sequential Model**:
- LSTM over bid history time series
- Bid velocity and acceleration features
- Soft-close extension patterns

---

## Modeling Recommendations

### Recommended Approach: Hierarchical Ensemble

```
Level 1: Specialized Models
├── Tabular Model (XGBoost) ──────┐
├── Text Model (DistilBERT) ──────┤
├── Image Model (EfficientNet) ───┼──> Level 2: Fusion Model
└── Sequential Model (LSTM) ──────┘     (Meta-learner)
                                               │
                                               ▼
                                        Final Prediction
```

### Implementation Priority

**Priority 1: Baseline** (Week 1)
- Train XGBoost on core tabular features
- Establish baseline metrics (MAE, RMSE, R²)
- Target: MAE < $20 on validation set

**Priority 2: Enhanced Tabular** (Week 2-3)
- Add historical category features
- Add temporal features
- Implement two-stage approach for zero-bids
- Target: MAE < $15

**Priority 3: Text Model** (Week 4-5)
- Fine-tune DistilBERT on item descriptions
- Combine with tabular via early fusion
- Target: MAE < $12

**Priority 4: Multi-Modal Fusion** (Week 6-8)
- Add image model
- Add sequential model (if bid history available)
- Train meta-learner (Ridge or small NN)
- Target: MAE < $10

---

## Data Quality & Validation

### Available Data Assets

**28,816 Auction Locations** (confirmed in local data):
- Geographic coordinates (lat/lng)
- Postal codes
- Distance from Toronto
- **Use for**: location-based features

**Expected from HuggingFace Dataset**:
- Auction-level aggregated metrics
- Item-level details (title, description, category)
- Bidding statistics (if available)

### Validation Strategy

**Time-Based Split** (required due to temporal dependencies):
```
Train:      Months 1-10 (70%)
Validation: Months 11-12 (15%)
Test:       Months 13-15 (15%)
```

**Metrics**:
- Primary: MAE (interpretable in dollars)
- Secondary: RMSE, MAPE, R²
- Business: % predictions within $10 of actual

---

## Critical Success Factors

### ✅ Do's

1. **Use time-based validation** - critical for temporal data
2. **Implement two-stage model** - handles zero-bids properly
3. **Start with XGBoost baseline** - fast iteration
4. **Log-transform prices** - handles skewness
5. **Target encode categories** - handles high cardinality
6. **Use bid_count if available** - strongest signal

### ❌ Don'ts

1. **Don't use random splits** - causes data leakage
2. **Don't ignore zero-bids** - they're real outcomes
3. **Don't over-engineer initially** - baseline first
4. **Don't forget category statistics** - essential features
5. **Don't mix train/test locations** - if location matters

---

## Next Actions

### Immediate (This Week)
1. ✅ Complete EDA - **DONE**
2. Load actual HuggingFace dataset (jpearce610/auction_data)
3. Implement core tabular features in `src/features.py`
4. Set up data pipeline for feature engineering

### Short-term (Next 2 Weeks)
1. Train XGBoost baseline model
2. Implement two-stage approach
3. Add historical category features
4. Evaluate on time-based splits

### Medium-term (Next 4-8 Weeks)
1. Add text embeddings
2. Add image embeddings
3. Implement fusion model
4. Deploy to Hugging Face Spaces

---

## References

- **EDA Outputs**: `exploratory/summary.md`, `exploratory/figures/`
- **Feature Guide**: `exploratory/feature_engineering_guide.txt`
- **Schema**: `references/schema.sql`
- **Scraper Config**: `src/data/scraper_config.yaml`

---

**Analysis Completed**: 2026-01-26
**Analyst**: GitHub Copilot
**Status**: Ready for Implementation ✅
