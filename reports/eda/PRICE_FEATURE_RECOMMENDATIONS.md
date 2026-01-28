
# =============================================================================
# RECOMMENDATIONS: Price Feature Analysis
# =============================================================================

## EXECUTIVE SUMMARY

Based on the exploratory data analysis of the item_current_bid column from the
MaxSold item dataset, we provide the following key recommendations for machine
learning model design and implementation.

## KEY FINDINGS

1. **Zero-Bid Items**
   - A significant proportion of items receive no bids (item_current_bid = 0)
   - This creates a zero-inflated distribution that requires special handling
   - Zero-bid items represent a distinct prediction problem

2. **Price Distribution**
   - Non-zero prices are heavily right-skewed
   - Wide range of prices with outliers
   - Not normally distributed (fails normality tests)

3. **Transformation Effectiveness**
   - Log1p transformation significantly reduces skewness
   - Box-Cox and Yeo-Johnson provide data-driven transformations
   - All transformations improve normality but don't fully normalize

## RECOMMENDED APPROACH

### Strategy: Two-Stage Modeling (RECOMMENDED)

We strongly recommend implementing a two-stage model:

**Stage 1: Classification**
- Target: will_sell (binary: price > 0)
- Purpose: Identify items likely to receive no bids
- Models: Logistic Regression, Random Forest, Gradient Boosting
- Evaluation: AUC-ROC, Precision-Recall, F1-Score

**Stage 2: Regression**
- Target: log1p(price) for items with price > 0
- Purpose: Predict sale price conditional on selling
- Models: XGBoost, LightGBM, Neural Networks
- Evaluation: RMSE, MAE, MAPE on log scale

**Final Prediction:**
```python
P_sell = stage1_model.predict_proba(features)[:, 1]
log_price = stage2_model.predict(features)
final_price = P_sell * (np.exp(log_price) - 1)
```

**Rationale:**
- Separates two distinct processes (will it sell? vs. for how much?)
- Each model can specialize and optimize for its task
- More interpretable and debuggable
- Handles zeros naturally without mathematical issues

### Alternative: Single Model with Log1p Transformation

If two-stage is too complex:

**Transformation:** log1p(price) = log(price + 1)
**Model:** LightGBM or XGBoost with Tweedie objective
**Inverse Transform:** price = exp(prediction) - 1

**Pros:**
- Simpler architecture
- Handles zeros naturally
- Reduces right skewness

**Cons:**
- Single model must learn two different processes
- Less interpretable
- May underperform two-stage approach

## FEATURE ENGINEERING

### Features for Zero-Bid Prediction

Engineer features specifically designed to identify unsellable items:

1. **Starting Price Features**
   - starting_bid_percentile_in_category
   - starting_bid_vs_category_median_ratio
   - is_starting_bid_unreasonable (> 2x category median)

2. **Quality Indicators**
   - has_description (boolean)
   - description_length
   - num_images
   - has_brand (boolean)
   - condition_score (derived from condition field)

3. **Engagement Signals**
   - item_viewed (if available pre-auction)
   - views_per_day_since_listed
   - category_popularity_score

4. **Market Factors**
   - category_historical_sell_rate
   - auction_location_demand_score
   - day_of_week_close_time
   - hour_of_day_close_time

### Features for Price Prediction

For items that do sell:

1. **Item Characteristics**
   - All quality indicators from above
   - brand_value_score (based on historical brand prices)
   - category_one_hot_encoding
   - condition_ordinal_encoding

2. **Comparative Features**
   - similar_items_avg_price (same category, condition, brand)
   - price_position_in_auction (compared to other items)
   - starting_bid_to_historical_ratio

3. **Temporal Features**
   - days_until_close
   - is_weekend_close
   - is_peak_hour_close (5-9 PM)
   - season (if relevant for certain categories)

## DATA SPLITTING

### Time-Based Split (MANDATORY)

**Do NOT use random splits** - auction data is temporal:

```python
# Sort by auction end date
df = df.sort_values('auction_end_date')

# Split chronologically
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]
```

**Rationale:**
- Prevents data leakage from future to past
- Realistic production scenario (predict future auctions)
- Tests model's ability to generalize to new market conditions

### Stratification Considerations

Within each time-based split, ensure balanced representation:
- Stratify by has_bids (equal zero/non-zero proportions)
- Check price distribution consistency across splits
- Monitor category representation

## EVALUATION METRICS

### For Two-Stage Model

**Stage 1 (Classification):**
- Primary: AUC-ROC
- Secondary: Precision at 50% Recall
- Business: Cost-based metric (FP/FN costs)

**Stage 2 (Regression on sold items only):**
- Primary: RMSE on log scale
- Secondary: MAE, MAPE
- Business: Within-$X accuracy (% predictions within $10 of actual)

**Overall (Combined):**
```python
def hybrid_metric(y_true, y_pred):
    # Classification accuracy on zeros
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    zero_accuracy = (y_true_binary == y_pred_binary).mean()
    
    # Regression error on non-zeros
    mask = y_true > 0
    if mask.sum() > 0:
        mae = np.abs(y_true[mask] - y_pred[mask]).mean()
        relative_mae = mae / y_true[mask].mean()
    else:
        relative_mae = 0
    
    # Combined metric (tune weights)
    combined = 0.5 * zero_accuracy - 0.3 * relative_mae
    return combined
```

### For Single Model

- Primary: Hybrid metric (above)
- Secondary: RMSE including zeros
- Business: Total revenue error (sum of predictions vs actuals)

## DATA QUALITY CONSIDERATIONS

### Handling Missing Values

1. **Critical for Zero-Bid Prediction:**
   - Missing description → Likely indicator of low effort listing
   - Missing images → Strong zero-bid predictor
   - Use "missingness" as a feature (is_missing_description)

2. **Imputation Strategies:**
   - Numeric features: Median by category
   - Categorical features: Mode or "Unknown" category
   - Never drop rows due to missing features

### Outlier Treatment

- Do NOT remove price outliers - they are valid data points
- Consider capping predictions at 99th percentile for stability
- Use robust models (tree-based) that handle outliers naturally
- For neural networks, consider robust loss functions (Huber)

## MODEL RECOMMENDATIONS

### Ranked by Expected Performance

1. **LightGBM Two-Stage (BEST)**
   - Stage 1: LGBMClassifier for binary prediction
   - Stage 2: LGBMRegressor with log1p target
   - Handles zero-inflation, missing data, and categorical features
   - Fast training and inference

2. **XGBoost Two-Stage**
   - Similar to LightGBM but slightly different regularization
   - Excellent for structured data
   - Good explainability with SHAP

3. **Neural Network Multi-Task**
   - Two output heads (classification + regression)
   - Shared representations
   - Requires more data and tuning
   - Consider only if sufficient data (>100k items)

4. **LightGBM Single Model with Tweedie Loss**
   - Objective='tweedie', tweedie_variance_power=1.5
   - Designed for zero-inflated continuous data
   - Simpler than two-stage
   - May underperform two-stage approach

## IMPLEMENTATION ROADMAP

### Phase 1: Baseline (Week 1)
- [ ] Implement simple baselines (category median, zeros only)
- [ ] Establish evaluation metrics and infrastructure
- [ ] Create train/val/test splits (time-based)

### Phase 2: Two-Stage Model (Week 2)
- [ ] Train Stage 1: Zero-bid classifier
- [ ] Train Stage 2: Price regressor (non-zeros only)
- [ ] Combine predictions and evaluate
- [ ] Feature importance analysis

### Phase 3: Feature Engineering (Week 3)
- [ ] Engineer zero-bid prediction features
- [ ] Engineer price prediction features
- [ ] Test feature impact on metrics
- [ ] Remove low-importance features

### Phase 4: Hyperparameter Tuning (Week 4)
- [ ] Grid/random search for Stage 1
- [ ] Grid/random search for Stage 2
- [ ] Optimize combination weights
- [ ] Cross-validation on time-series splits

### Phase 5: Ensemble & Refinement (Week 5)
- [ ] Train multiple model types
- [ ] Ensemble predictions (weighted average, stacking)
- [ ] Calibrate probability estimates
- [ ] Final evaluation on held-out test set

## CRITICAL WARNINGS

### Data Leakage Prevention

⚠️ **DO NOT USE THESE FEATURES:**
- item_current_bid (this IS the target!)
- item_bid_count (leaks final price information)
- item_bidding_extended (leaks popularity)
- Any feature derived from bid history AFTER auction starts

⚠️ **CLARIFY PREDICTION TIME:**
- Pre-auction: Use only item attributes, historical data
- Mid-auction: Can use partial bid data (with caution)
- Post-auction: Academic exercise, not production use case

### Model Validation

- Always use time-based splits (never random)
- Validate on recent data (most recent 15% as test)
- Monitor distribution shift over time
- Check for temporal dependencies in residuals

### Business Metrics

- Model should optimize for business value, not just RMSE
- Consider costs of over-prediction vs under-prediction
- May want different models for different price ranges
- High-value items may need special handling

## EXPECTED PERFORMANCE

Based on similar auction prediction tasks:

**Stage 1 (Zero-Bid Classification):**
- Target AUC-ROC: 0.75-0.85
- Target Precision@50%: 0.70+

**Stage 2 (Price Regression on Non-Zeros):**
- Target RMSE (log scale): 0.3-0.5
- Target MAE: $10-$20 (depends on price range)
- Target Within-$10 Accuracy: 40-60%

**Overall:**
- Better than "predict category median" baseline by 30%+
- Better than "predict starting bid" baseline by 50%+

## NEXT STEPS

1. **Immediate:**
   - Implement baseline models for comparison
   - Set up evaluation framework
   - Create time-based data splits

2. **Short-term:**
   - Implement two-stage LightGBM model
   - Engineer zero-bid prediction features
   - Validate on held-out test set

3. **Medium-term:**
   - Experiment with alternative approaches (Tweedie, multi-task)
   - Feature selection and engineering
   - Hyperparameter optimization

4. **Long-term:**
   - Integrate other data modalities (images, text, bid sequences)
   - Develop ensemble/fusion model
   - Deploy to production with monitoring

## REFERENCES

- Issue #4: Data leakage considerations
- Issue #7: Bid history handling
- Dataset: https://huggingface.co/datasets/jpearce610/item_data
- Similar work: eBay auction price prediction, Airbnb pricing

## CONCLUSION

The price feature (item_current_bid) presents a classic zero-inflated regression
problem that is best addressed with a two-stage modeling approach. The first stage
should classify whether an item will sell, and the second stage should predict the
price for items that do sell using log-transformed targets.

This approach separates two distinct processes, allows each model to specialize,
and handles the zero-inflation naturally. Combined with proper time-based splitting,
zero-bid feature engineering, and robust evaluation metrics, this strategy should
yield the best performance for the auction price prediction task.

Generated by: EDA Analysis Script
Date: 2026-01-28 04:03:37
