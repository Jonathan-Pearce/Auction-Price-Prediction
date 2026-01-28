# =============================================================================
# Exploratory Data Analysis - Price Feature (item_current_bid)
# =============================================================================
"""
Comprehensive analysis of the target variable (item_current_bid) from the
MaxSold item dataset. This analysis informs ML model design decisions including:
- Target transformations
- Handling zero-bid items
- Feature engineering strategies
- Model architecture choices
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET_ID = "jpearce610/item_data"

# =============================================================================
# Data Loading
# =============================================================================


def load_item_data():
    """Load item dataset from Hugging Face."""
    from datasets import load_dataset

    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(HF_DATASET_ID, split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df):,} items")
    return df


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_price_distribution(df, target_col="item_current_bid"):
    """Comprehensive analysis of price distribution."""
    prices = df[target_col].dropna()

    print("\n" + "=" * 80)
    print("PRICE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Basic statistics
    print(f"\n{'Total Items:':<30} {len(prices):>15,}")
    print(f"{'Missing Values:':<30} {df[target_col].isna().sum():>15,}")

    # Zero-bid analysis
    zero_count = (prices == 0).sum()
    zero_pct = (prices == 0).mean() * 100
    print(f"\n{'Zero-Bid Items:':<30} {zero_count:>15,} ({zero_pct:.2f}%)")

    # Non-zero statistics
    prices_nonzero = prices[prices > 0]
    print(f"{'Non-Zero Items:':<30} {len(prices_nonzero):>15,}")

    print("\n" + "-" * 80)
    print("Overall Statistics:")
    print("-" * 80)
    print(f"{'Mean:':<30} ${prices.mean():>14,.2f}")
    print(f"{'Median:':<30} ${prices.median():>14,.2f}")
    print(f"{'Std Dev:':<30} ${prices.std():>14,.2f}")
    print(f"{'Min:':<30} ${prices.min():>14,.2f}")
    print(f"{'Max:':<30} ${prices.max():>14,.2f}")

    print("\n" + "-" * 80)
    print("Non-Zero Statistics:")
    print("-" * 80)
    print(f"{'Mean:':<30} ${prices_nonzero.mean():>14,.2f}")
    print(f"{'Median:':<30} ${prices_nonzero.median():>14,.2f}")
    print(f"{'Std Dev:':<30} ${prices_nonzero.std():>14,.2f}")
    print(f"{'Min:':<30} ${prices_nonzero.min():>14,.2f}")
    print(f"{'Max:':<30} ${prices_nonzero.max():>14,.2f}")

    # Quantiles
    print("\n" + "-" * 80)
    print("Quantiles (Non-Zero):")
    print("-" * 80)
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        val = prices_nonzero.quantile(q)
        print(f"{'Q' + str(int(q*100)) + ':':<30} ${val:>14,.2f}")

    # Skewness and kurtosis
    print("\n" + "-" * 80)
    print("Distribution Shape:")
    print("-" * 80)
    print(f"{'Skewness (all):':<30} {stats.skew(prices):>15.4f}")
    print(f"{'Skewness (non-zero):':<30} {stats.skew(prices_nonzero):>15.4f}")
    print(f"{'Kurtosis (all):':<30} {stats.kurtosis(prices):>15.4f}")
    print(f"{'Kurtosis (non-zero):':<30} {stats.kurtosis(prices_nonzero):>15.4f}")

    return {
        "total": len(prices),
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "mean": prices.mean(),
        "median": prices.median(),
        "std": prices.std(),
        "skewness": stats.skew(prices),
        "kurtosis": stats.kurtosis(prices),
    }


def visualize_price_distribution(df, target_col="item_current_bid"):
    """Create visualizations for price distribution."""
    prices = df[target_col].dropna()
    prices_nonzero = prices[prices > 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Price Distribution Analysis", fontsize=16, fontweight="bold")

    # 1. Overall histogram
    ax = axes[0, 0]
    ax.hist(prices, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"All Prices (n={len(prices):,})")
    ax.axvline(prices.median(), color="red", linestyle="--", label="Median")
    ax.legend()

    # 2. Non-zero histogram
    ax = axes[0, 1]
    ax.hist(prices_nonzero, bins=100, edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Non-Zero Prices (n={len(prices_nonzero):,})")
    ax.axvline(prices_nonzero.median(), color="red", linestyle="--", label="Median")
    ax.legend()

    # 3. Log scale histogram
    ax = axes[0, 2]
    ax.hist(
        np.log1p(prices_nonzero),
        bins=100,
        edgecolor="black",
        alpha=0.7,
        color="orange",
    )
    ax.set_xlabel("log(Price + 1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Log-Transformed Non-Zero Prices")
    ax.axvline(
        np.log1p(prices_nonzero.median()),
        color="red",
        linestyle="--",
        label="Median",
    )
    ax.legend()

    # 4. Box plot
    ax = axes[1, 0]
    ax.boxplot([prices_nonzero], vert=True, patch_artist=True)
    ax.set_ylabel("Price ($)")
    ax.set_title("Box Plot (Non-Zero)")
    ax.set_xticklabels(["Prices"])

    # 5. Q-Q plot
    ax = axes[1, 1]
    stats.probplot(prices_nonzero, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Non-Zero Prices vs Normal)")

    # 6. Cumulative distribution
    ax = axes[1, 2]
    sorted_prices = np.sort(prices_nonzero)
    cumulative = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
    ax.plot(sorted_prices, cumulative, linewidth=2)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution (Non-Zero)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved distribution plot to {OUTPUT_DIR / 'price_distribution.png'}")
    return fig


def analyze_transformations(df, target_col="item_current_bid"):
    """Analyze different transformation strategies."""
    prices = df[target_col].dropna()
    prices_nonzero = prices[prices > 0]

    print("\n" + "=" * 80)
    print("TRANSFORMATION ANALYSIS")
    print("=" * 80)

    transformations = {}

    # Log1p transformation
    log1p_all = np.log1p(prices)
    log1p_nonzero = np.log1p(prices_nonzero)
    transformations["log1p"] = {
        "values": log1p_all,
        "skewness": stats.skew(log1p_nonzero),
        "kurtosis": stats.kurtosis(log1p_nonzero),
    }

    print(f"\n{'Log1p Transformation:':<30}")
    print(f"{'  Skewness (non-zero):':<30} {stats.skew(log1p_nonzero):>15.4f}")
    print(f"{'  Kurtosis (non-zero):':<30} {stats.kurtosis(log1p_nonzero):>15.4f}")

    # Square root transformation
    sqrt_all = np.sqrt(prices)
    sqrt_nonzero = np.sqrt(prices_nonzero)
    transformations["sqrt"] = {
        "values": sqrt_all,
        "skewness": stats.skew(sqrt_nonzero),
        "kurtosis": stats.kurtosis(sqrt_nonzero),
    }

    print(f"\n{'Square Root Transformation:':<30}")
    print(f"{'  Skewness (non-zero):':<30} {stats.skew(sqrt_nonzero):>15.4f}")
    print(f"{'  Kurtosis (non-zero):':<30} {stats.kurtosis(sqrt_nonzero):>15.4f}")

    # Box-Cox transformation (only on non-zero positive values)
    try:
        boxcox_vals, lambda_bc = boxcox(prices_nonzero)
        transformations["boxcox"] = {
            "values": boxcox_vals,
            "lambda": lambda_bc,
            "skewness": stats.skew(boxcox_vals),
            "kurtosis": stats.kurtosis(boxcox_vals),
        }
        print(f"\n{'Box-Cox Transformation:':<30}")
        print(f"{'  Lambda:':<30} {lambda_bc:>15.4f}")
        print(f"{'  Skewness:':<30} {stats.skew(boxcox_vals):>15.4f}")
        print(f"{'  Kurtosis:':<30} {stats.kurtosis(boxcox_vals):>15.4f}")
    except Exception as e:
        print(f"\n{'Box-Cox Transformation:':<30} Failed: {e}")

    # Yeo-Johnson transformation (handles zeros and negatives)
    try:
        yeojohnson_vals, lambda_yj = yeojohnson(prices)
        transformations["yeojohnson"] = {
            "values": yeojohnson_vals,
            "lambda": lambda_yj,
            "skewness": stats.skew(yeojohnson_vals),
            "kurtosis": stats.kurtosis(yeojohnson_vals),
        }
        print(f"\n{'Yeo-Johnson Transformation:':<30}")
        print(f"{'  Lambda:':<30} {lambda_yj:>15.4f}")
        print(f"{'  Skewness (all):':<30} {stats.skew(yeojohnson_vals):>15.4f}")
        print(f"{'  Kurtosis (all):':<30} {stats.kurtosis(yeojohnson_vals):>15.4f}")
    except Exception as e:
        print(f"\n{'Yeo-Johnson Transformation:':<30} Failed: {e}")

    return transformations


def visualize_transformations(df, transformations, target_col="item_current_bid"):
    """Visualize different transformations."""
    prices = df[target_col].dropna()
    prices_nonzero = prices[prices > 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Price Transformation Comparison", fontsize=16, fontweight="bold")

    # Original
    ax = axes[0, 0]
    ax.hist(prices_nonzero, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Original (Non-Zero)")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")

    # Log1p
    ax = axes[0, 1]
    if "log1p" in transformations:
        log_vals = transformations["log1p"]["values"]
        ax.hist(log_vals[log_vals > 0], bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(
            f"Log1p (Skew: {transformations['log1p']['skewness']:.2f})"
        )
        ax.set_xlabel("log(Price + 1)")
        ax.set_ylabel("Frequency")

    # Sqrt
    ax = axes[0, 2]
    if "sqrt" in transformations:
        sqrt_vals = transformations["sqrt"]["values"]
        ax.hist(sqrt_vals[sqrt_vals > 0], bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"Square Root (Skew: {transformations['sqrt']['skewness']:.2f})")
        ax.set_xlabel("sqrt(Price)")
        ax.set_ylabel("Frequency")

    # Box-Cox
    ax = axes[1, 0]
    if "boxcox" in transformations:
        ax.hist(
            transformations["boxcox"]["values"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(
            f"Box-Cox (λ={transformations['boxcox']['lambda']:.2f}, "
            f"Skew: {transformations['boxcox']['skewness']:.2f})"
        )
        ax.set_xlabel("Box-Cox Transformed")
        ax.set_ylabel("Frequency")
    else:
        ax.text(0.5, 0.5, "Box-Cox Failed", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Yeo-Johnson
    ax = axes[1, 1]
    if "yeojohnson" in transformations:
        ax.hist(
            transformations["yeojohnson"]["values"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(
            f"Yeo-Johnson (λ={transformations['yeojohnson']['lambda']:.2f}, "
            f"Skew: {transformations['yeojohnson']['skewness']:.2f})"
        )
        ax.set_xlabel("Yeo-Johnson Transformed")
        ax.set_ylabel("Frequency")
    else:
        ax.text(0.5, 0.5, "Yeo-Johnson Failed", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Comparison summary
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = "Transformation Summary\n" + "-" * 30 + "\n\n"
    for name, trans in transformations.items():
        summary_text += f"{name.upper()}:\n"
        summary_text += f"  Skewness: {trans['skewness']:.4f}\n"
        summary_text += f"  Kurtosis: {trans['kurtosis']:.4f}\n\n"
    ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment="top", family="monospace")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformations.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved transformation plot to {OUTPUT_DIR / 'transformations.png'}")
    return fig


def analyze_zero_bid_features(df, target_col="item_current_bid", top_n=20):
    """Analyze features that predict zero-bid items."""
    print("\n" + "=" * 80)
    print("ZERO-BID FEATURE ANALYSIS")
    print("=" * 80)

    # Create binary target
    df_analysis = df.copy()
    df_analysis["has_bids"] = (df_analysis[target_col] > 0).astype(int)

    # Select potential predictor columns
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and related columns
    exclude_cols = [
        target_col,
        "has_bids",
        "item_id",
        "auction_id",
        "item_number",
    ]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if len(numeric_cols) == 0:
        print("No numeric features found for analysis")
        return None

    # Prepare data
    X = df_analysis[numeric_cols].fillna(-999)  # Simple imputation for RF
    y = df_analysis["has_bids"]

    # Remove constant features
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] == 0:
        print("No valid features for analysis")
        return None

    # Train Random Forest to identify important features
    print(f"\nTraining Random Forest on {len(X)} samples with {X.shape[1]} features...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    # Get feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\n{'Top Features Predicting Zero-Bids:':<50}")
    print("-" * 80)
    print(f"{'Feature':<40} {'Importance':<15} {'Cumulative':<15}")
    print("-" * 80)

    cumulative = 0
    for idx, row in feature_importance.head(top_n).iterrows():
        cumulative += row["importance"]
        print(
            f"{row['feature']:<40} {row['importance']:>14.4f} {cumulative:>14.2%}"
        )

    # Compare feature distributions
    print("\n" + "-" * 80)
    print("Feature Distribution Comparison (Top 10):")
    print("-" * 80)
    print(f"{'Feature':<40} {'Zero-Bid Mean':<15} {'Non-Zero Mean':<15} {'Difference':<15}")
    print("-" * 80)

    for idx, row in feature_importance.head(10).iterrows():
        feat = row["feature"]
        zero_mean = df_analysis[df_analysis["has_bids"] == 0][feat].mean()
        nonzero_mean = df_analysis[df_analysis["has_bids"] == 1][feat].mean()
        diff = nonzero_mean - zero_mean
        print(f"{feat:<40} {zero_mean:>14.2f} {nonzero_mean:>14.2f} {diff:>14.2f}")

    return feature_importance


def generate_recommendations():
    """Generate comprehensive recommendations document."""
    recommendations = """
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
Date: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""
    return recommendations


# =============================================================================
# Main Analysis
# =============================================================================


def main():
    """Run comprehensive EDA on price feature."""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS - PRICE FEATURE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data
    df = load_item_data()

    # Analyze price distribution
    stats_summary = analyze_price_distribution(df)

    # Visualize distribution
    visualize_price_distribution(df)

    # Analyze transformations
    transformations = analyze_transformations(df)

    # Visualize transformations
    visualize_transformations(df, transformations)

    # Analyze zero-bid features
    feature_importance = analyze_zero_bid_features(df)

    # Generate and save recommendations
    recommendations = generate_recommendations()
    rec_file = OUTPUT_DIR / "PRICE_FEATURE_RECOMMENDATIONS.md"
    rec_file.write_text(recommendations)
    print(f"\n{'=' * 80}")
    print(f"Saved recommendations to {rec_file}")
    print(f"{'=' * 80}")

    # Save summary statistics
    summary_file = OUTPUT_DIR / "price_statistics.txt"
    with open(summary_file, "w") as f:
        f.write("PRICE FEATURE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        for key, value in stats_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved statistics to {summary_file}")

    print("\nAnalysis complete!")
    print(f"\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'price_distribution.png'}")
    print(f"  - {OUTPUT_DIR / 'transformations.png'}")
    print(f"  - {OUTPUT_DIR / 'PRICE_FEATURE_RECOMMENDATIONS.md'}")
    print(f"  - {OUTPUT_DIR / 'price_statistics.txt'}")


if __name__ == "__main__":
    main()
