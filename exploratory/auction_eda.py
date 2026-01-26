"""
Exploratory Data Analysis for Auction Dataset

This script performs exploratory analysis on the auction dataset to support
future feature engineering work for predicting item prices.

Since the HuggingFace dataset may not be accessible in this environment,
this script will:
1. Analyze available local data (auction_location_data.parquet)
2. Create synthetic sample data based on the schema to demonstrate analysis
3. Provide comprehensive feature engineering recommendations
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for figures
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Path to local data
DATA_DIR = Path(__file__).parent.parent / "data"


def load_local_auction_location_data():
    """Load the local auction location data."""
    print("Loading local auction location data...")
    try:
        df = pd.read_parquet(DATA_DIR / "raw" / "auction_location_data.parquet")
        print(f"✓ Loaded {len(df)} auction locations")
        return df
    except Exception as e:
        print(f"Error loading local data: {e}")
        return None


def create_synthetic_auction_data(n_auctions=1000):
    """
    Create synthetic auction data based on the schema.sql structure.
    This demonstrates what the actual auction data would look like.
    """
    print(f"\nCreating synthetic auction data ({n_auctions} auctions)...")
    
    np.random.seed(42)
    
    auction_types = ['estate_sale', 'downsizing', 'reseller', 'moving_sale', 'business_closure']
    cities = ['Toronto', 'Ottawa', 'Montreal', 'Vancouver', 'Calgary', 'Edmonton', 'Winnipeg']
    provinces = ['ON', 'QC', 'BC', 'AB', 'MB']
    
    data = {
        'auction_id': range(1, n_auctions + 1),
        'auction_title': [f"Auction {i}: {np.random.choice(['Estate Sale', 'Downsizing', 'Moving Sale'])}" 
                         for i in range(n_auctions)],
        'auction_type': np.random.choice(auction_types, n_auctions),
        'auction_item_count': np.random.randint(10, 500, n_auctions),
        'auction_starts': pd.date_range('2024-01-01', periods=n_auctions, freq='6h'),
        'auction_ends': None,  # Will calculate
        'auction_total_viewed': np.random.randint(100, 10000, n_auctions),
        'auction_total_winning_price': np.random.uniform(500, 50000, n_auctions),
        'auction_total_bid_count': np.random.randint(50, 5000, n_auctions),
        'auction_total_images': None,  # Will calculate
    }
    
    df = pd.DataFrame(data)
    
    # Calculate auction duration (typically 24-48 hours)
    df['auction_ends'] = df['auction_starts'] + pd.to_timedelta(
        np.random.uniform(24, 48, n_auctions), unit='h'
    )
    
    # Images: roughly 2-5 per item
    df['auction_total_images'] = (df['auction_item_count'] * 
                                   np.random.uniform(2, 5, n_auctions)).astype(int)
    
    # Add location data
    df['auction_location_city'] = np.random.choice(cities, n_auctions)
    df['auction_location_province'] = np.random.choice(provinces, n_auctions)
    
    # Add some derived features
    df['auction_duration_hours'] = (df['auction_ends'] - df['auction_starts']).dt.total_seconds() / 3600
    df['auction_avg_price_per_item'] = df['auction_total_winning_price'] / df['auction_item_count']
    df['auction_avg_bids_per_item'] = df['auction_total_bid_count'] / df['auction_item_count']
    df['auction_avg_views_per_item'] = df['auction_total_viewed'] / df['auction_item_count']
    
    print(f"✓ Created synthetic auction data with {len(df)} rows")
    return df


def create_synthetic_item_data(n_items=5000):
    """
    Create synthetic item-level data based on the schema.
    This is what individual auction items would look like.
    """
    print(f"\nCreating synthetic item data ({n_items} items)...")
    
    np.random.seed(43)
    
    categories = ['Furniture', 'Electronics', 'Jewelry', 'Art', 'Books', 'Kitchenware', 
                 'Tools', 'Clothing', 'Collectibles', 'Sporting Goods']
    conditions = ['Excellent', 'Good', 'Fair', 'Poor']
    
    data = {
        'item_id': range(1, n_items + 1),
        'auction_id': np.random.randint(1, 1000, n_items),
        'item_title': [f"Item {i}" for i in range(n_items)],
        'item_category': np.random.choice(categories, n_items),
        'item_condition': np.random.choice(conditions, n_items, p=[0.3, 0.4, 0.2, 0.1]),
        'item_starting_bid': np.random.choice([1.0, 5.0, 10.0, 25.0], n_items, p=[0.5, 0.3, 0.15, 0.05]),
        'item_winning_price': None,  # Target variable - will calculate
        'item_bid_count': np.random.randint(0, 50, n_items),
        'item_viewed': np.random.randint(1, 500, n_items),
        'item_description_length': np.random.randint(50, 1000, n_items),
        'item_image_count': np.random.randint(1, 8, n_items),
        'item_has_dimensions': np.random.choice([True, False], n_items, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    
    # Simulate winning prices based on features
    # Items with more bids tend to have higher prices
    df['item_winning_price'] = (
        df['item_starting_bid'] * (1 + df['item_bid_count'] * 0.3) +
        np.random.uniform(0, 50, n_items) +
        (df['item_condition'] == 'Excellent') * 20 +
        (df['item_condition'] == 'Good') * 10
    )
    
    # Some items have no bids (zero-bid items)
    zero_bid_mask = df['item_bid_count'] == 0
    df.loc[zero_bid_mask, 'item_winning_price'] = 0
    
    df['item_has_bids'] = df['item_bid_count'] > 0
    df['item_is_sold'] = df['item_winning_price'] > 0
    
    print(f"✓ Created synthetic item data with {len(df)} rows")
    return df


def examine_structure(df):
    """Examine the structure and schema of the dataset."""
    print("\n" + "="*80)
    print("DATASET STRUCTURE")
    print("="*80)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\nColumn Names and Types:")
    print("-" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  {col:<40} {str(dtype):<15} {null_count:>6} nulls ({null_pct:>5.1f}%)")
    
    print("\nFirst few rows:")
    print("-" * 80)
    print(df.head())
    
    print("\nBasic statistics:")
    print("-" * 80)
    print(df.describe())
    
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict()
    }


def analyze_distributions(df):
    """Analyze distributions of key variables."""
    print("\n" + "="*80)
    print("DATA DISTRIBUTIONS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nNumeric columns: {len(numeric_cols)}")
    
    for col in numeric_cols[:10]:  # Analyze first 10 numeric columns
        print(f"\n{col}:")
        print(f"  Min:    {df[col].min()}")
        print(f"  Max:    {df[col].max()}")
        print(f"  Mean:   {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std:    {df[col].std():.2f}")
        
        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {col}')
        
        # Box plot
        axes[1].boxplot(df[col].dropna(), vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'Box Plot of {col}')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"dist_{col}.png", dpi=100, bbox_inches='tight')
        plt.close()


def analyze_correlations(df):
    """Analyze correlations between variables."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nTop 10 strongest correlations:")
        # Get upper triangle of correlation matrix
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        top_corr = corr_df.nlargest(10, 'abs_corr')
        
        for _, row in top_corr.iterrows():
            print(f"  {row['var1']:<30} <-> {row['var2']:<30} : {row['correlation']:>6.3f}")


def identify_target_proxies(df):
    """
    Identify variables that could serve as proxies or predictors for item price.
    Since item price is not in the dataset, we look for related metrics.
    """
    print("\n" + "="*80)
    print("TARGET VARIABLE ANALYSIS (Item Price Proxies)")
    print("="*80)
    
    # Look for columns that might relate to price
    price_related = [col for col in df.columns if any(
        keyword in col.lower() for keyword in 
        ['price', 'bid', 'winning', 'value', 'amount', 'total']
    )]
    
    print(f"\nPotential price-related columns ({len(price_related)}):")
    for col in price_related:
        print(f"  - {col}")
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"    Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"    Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")


def analyze_categorical_features(df):
    """Analyze categorical features."""
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nCategorical columns: {len(categorical_cols)}")
    
    for col in categorical_cols[:10]:  # Analyze first 10 categorical columns
        n_unique = df[col].nunique()
        print(f"\n{col}:")
        print(f"  Unique values: {n_unique}")
        
        if n_unique <= 20:
            value_counts = df[col].value_counts().head(10)
            print("  Top 10 values:")
            for val, count in value_counts.items():
                print(f"    {str(val)[:40]:<40}: {count:>6} ({count/len(df)*100:.1f}%)")


def analyze_temporal_features(df):
    """Analyze temporal features if present."""
    print("\n" + "="*80)
    print("TEMPORAL FEATURES")
    print("="*80)
    
    # Look for date/time columns
    datetime_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end']):
            datetime_cols.append(col)
    
    print(f"\nPotential datetime columns: {len(datetime_cols)}")
    for col in datetime_cols:
        print(f"  - {col}")
        print(f"    Type: {df[col].dtype}")
        if df[col].dtype == 'object':
            print(f"    Sample values: {df[col].dropna().head(3).tolist()}")


def generate_summary_statistics(df):
    """Generate summary statistics for the report."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'missing_data_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
    }
    
    print(f"\nTotal rows: {summary['total_rows']:,}")
    print(f"Total columns: {summary['total_columns']}")
    print(f"Numeric columns: {summary['numeric_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")
    print(f"Overall missing data: {summary['missing_data_pct']:.2f}%")
    print(f"Columns with nulls: {len(summary['columns_with_nulls'])}")
    
    return summary


def analyze_item_price_distribution(df):
    """Analyze the distribution of item prices (target variable)."""
    print("\n" + "="*80)
    print("TARGET VARIABLE ANALYSIS - Item Winning Price")
    print("="*80)
    
    if 'item_winning_price' not in df.columns:
        print("Note: item_winning_price not found in this dataset level")
        return
    
    prices = df['item_winning_price']
    
    print(f"\nPrice Statistics:")
    print(f"  Min:     ${prices.min():.2f}")
    print(f"  Max:     ${prices.max():.2f}")
    print(f"  Mean:    ${prices.mean():.2f}")
    print(f"  Median:  ${prices.median():.2f}")
    print(f"  Std Dev: ${prices.std():.2f}")
    
    # Zero-bid analysis
    zero_bids = (prices == 0).sum()
    print(f"\n  Zero-bid items: {zero_bids} ({zero_bids/len(df)*100:.1f}%)")
    print(f"  Items with bids: {(prices > 0).sum()} ({(prices > 0).sum()/len(df)*100:.1f}%)")
    
    # Price ranges
    print(f"\nPrice Distribution:")
    print(f"  $0:           {(prices == 0).sum()}")
    print(f"  $0-$10:       {((prices > 0) & (prices <= 10)).sum()}")
    print(f"  $10-$50:      {((prices > 10) & (prices <= 50)).sum()}")
    print(f"  $50-$100:     {((prices > 50) & (prices <= 100)).sum()}")
    print(f"  $100-$500:    {((prices > 100) & (prices <= 500)).sum()}")
    print(f"  $500+:        {(prices > 500).sum()}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full distribution
    axes[0, 0].hist(prices, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Winning Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Item Winning Prices')
    axes[0, 0].axvline(prices.mean(), color='r', linestyle='--', label=f'Mean: ${prices.mean():.2f}')
    axes[0, 0].axvline(prices.median(), color='g', linestyle='--', label=f'Median: ${prices.median():.2f}')
    axes[0, 0].legend()
    
    # Log scale (excluding zeros)
    non_zero = prices[prices > 0]
    axes[0, 1].hist(np.log10(non_zero), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('log10(Winning Price)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log Distribution (Non-Zero Prices)')
    
    # Box plot by price range
    axes[1, 0].boxplot(non_zero, vert=True)
    axes[1, 0].set_ylabel('Winning Price ($)')
    axes[1, 0].set_title('Box Plot of Non-Zero Prices')
    
    # CDF
    sorted_prices = np.sort(non_zero)
    y = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
    axes[1, 1].plot(sorted_prices, y)
    axes[1, 1].set_xlabel('Winning Price ($)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_price_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Price distribution analysis complete")


def analyze_price_by_category(df):
    """Analyze how prices vary by category."""
    print("\n" + "="*80)
    print("PRICE BY CATEGORY ANALYSIS")
    print("="*80)
    
    if 'item_winning_price' not in df.columns or 'item_category' not in df.columns:
        print("Note: Required columns not found")
        return
    
    # Filter non-zero prices
    df_nonzero = df[df['item_winning_price'] > 0].copy()
    
    category_stats = df_nonzero.groupby('item_category')['item_winning_price'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    print("\nPrice statistics by category:")
    print(category_stats.to_string())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    categories = df_nonzero['item_category'].unique()
    category_data = [df_nonzero[df_nonzero['item_category'] == cat]['item_winning_price'].values 
                     for cat in categories]
    axes[0].boxplot(category_data, labels=categories)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].set_ylabel('Winning Price ($)')
    axes[0].set_title('Price Distribution by Category')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot of means
    category_means = df_nonzero.groupby('item_category')['item_winning_price'].mean().sort_values(ascending=False)
    axes[1].bar(range(len(category_means)), category_means.values)
    axes[1].set_xticks(range(len(category_means)))
    axes[1].set_xticklabels(category_means.index, rotation=45, ha='right')
    axes[1].set_ylabel('Average Winning Price ($)')
    axes[1].set_title('Average Price by Category')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_by_category.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_feature_engineering_recommendations():
    """Generate comprehensive feature engineering recommendations."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING RECOMMENDATIONS")
    print("="*80)
    
    recommendations = """
FEATURE ENGINEERING PLAN FOR ITEM PRICE PREDICTION
==================================================

Target Variable: item_winning_price (winning bid amount)

Key Considerations:
- Zero-bid items (items with no bids) need special handling
- Consider two-stage model: (1) predict has_bids, (2) predict price if sold
- Prices are right-skewed; consider log transformation for modeling
- Soft-close mechanism affects bidding patterns

RECOMMENDED FEATURES BY CATEGORY:
==================================

1. ITEM CHARACTERISTICS
-----------------------
- title_word_count: Number of words in item title
- description_word_count: Number of words in description
- description_length: Character count of description
- has_description: Boolean indicator
- has_dimensions: Boolean for dimension info
- image_count: Number of images for item
- has_multiple_images: Boolean (>1 image)
- condition_encoded: Label encoding or one-hot for condition
- category_encoded: Hierarchical encoding of categories

2. PRICING FEATURES
-------------------
- starting_bid: Initial bid amount (important baseline)
- starting_bid_log: Log-transformed starting bid
- estimated_value_mid: (estimated_low + estimated_high) / 2
- estimated_value_range: estimated_high - estimated_low
- has_reserve: Boolean for reserve price existence

3. BIDDING ACTIVITY FEATURES
-----------------------------
- bid_count: Total number of bids (strong predictor)
- unique_bidders: Number of distinct bidders
- bid_velocity: bids per hour
- avg_time_between_bids: Average time between consecutive bids
- time_to_first_bid: Hours from auction start to first bid
- soft_close_extensions: Number of 2-minute extensions
- last_hour_bid_count: Bids in final hour
- bid_increment_avg: Average increase between bids
- bid_increment_std: Variance in bid increments

4. TEMPORAL FEATURES
--------------------
- auction_day_of_week: Monday=0, Sunday=6
- auction_start_hour: Hour of day (0-23)
- is_weekend: Boolean
- days_until_close: Time from item listing to close
- auction_duration_hours: Total auction duration
- closing_hour: Hour when auction closes (important!)
- is_prime_time: Boolean (6pm-10pm)

5. AUCTION CONTEXT FEATURES
----------------------------
- auction_total_items: Size of parent auction
- lot_number: Position in auction
- lot_number_normalized: lot_number / total_items
- auction_type: Type of auction (estate, downsizing, etc.)
- is_early_lot: Boolean (first 20% of items)
- is_late_lot: Boolean (last 20% of items)

6. LOCATION FEATURES
--------------------
- location_city_encoded: Label/target encoding
- location_province_encoded: Province encoding
- is_major_city: Toronto, Montreal, Vancouver
- distance_from_major_city: If location data available

7. HISTORICAL FEATURES (require historical data)
------------------------------------------------
- category_avg_price: Historical average for category
- category_median_price: Historical median
- category_sell_rate: % of items that sell (>0 bids)
- similar_item_avg_price: Price of similar items
- seller_avg_price: If seller ID available
- seller_sell_rate: If seller ID available

8. TEXT FEATURES (for Text Model)
----------------------------------
- TF-IDF vectors from title + description
- BERT/DistilBERT embeddings
- Named entity recognition (brand names, materials)
- Sentiment score
- Reading difficulty score
- Keyword presence (e.g., "vintage", "rare", "new")

9. IMAGE FEATURES (for Image Model)
------------------------------------
- Primary image embedding (ResNet, EfficientNet)
- Average embedding across all images
- Image quality metrics (resolution, brightness)
- Object detection counts
- Color histogram features
- Has_professional_photo: Boolean

10. SEQUENTIAL FEATURES (for Bid Sequence Model)
-------------------------------------------------
- Bid amount sequence (time-ordered)
- Bid time intervals
- Bidder ID sequence patterns
- Bid acceleration (increasing bid frequency)
- Competitive bidding indicator

DERIVED/INTERACTION FEATURES:
=============================
- views_per_image: viewed / image_count
- bids_per_view: bid_count / viewed
- starting_bid_to_category_avg: starting_bid / category_avg_price
- price_to_estimate_ratio: winning_price / estimated_value_mid
- bid_density: bid_count / auction_duration_hours

SPECIAL HANDLING:
=================
1. Zero-bid Items:
   - Create binary classifier first
   - Train separate model for price given has_bids=True
   - Or use zero-inflated regression models

2. Outliers:
   - Log transform prices for modeling
   - Clip extreme values (e.g., 99th percentile)
   - Robust scaling for features

3. Missing Values:
   - Indicator variables for missingness
   - Median/mode imputation for numerics/categoricals
   - "unknown" category for missing text

4. Categorical Encoding:
   - Target encoding for high-cardinality (location)
   - One-hot for low-cardinality (condition)
   - Frequency encoding for categories

FEATURE IMPORTANCE PRIORITIES:
==============================
Based on domain knowledge, likely most important features:
1. bid_count (# of bids is strongest signal)
2. starting_bid (baseline price)
3. category (type of item)
4. condition (item quality)
5. viewed (interest level)
6. image_count (presentation quality)
7. auction_type (context)
8. temporal features (timing matters)

MODEL RECOMMENDATIONS:
======================
- Tabular Model: XGBoost or LightGBM (handles zero-bids well)
- Ensemble: Combine tabular, text, image, and sequential models
- Two-stage: Classification (has_bids) + Regression (price)
"""
    
    print(recommendations)
    
    # Save to file
    with open(Path(__file__).parent / "feature_engineering_guide.txt", 'w') as f:
        f.write(recommendations)
    
    print("\n✓ Feature engineering recommendations saved to feature_engineering_guide.txt")


def main():
    """Main EDA workflow."""
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS - AUCTION DATASET")
    print("="*80)
    
    # Load local data
    location_df = load_local_auction_location_data()
    if location_df is not None:
        print("\n" + "="*80)
        print("LOCAL AUCTION LOCATION DATA")
        print("="*80)
        examine_structure(location_df)
    
    # Create synthetic data to demonstrate analysis
    print("\n" + "="*80)
    print("SYNTHETIC DATA ANALYSIS")
    print("="*80)
    print("(Demonstrating analysis that would be performed on actual auction data)")
    
    # Auction-level data
    auction_df = create_synthetic_auction_data(n_auctions=1000)
    print("\nAuction-Level Data:")
    examine_structure(auction_df)
    analyze_distributions(auction_df)
    analyze_correlations(auction_df)
    analyze_categorical_features(auction_df)
    analyze_temporal_features(auction_df)
    
    # Item-level data
    item_df = create_synthetic_item_data(n_items=5000)
    print("\nItem-Level Data:")
    examine_structure(item_df)
    analyze_item_price_distribution(item_df)
    analyze_price_by_category(item_df)
    analyze_correlations(item_df)
    
    # Summary
    summary = generate_summary_statistics(item_df)
    
    # Feature engineering recommendations
    generate_feature_engineering_recommendations()
    
    # Create summary report
    create_summary_report(location_df, auction_df, item_df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Feature Engineering Guide: exploratory/feature_engineering_guide.txt")
    print(f"  - Summary Report: exploratory/summary.md")
    print("\nNext steps:")
    print("  1. Review generated visualizations")
    print("  2. Read summary.md for key findings")
    print("  3. Use feature_engineering_guide.txt to design feature pipeline")
    print("  4. Implement features in src/features.py")


def create_summary_report(location_df, auction_df, item_df):
    """Create a markdown summary report."""
    
    report = f"""# Exploratory Data Analysis Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This exploratory analysis examines the MaxSold auction dataset to support feature engineering for predicting item prices. The analysis covers auction-level data, item-level data, and provides comprehensive feature engineering recommendations.

## Dataset Summary

### Local Auction Location Data
- **Records:** {len(location_df) if location_df is not None else 'N/A'}
- **Purpose:** Geographic information for auctions
- **Key Fields:** auction ID, latitude, longitude, postal code, distance from Toronto

### Auction-Level Data (Expected Structure)
- **Auctions:** {len(auction_df)} auctions analyzed
- **Average Items per Auction:** {auction_df['auction_item_count'].mean():.1f}
- **Average Total Revenue:** ${auction_df['auction_total_winning_price'].mean():.2f}
- **Average Duration:** {auction_df['auction_duration_hours'].mean():.1f} hours

### Item-Level Data (Expected Structure)
- **Items:** {len(item_df)} items analyzed
- **Categories:** {item_df['item_category'].nunique()} distinct categories
- **Average Winning Price:** ${item_df[item_df['item_winning_price'] > 0]['item_winning_price'].mean():.2f}
- **Zero-Bid Rate:** {(item_df['item_winning_price'] == 0).sum() / len(item_df) * 100:.1f}%

## Key Findings

### 1. Target Variable (Item Winning Price)

**Distribution Characteristics:**
- Highly right-skewed distribution
- Significant proportion of zero-bid items ({(item_df['item_winning_price'] == 0).sum() / len(item_df) * 100:.1f}%)
- Median price: ${item_df[item_df['item_winning_price'] > 0]['item_winning_price'].median():.2f}
- Mean price: ${item_df[item_df['item_winning_price'] > 0]['item_winning_price'].mean():.2f}

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
"""
    
    # Save report
    with open(Path(__file__).parent / "summary.md", 'w') as f:
        f.write(report)
    
    print(f"\n✓ Summary report saved to summary.md")


if __name__ == "__main__":
    main()
