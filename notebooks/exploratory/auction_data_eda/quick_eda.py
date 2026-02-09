"""
Quick Exploratory Data Analysis Script for Auction Data

This script performs a rapid analysis of the auction dataset and outputs
key statistics and engineered features.

Usage:
    python quick_eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load auction data from parquet file."""
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "external" / "auction_data.parquet"
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} auctions with {len(df.columns)} columns\n")
    return df


def create_temporal_features(df):
    """Create temporal features from datetime columns."""
    df = df.copy()
    
    # Parse datetimes (handle mixed timezones)
    df['auction_starts_dt'] = pd.to_datetime(df['auction_starts'], utc=True)
    df['auction_ends_dt'] = pd.to_datetime(df['auction_ends'], utc=True)
    df['auction_last_item_closes_dt'] = pd.to_datetime(df['auction_last_item_closes'], utc=True)
    
    # Calculate durations
    df['auction_duration_hours'] = (df['auction_ends_dt'] - df['auction_starts_dt']).dt.total_seconds() / 3600
    df['auction_extended_hours'] = (df['auction_last_item_closes_dt'] - df['auction_ends_dt']).dt.total_seconds() / 3600
    
    # Extract time components
    df['start_year'] = df['auction_starts_dt'].dt.year
    df['start_month'] = df['auction_starts_dt'].dt.month
    df['start_day_of_week'] = df['auction_starts_dt'].dt.dayofweek
    df['start_hour'] = df['auction_starts_dt'].dt.hour
    
    return df


def create_derived_metrics(df):
    """Create derived metrics and ratios."""
    df = df.copy()
    
    # Averages per item
    df['avg_price_per_item'] = df['auction_total_winning_price'] / df['auction_item_count'].replace(0, np.nan)
    df['avg_views_per_item'] = df['auction_total_viewed'] / df['auction_item_count'].replace(0, np.nan)
    df['avg_bids_per_item'] = df['auction_total_bid_count'] / df['auction_item_count'].replace(0, np.nan)
    df['avg_images_per_item'] = df['auction_total_images'] / df['auction_item_count'].replace(0, np.nan)
    
    # Engagement metrics
    df['bid_to_view_ratio'] = df['auction_total_bid_count'] / df['auction_total_viewed'].replace(0, np.nan)
    
    # Text features
    df['title_length'] = df['auction_title'].str.len()
    df['title_word_count'] = df['auction_title'].str.split().str.len()
    df['intro_length'] = df['auction_intro'].fillna('').str.len()
    df['has_partner_url'] = df['auction_partner_url'].notna().astype(int)
    
    return df


def extract_location(title):
    """Extract city from auction title."""
    import re
    match = re.search(r'^([^(]+)\(', title)
    if match:
        return match.group(1).strip()
    return None


def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS - AUCTION DATA")
    print("=" * 80)
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Total Auctions: {len(df):,}")
    print(f"   Date Range: {df['start_year'].min()} - {df['start_year'].max()}")
    print(f"   Total Columns: {len(df.columns)}")
    
    print(f"\n2. AUCTION CHARACTERISTICS")
    print(f"   Items per Auction:")
    print(f"     - Median: {df['auction_item_count'].median():.0f}")
    print(f"     - Mean: {df['auction_item_count'].mean():.1f}")
    print(f"     - Range: {df['auction_item_count'].min()} - {df['auction_item_count'].max()}")
    
    print(f"\n   Total Winning Price per Auction:")
    print(f"     - Median: ${df['auction_total_winning_price'].median():,.2f}")
    print(f"     - Mean: ${df['auction_total_winning_price'].mean():,.2f}")
    print(f"     - Range: ${df['auction_total_winning_price'].min():,.2f} - ${df['auction_total_winning_price'].max():,.2f}")
    
    print(f"\n3. ENGAGEMENT METRICS")
    print(f"   Views per Auction:")
    print(f"     - Median: {df['auction_total_viewed'].median():,.0f}")
    print(f"     - Mean: {df['auction_total_viewed'].mean():,.0f}")
    
    print(f"\n   Bids per Auction:")
    print(f"     - Median: {df['auction_total_bid_count'].median():,.0f}")
    print(f"     - Mean: {df['auction_total_bid_count'].mean():,.0f}")
    
    print(f"\n   Bid-to-View Ratio:")
    print(f"     - Median: {df['bid_to_view_ratio'].median():.4f} ({df['bid_to_view_ratio'].median()*100:.2f}%)")
    
    print(f"\n4. DERIVED METRICS")
    print(f"   Average Price per Item:")
    print(f"     - Median: ${df['avg_price_per_item'].median():,.2f}")
    print(f"     - Mean: ${df['avg_price_per_item'].mean():,.2f}")
    
    print(f"\n   Average Views per Item:")
    print(f"     - Median: {df['avg_views_per_item'].median():,.0f}")
    print(f"     - Mean: {df['avg_views_per_item'].mean():,.0f}")
    
    print(f"\n5. TEMPORAL PATTERNS")
    print(f"   Auction Duration:")
    print(f"     - Median: {df['auction_duration_hours'].median():.1f} hours ({df['auction_duration_hours'].median()/24:.1f} days)")
    print(f"     - Mean: {df['auction_duration_hours'].mean():.1f} hours")
    
    print(f"\n   Extended Bidding:")
    extended_count = df['auction_extended_bidding'].sum()
    extended_pct = (extended_count / len(df)) * 100
    print(f"     - Auctions with extended bidding: {extended_count:,} ({extended_pct:.1f}%)")
    
    print(f"\n6. GEOGRAPHIC DISTRIBUTION")
    df['location'] = df['auction_title'].apply(extract_location)
    top_locations = df['location'].value_counts().head(10)
    print(f"   Unique Locations: {df['location'].nunique()}")
    print(f"   Top 5 Locations:")
    for i, (location, count) in enumerate(top_locations.head(5).items(), 1):
        print(f"     {i}. {location}: {count:,} auctions")
    
    print(f"\n7. DATA QUALITY")
    missing = df[['auction_id', 'auction_title', 'auction_item_count', 
                   'auction_total_winning_price', 'auction_total_viewed']].isnull().sum().sum()
    print(f"   Missing values in key columns: {missing}")
    print(f"   ✓ Dataset quality: {'Excellent' if missing == 0 else 'Needs attention'}")
    
    print(f"\n8. FEATURE CORRELATIONS (with total winning price)")
    numeric_cols = ['auction_item_count', 'auction_total_viewed', 
                    'auction_total_bid_count', 'avg_views_per_item', 'auction_duration_hours']
    correlations = df[numeric_cols + ['auction_total_winning_price']].corr()['auction_total_winning_price'].drop('auction_total_winning_price').sort_values(ascending=False)
    for col, corr in correlations.items():
        print(f"   {col}: {corr:.3f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
    print("\nEngineered features have been created. See SUMMARY.md for detailed recommendations.")
    print("Full analysis available in: 01_auction_data_exploration.ipynb")
    print("=" * 80)


def save_processed_data(df):
    """Save processed data with engineered features."""
    output_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "auction_data_with_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns to save
    feature_columns = [
        'auction_id', 'auction_title', 'auction_starts', 'auction_ends',
        'auction_item_count', 'auction_total_viewed', 'auction_total_winning_price',
        'auction_total_bid_count', 'auction_total_images', 'auction_extended_bidding',
        'auction_duration_hours', 'auction_extended_hours',
        'start_year', 'start_month', 'start_day_of_week', 'start_hour',
        'avg_price_per_item', 'avg_views_per_item', 'avg_bids_per_item',
        'avg_images_per_item', 'bid_to_view_ratio', 'location',
        'title_length', 'title_word_count', 'intro_length', 'has_partner_url'
    ]
    
    df_processed = df[feature_columns].copy()
    df_processed.to_parquet(output_path, index=False)
    print(f"\n✓ Processed data saved to: {output_path}")
    print(f"  Rows: {len(df_processed):,}")
    print(f"  Columns: {len(df_processed.columns)}")


def main():
    """Main execution function."""
    # Load data
    df = load_data()
    
    # Create features
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating derived metrics...")
    df = create_derived_metrics(df)
    
    print("Analysis complete!\n")
    
    # Print summary
    print_summary_statistics(df)
    
    # Save processed data
    save_processed_data(df)


if __name__ == "__main__":
    main()
