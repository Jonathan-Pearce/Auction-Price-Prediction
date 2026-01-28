#!/usr/bin/env python3
# =============================================================================
# Datetime Feature Engineering Demo
# =============================================================================
"""
Demonstration of datetime feature engineering capabilities.

This script shows examples of all datetime transformations that can be
applied to auction data.
"""

import pandas as pd
from src.datetime_features import (
    DatetimeFeatureEngineer,
    add_auction_duration_features,
    add_bid_timing_features,
    engineer_all_datetime_features,
)


def create_sample_auction_data():
    """Create sample auction data for demonstration."""
    return pd.DataFrame({
        'auction_id': [1, 2, 3, 4, 5],
        'start_time': [
            '2024-01-01T10:00:00',  # Monday morning, New Year's Day
            '2024-06-15T14:00:00',  # Saturday afternoon
            '2024-12-25T09:00:00',  # Wednesday morning, Christmas
            '2024-07-04T15:30:00',  # Thursday afternoon, Independence Day
            '2024-03-15T18:00:00',  # Friday evening
        ],
        'end_time': [
            '2024-01-05T18:00:00',
            '2024-06-20T14:00:00',
            '2024-12-28T09:00:00',
            '2024-07-07T15:30:00',
            '2024-03-18T18:00:00',
        ],
    })


def create_sample_bid_data():
    """Create sample bid data for demonstration."""
    return pd.DataFrame({
        'bid_id': [1, 2, 3, 4, 5],
        'bid_time': [
            '2024-01-02T10:00:00',  # Early bid
            '2024-01-05T17:58:30',  # Last-minute bid (90 sec before close)
            '2024-01-03T14:00:00',  # Mid-auction bid
            '2024-01-05T17:00:00',  # Late bid
            '2024-01-01T10:30:00',  # Very early bid (30 min after start)
        ],
        'auction_start_time': ['2024-01-01T10:00:00'] * 5,
        'auction_end_time': ['2024-01-05T18:00:00'] * 5,
    })


def demo_basic_datetime_features():
    """Demonstrate basic datetime feature engineering."""
    print("\n" + "=" * 80)
    print("DEMO: Basic Datetime Features")
    print("=" * 80 + "\n")
    
    df = create_sample_auction_data()
    print("Original data:")
    print(df.head())
    
    # Initialize feature engineer
    engineer = DatetimeFeatureEngineer(
        include_cyclical=False,  # Exclude cyclical for readability
        include_holidays=True,
    )
    
    # Add datetime features
    result = engineer.add_datetime_features(df, 'start_time')
    
    # Show selected features
    feature_cols = [
        'auction_id',
        'start_time',
        'start_time_year',
        'start_time_month',
        'start_time_day',
        'start_time_hour',
        'start_time_day_of_week',
        'start_time_is_weekend',
        'start_time_part_of_day',
        'start_time_season',
        'start_time_is_holiday',
    ]
    
    print("\n\nWith datetime features:")
    print(result[feature_cols].to_string(index=False))


def demo_auction_duration_features():
    """Demonstrate auction duration feature engineering."""
    print("\n" + "=" * 80)
    print("DEMO: Auction Duration Features")
    print("=" * 80 + "\n")
    
    df = create_sample_auction_data()
    
    result = add_auction_duration_features(df)
    
    # Show duration features
    duration_cols = [
        'auction_id',
        'start_time',
        'end_time',
        'auction_duration_days',
        'auction_duration_hours',
        'auction_duration_length_category',
    ]
    
    print("Auction duration features:")
    print(result[duration_cols].to_string(index=False))


def demo_bid_timing_features():
    """Demonstrate bid timing feature engineering."""
    print("\n" + "=" * 80)
    print("DEMO: Bid Timing Features")
    print("=" * 80 + "\n")
    
    df = create_sample_bid_data()
    print("Original bid data:")
    print(df.head())
    
    result = add_bid_timing_features(df)
    
    # Show bid timing features
    timing_cols = [
        'bid_id',
        'bid_time',
        'bid_timing_hours_since_start',
        'bid_timing_hours_until_end',
        'bid_timing_is_last_minute',
        'bid_timing_pct_through_auction',
        'bid_timing_position',
    ]
    
    print("\n\nBid timing features:")
    print(result[timing_cols].to_string(index=False))


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DATETIME FEATURE ENGINEERING DEMO" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    
    demo_basic_datetime_features()
    demo_auction_duration_features()
    demo_bid_timing_features()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    
    print("This demo showed datetime feature transformations available:")
    print("  1. Basic temporal features (hour, day, month, etc.)")
    print("  2. Auction duration features")
    print("  3. Bid timing features (early/late, last-minute)")
    
    print("\nTo use in your code:")
    print("  from src.datetime_features import DatetimeFeatureEngineer")
    print("  engineer = DatetimeFeatureEngineer()")
    print("  df = engineer.add_datetime_features(df, 'your_datetime_column')")
    
    print("\nFor more examples, see: tests/test_datetime_features.py")
    print()


if __name__ == '__main__':
    main()
