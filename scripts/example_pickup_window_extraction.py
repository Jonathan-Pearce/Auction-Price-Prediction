#!/usr/bin/env python3
"""
Example script demonstrating pickup window feature extraction.

This script shows how to use the new pickup window feature engineering
functionality with real auction data from the Hugging Face dataset.
"""

from datasets import load_dataset
import pandas as pd
from src.features import engineer_auction_features, extract_pickup_windows
from loguru import logger


def main():
    """
    Demonstrate pickup window feature extraction on real auction data.
    """
    logger.info("Loading sample auction data from Hugging Face...")
    
    # Load a small sample from the dataset
    ds = load_dataset('jpearce610/auction_data', split='train', streaming=True)
    
    # Get first 20 examples
    examples = []
    for i, example in enumerate(ds):
        if i >= 20:
            break
        examples.append(example)
    
    # Create DataFrame
    df = pd.DataFrame(examples)
    logger.info(f"Loaded {len(df)} auctions")
    
    # Extract pickup window features
    logger.info("\nExtracting pickup window features...")
    enriched_df = engineer_auction_features(df)
    
    # Display the extracted features
    pickup_features = [
        'num_pickup_windows',
        'total_pickup_hours', 
        'first_pickup_start_hour',
        'last_pickup_end_hour',
        'pickup_day_of_week',
        'has_category_windows'
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("PICKUP WINDOW FEATURES SUMMARY")
    logger.info("=" * 80)
    
    # Show statistics
    logger.info("\nFeature Statistics:")
    logger.info(f"\n{enriched_df[pickup_features].describe()}")
    
    # Show a few examples
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE AUCTIONS WITH PICKUP WINDOW FEATURES")
    logger.info("=" * 80)
    
    for idx in range(min(5, len(enriched_df))):
        logger.info(f"\n--- Auction {idx + 1} (ID: {enriched_df.loc[idx, 'auction_id']}) ---")
        
        # Show features
        for feature in pickup_features:
            value = enriched_df.loc[idx, feature]
            logger.info(f"  {feature:25s}: {value}")
        
        # Show snippet of original HTML
        if enriched_df.loc[idx, 'auction_removal_info']:
            html = enriched_df.loc[idx, 'auction_removal_info']
            # Extract just the pickup line for display
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            pickup_line = next((line for line in lines if 'Pickup:' in line), 'N/A')
            logger.info(f"  Original text: {pickup_line[:80]}...")
    
    # Show value counts for categorical features
    logger.info("\n" + "=" * 80)
    logger.info("PICKUP DAY OF WEEK DISTRIBUTION")
    logger.info("=" * 80)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = enriched_df['pickup_day_of_week'].value_counts().sort_index()
    for day_num, count in day_counts.items():
        if pd.notna(day_num):
            day_name = day_names[int(day_num)]
            logger.info(f"  {day_name:10s}: {count:2d} auctions")
    
    logger.info("\n" + "=" * 80)
    logger.info("CATEGORY WINDOWS BREAKDOWN")
    logger.info("=" * 80)
    category_counts = enriched_df['has_category_windows'].value_counts()
    logger.info(f"  Auctions with category-based windows: {category_counts.get(True, 0)}")
    logger.info(f"  Auctions with single time windows:    {category_counts.get(False, 0)}")
    
    # Demonstrate using extract_pickup_windows directly
    logger.info("\n" + "=" * 80)
    logger.info("DIRECT USAGE EXAMPLE")
    logger.info("=" * 80)
    
    example_html = """
    <div class="bold">
        <strong>Pickup: Saturday, March 21 EDT, 2:00PM - 5:00PM</strong>
    </div>
    """
    
    result = extract_pickup_windows(example_html)
    logger.info("\nExtract pickup windows from HTML:")
    logger.info(f"Input HTML: {example_html.strip()}")
    logger.info(f"\nExtracted features:")
    for key, value in result.items():
        logger.info(f"  {key:25s}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
