"""
Test feature engineering on enriched item data.

Uses mock data to avoid OOM issues when testing locally.
For production, use this pattern with real data in smaller batches.
"""
from src.features import engineer_all_enriched_features
from loguru import logger
import pandas as pd
import json

def create_mock_enriched_data(n_rows=100):
    """Create mock enriched item data for testing."""
    logger.info(f"Creating mock data with {n_rows} rows...")
    
    data = []
    for i in range(n_rows):
        # Sample enriched JSON columns (matching HuggingFace schema)
        row = {
            "item_id": 10000 + i,
            "auction_id": 100 + (i // 10),
            "item_current_bid": 25.50 + (i % 50) * 5,
            
            # enriched_item_brands (JSON list of brand objects)
            "enriched_item_brands": json.dumps([
                {"brandName": "Sony", "brandId": 1},
                {"brandName": "Samsung", "brandId": 2}
            ]) if i % 3 == 0 else json.dumps([]),
            
            # enriched_item_categories (JSON list of category objects)
            "enriched_item_categories": json.dumps([
                {"categoryId": 10, "categoryName": "Electronics", "categoryPath": "Home > Electronics"},
                {"categoryId": 11, "categoryName": "TV & Video", "categoryPath": "Home > Electronics > TV & Video"}
            ]) if i % 2 == 0 else json.dumps([{"categoryId": 20, "categoryName": "Furniture", "categoryPath": "Home > Furniture"}]),
            
            # enriched_item_items (JSON list with single item)
            "enriched_item_items": json.dumps([{
                "itemId": 10000 + i,
                "itemTitle": f"Item {i} - Vintage Collection",
                "itemDescription": f"This is a detailed description for item {i}. It's in excellent condition.",
                "itemCurrentBid": 25.50 + (i % 50) * 5,
                "itemStartBid": 10.00,
                "itemBidCount": i % 20
            }]),
            
            # enriched_item_attributes (JSON list of attribute objects)
            "enriched_item_attributes": json.dumps([
                {"attributeName": "Color", "attributeValue": ["Red", "Blue"][i % 2]},
                {"attributeName": "Condition", "attributeValue": "Excellent"},
                {"attributeName": "Year", "attributeValue": "2020"}
            ]) if i % 4 == 0 else json.dumps([]),
            
            # enriched_item_photosTaken (JSON list of photo objects)
            "enriched_item_photosTaken": json.dumps([
                {
                    "imagePath": f"images/item_{i}_1.jpg",
                    "description": "Front view",
                    "photoNumber": 1
                },
                {
                    "imagePath": f"images/item_{i}_2.jpg",
                    "description": "Side view",
                    "photoNumber": 2
                }
            ]) if i % 5 != 0 else json.dumps([])
        }
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    # Create mock data
    df = create_mock_enriched_data(n_rows=100)
    
    logger.info(f"Mock data created: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nSample row:")
    logger.info(df.iloc[0].to_dict())
    
    # Apply feature engineering
    logger.info("\n" + "="*70)
    logger.info("Starting feature engineering...")
    logger.info("="*70)
    
    df_with_features = engineer_all_enriched_features(df)
    
    logger.info("\n" + "="*70)
    logger.info("✅ Feature engineering complete!")
    logger.info("="*70)
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Original columns: {len(df.columns)}")
    logger.info(f"  After features: {len(df_with_features.columns)}")
    logger.info(f"  New features added: {len(df_with_features.columns) - len(df.columns)}")
    
    # Show sample of new columns
    new_cols = [col for col in df_with_features.columns if col not in df.columns]
    logger.info(f"\n📋 New features created ({len(new_cols)} total):")
    
    # Group by prefix for better readability
    from collections import defaultdict
    grouped = defaultdict(list)
    for col in sorted(new_cols):
        prefix = col.split('_')[0] if '_' in col else 'other'
        grouped[prefix].append(col)
    
    for prefix, cols in sorted(grouped.items()):
        logger.info(f"\n  {prefix.upper()} features ({len(cols)}):")
        for col in cols[:5]:  # Show first 5 of each group
            non_null = df_with_features[col].notna().sum()
            sample_val = df_with_features[col].dropna().iloc[0] if non_null > 0 else None
            logger.info(f"    • {col:40s} ({non_null:>3d}/{len(df)} non-null) = {sample_val}")
        if len(cols) > 5:
            logger.info(f"    ... and {len(cols) - 5} more")
    
    # Show basic stats
    logger.info(f"\n📈 Data shape: {df_with_features.shape}")
    logger.info(f"💾 Memory usage: {df_with_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save sample results
    import os
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/features_test.parquet"
    df_with_features.to_parquet(output_file)
    logger.info(f"\n💾 Saved test results to: {output_file}")
    logger.info(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Test completed successfully!")
    logger.info("="*70)
    logger.info("\nTo use with real data:")
    logger.info("  1. Load data in chunks: dataset = load_dataset(..., streaming=True)")
    logger.info("  2. Process in batches: for batch in dataset.iter(batch_size=1000)")
    logger.info("  3. Apply features: engineer_all_enriched_features(batch)")
    logger.info("  4. Save incrementally to avoid OOM")

if __name__ == "__main__":
    main()