# =============================================================================
# Integration Example: Combining Item and Enriched Item Data
# =============================================================================
"""
This script demonstrates how to combine data from the item scraper
and enriched item scraper for comprehensive item analysis.
"""

import pandas as pd
from pathlib import Path


def load_datasets():
    """Load item and enriched item datasets."""
    item_file = Path("data/processed/items/item_data.parquet")
    enriched_file = Path("data/processed/enriched_items/enriched_item_data.parquet")
    
    # Check if files exist
    if not item_file.exists():
        print(f"Item data not found at {item_file}")
        print("Run: make scrape-items")
        return None, None
    
    if not enriched_file.exists():
        print(f"Enriched item data not found at {enriched_file}")
        print("Run: make scrape-enriched")
        return None, None
    
    # Load data
    items_df = pd.read_parquet(item_file)
    enriched_df = pd.read_parquet(enriched_file)
    
    print(f"✓ Loaded {len(items_df):,} items")
    print(f"✓ Loaded {len(enriched_df):,} enriched items")
    
    return items_df, enriched_df


def combine_datasets(items_df, enriched_df):
    """Combine item and enriched item datasets."""
    # Join on item_id and auction_id
    combined_df = items_df.merge(
        enriched_df,
        on=["item_id", "auction_id"],
        how="left",  # Keep all items, even without enriched data
        suffixes=("", "_enriched")
    )
    
    print(f"\n✓ Combined dataset: {len(combined_df):,} rows")
    print(f"✓ Items with enriched data: {combined_df['enriched_item_generatedDescription_title'].notna().sum():,}")
    
    return combined_df


def analyze_combined_data(combined_df):
    """Analyze the combined dataset."""
    print("\n" + "=" * 60)
    print("Combined Dataset Analysis")
    print("=" * 60)
    
    # Basic stats
    print("\nDataset Size:")
    print(f"  Total items: {len(combined_df):,}")
    print(f"  Items with enriched data: {combined_df['enriched_item_generatedDescription_title'].notna().sum():,}")
    print(f"  Coverage: {combined_df['enriched_item_generatedDescription_title'].notna().mean()*100:.1f}%")
    
    # Enriched fields coverage
    print("\nEnriched Fields Coverage:")
    enriched_cols = [col for col in combined_df.columns if col.startswith("enriched_item_")]
    for col in enriched_cols[:5]:  # Show first 5
        coverage = combined_df[col].notna().mean() * 100
        print(f"  {col}: {coverage:.1f}%")
    
    # Sample enriched data
    print("\nSample Enriched Items:")
    enriched_items = combined_df[combined_df["enriched_item_generatedDescription_title"].notna()]
    if len(enriched_items) > 0:
        sample = enriched_items[["item_id", "item_title", "enriched_item_generatedDescription_brand", 
                                  "enriched_item_generatedDescription_condition"]].head(5)
        print(sample.to_string())
    
    # Brand analysis
    if "enriched_item_generatedDescription_brand" in combined_df.columns:
        print("\nTop Brands:")
        top_brands = combined_df["enriched_item_generatedDescription_brand"].value_counts().head(10)
        for brand, count in top_brands.items():
            if pd.notna(brand):
                print(f"  {brand}: {count:,} items")


def demonstrate_use_cases(combined_df):
    """Demonstrate practical use cases."""
    print("\n" + "=" * 60)
    print("Use Case Examples")
    print("=" * 60)
    
    # Use case 1: Filter by brand
    print("\n1. Filter items by brand:")
    if "enriched_item_generatedDescription_brand" in combined_df.columns:
        branded_items = combined_df[combined_df["enriched_item_generatedDescription_brand"].notna()]
        print(f"   Items with known brands: {len(branded_items):,}")
    
    # Use case 2: Filter by condition
    print("\n2. Items in excellent condition:")
    if "enriched_item_generatedDescription_condition" in combined_df.columns:
        excellent = combined_df[
            combined_df["enriched_item_generatedDescription_condition"].str.contains("Excellent", na=False)
        ]
        print(f"   Items in excellent condition: {len(excellent):,}")
    
    # Use case 3: Working status analysis
    print("\n3. Working status analysis:")
    if "enriched_item_generatedDescription_working" in combined_df.columns:
        working_items = combined_df["enriched_item_generatedDescription_working"].value_counts()
        for status, count in working_items.items():
            print(f"   {status}: {count:,} items")
    
    # Use case 4: Price by condition
    print("\n4. Average price by condition:")
    if "enriched_item_generatedDescription_condition" in combined_df.columns and "item_winning_price" in combined_df.columns:
        avg_price = combined_df.groupby("enriched_item_generatedDescription_condition")["item_winning_price"].mean()
        for condition, price in avg_price.head(5).items():
            if pd.notna(condition):
                print(f"   {condition}: ${price:.2f}")


def main():
    """Main function."""
    print("=" * 60)
    print("Item + Enriched Item Data Integration Example")
    print("=" * 60)
    print()
    
    # Load datasets
    items_df, enriched_df = load_datasets()
    
    if items_df is None or enriched_df is None:
        return
    
    # Combine datasets
    combined_df = combine_datasets(items_df, enriched_df)
    
    # Analyze combined data
    analyze_combined_data(combined_df)
    
    # Demonstrate use cases
    demonstrate_use_cases(combined_df)
    
    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
