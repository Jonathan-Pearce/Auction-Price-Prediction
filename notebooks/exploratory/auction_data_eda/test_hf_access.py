"""
Test script to check if we can access the Hugging Face auction dataset.
"""

from datasets import load_dataset

print("Attempting to load auction_data from Hugging Face...")
print("Dataset: jpearce610/auction_data")

try:
    dataset = load_dataset("jpearce610/auction_data")
    print(f"\n✓ Successfully loaded dataset!")
    print(f"\nDataset info:")
    print(f"  Available splits: {list(dataset.keys())}")
    
    # Get the first split
    first_split = list(dataset.keys())[0]
    print(f"\n  First split: {first_split}")
    print(f"  Number of rows: {len(dataset[first_split])}")
    print(f"  Columns: {dataset[first_split].column_names}")
    
    # Show first few rows
    print(f"\n  First 3 rows:")
    for i in range(min(3, len(dataset[first_split]))):
        print(f"    Row {i}: {dataset[first_split][i]}")
    
except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    print(f"\nThis may be due to firewall restrictions.")
    import sys
    sys.exit(1)
