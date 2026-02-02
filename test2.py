from datasets import load_dataset
from src.features import engineer_all_enriched_features
import pandas as pd
import time
import os
from loguru import logger

print("="*70)
print("Feature Engineering - Batch Processing")
print("="*70)

# Stream data in batches
print("\n📥 Loading dataset in streaming mode...")
dataset = load_dataset("jpearce610/enriched_item_data", split="train", streaming=True)
print("✅ Dataset loaded successfully")

batch_size = 1000
all_features = []

# Create output directory
os.makedirs("data/processed", exist_ok=True)
print(f"\n📁 Output directory: data/processed/")
print(f"📦 Batch size: {batch_size:,} rows")
print(f"⚙️  Starting batch processing...\n")

start_time = time.time()
total_rows = 0

for i, batch in enumerate(dataset.iter(batch_size)):
    batch_start = time.time()
    
    print(f"\n{'─'*70}")
    print(f"Batch {i+1}")
    print(f"{'─'*70}")
    
    # Convert to DataFrame
    print(f"  📊 Converting batch to DataFrame...")
    df_batch = pd.DataFrame(batch)
    batch_rows = len(df_batch)
    total_rows += batch_rows
    print(f"  ✓ Loaded {batch_rows:,} rows")
    
    # Apply feature engineering
    print(f"  🔧 Engineering features...")
    df_features = engineer_all_enriched_features(df_batch)
    new_cols = len(df_features.columns) - len(df_batch.columns)
    print(f"  ✓ Created {new_cols} new features ({len(df_features.columns)} total columns)")
    
    # Save batch
    output_file = f"data/processed/features_batch_{i:04d}.parquet"
    print(f"  💾 Saving to {output_file}...")
    df_features.to_parquet(output_file)
    file_size = os.path.getsize(output_file) / 1024  # KB
    print(f"  ✓ Saved ({file_size:.1f} KB)")
    
    # Timing and memory stats
    batch_time = time.time() - batch_start
    elapsed = time.time() - start_time
    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
    
    print(f"\n  ⏱️  Batch time: {batch_time:.2f}s")
    print(f"  📈 Total processed: {total_rows:,} rows")
    print(f"  🚀 Speed: {rows_per_sec:.1f} rows/sec")
    print(f"  ⏰ Total elapsed: {elapsed:.1f}s")
    
    # Memory usage
    mem_mb = df_features.memory_usage(deep=True).sum() / 1024**2
    print(f"  💾 Memory (batch): {mem_mb:.2f} MB")

print(f"\n{'='*70}")
print(f"✅ Processing Complete!")
print(f"{'='*70}")
print(f"  Total batches: {i+1}")
print(f"  Total rows: {total_rows:,}")
print(f"  Total time: {time.time() - start_time:.1f}s")
print(f"  Average speed: {total_rows / (time.time() - start_time):.1f} rows/sec")
print(f"{'='*70}")