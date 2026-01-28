#!/usr/bin/env python3
"""
Example script demonstrating text feature engineering for auction items.

This script shows how to use the various text feature extraction methods
and their performance characteristics.
"""

import time
import pandas as pd
from src.features import extract_text_features, extract_handcrafted_text_features


def main():
    """Demonstrate text feature engineering methods."""
    
    # Sample auction item descriptions
    sample_descriptions = [
        "Vintage leather handbag in excellent condition. Genuine Italian leather with gold hardware.",
        "Modern glass vase, hand-blown art piece. Beautiful blue and green swirls.",
        "Antique wooden chair with carved details. Shows some wear but structurally sound.",
        "Rare collectible coin set from 1950s. Complete set in original packaging.",
        "Brand new kitchen appliance, never used. Still in original box with warranty.",
        "Authentic signed baseball from famous player. Certificate of authenticity included.",
        "Classic vinyl records collection. Mint condition, rare pressings.",
        "Quality silver jewelry set. Sterling silver with genuine gemstones.",
        "Oil painting by local artist. Landscape scene, framed and ready to hang.",
        "Crystal chandelier with brass fittings. All pieces intact, needs cleaning.",
    ]
    
    print("=" * 70)
    print("Text Feature Engineering for Auction Items")
    print("=" * 70)
    print()
    
    # Method 1: Hand-crafted features (fastest)
    print("1. Hand-crafted Features (Fastest - recommended for tabular models)")
    print("-" * 70)
    start = time.time()
    features, _ = extract_text_features(sample_descriptions, method='handcrafted')
    elapsed = (time.time() - start) * 1000
    print(f"   Time: {elapsed:.2f}ms for {len(sample_descriptions)} items")
    print(f"   Output shape: {features.shape}")
    print(f"   Features: {', '.join(features.columns[:5])}...")
    print(f"   Avg words per item: {features['text_word_count'].mean():.1f}")
    print()
    
    # Method 2: TF-IDF (fast, traditional)
    print("2. TF-IDF Features (Fast - good for baseline models)")
    print("-" * 70)
    start = time.time()
    features, vectorizer = extract_text_features(sample_descriptions, method='tfidf', max_features=100)
    elapsed = (time.time() - start) * 1000
    print(f"   Time: {elapsed:.2f}ms for {len(sample_descriptions)} items")
    print(f"   Output shape: {features.shape}")
    print(f"   Top features: {', '.join(vectorizer.get_feature_names_out()[:5])}...")
    print()
    
    # Method 3: Bag of Words
    print("3. Bag of Words (Simplest baseline)")
    print("-" * 70)
    start = time.time()
    features, vectorizer = extract_text_features(sample_descriptions, method='bow', max_features=100)
    elapsed = (time.time() - start) * 1000
    print(f"   Time: {elapsed:.2f}ms for {len(sample_descriptions)} items")
    print(f"   Output shape: {features.shape}")
    print()
    
    # Method 4: Sentence embeddings (requires sentence-transformers)
    print("4. Sentence Embeddings (RECOMMENDED for production)")
    print("-" * 70)
    try:
        start = time.time()
        features, model = extract_text_features(sample_descriptions, method='embeddings')
        elapsed = (time.time() - start) * 1000
        print(f"   Time: {elapsed:.2f}ms for {len(sample_descriptions)} items")
        print(f"   Output shape: {features.shape}")
        print(f"   Avg inference per item: {elapsed/len(sample_descriptions):.2f}ms")
        print(f"   Model: all-MiniLM-L6-v2 (384 dimensions)")
        print()
    except ImportError:
        print("   ⚠️  sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        print()
    
    # Comparison summary
    print("=" * 70)
    print("Summary & Recommendations")
    print("=" * 70)
    print()
    print("For Web App Deployment:")
    print("  1. Use handcrafted features → add to tabular model (fastest)")
    print("  2. Use sentence embeddings → dedicated text model (best quality)")
    print("  3. Consider TF-IDF if you need <5ms latency")
    print()
    print("Expected Inference Times:")
    print("  - Handcrafted: <1ms per item")
    print("  - TF-IDF: <1ms per item")
    print("  - Sentence embeddings: ~15ms per item (batch: ~1ms per item)")
    print()
    print("Expected Performance (MAE):")
    print("  - Handcrafted only: ~$25")
    print("  - TF-IDF: ~$20")
    print("  - Sentence embeddings: ~$12-15")
    print()


if __name__ == "__main__":
    main()
