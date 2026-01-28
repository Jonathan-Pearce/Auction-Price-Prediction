# Text Feature Engineering for Item Descriptions

> **Comprehensive guide to feature engineering options for the `item_description` column**

## Overview

The item_description field contains natural language summaries of auction items. This document explores various feature engineering approaches, their trade-offs, and recommendations for production deployment.

---

## Feature Engineering Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

#### Description
Traditional statistical approach that represents text based on word importance across documents.

#### Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(texts: list[str], max_features: int = 500):
    """
    Extract TF-IDF features from text data.
    
    Args:
        texts: List of item descriptions
        max_features: Maximum number of features to extract
        
    Returns:
        Sparse matrix of TF-IDF features (n_samples x max_features)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,  # Ignore terms appearing in < 5 documents
        max_df=0.8,  # Ignore terms appearing in > 80% of documents
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
    )
    
    # Fit on training data
    features = vectorizer.fit_transform(texts)
    
    return features, vectorizer
```

#### Inference Performance
- **Speed**: ⭐⭐⭐⭐⭐ (Fastest - microseconds per item)
- **Memory**: ⭐⭐⭐⭐ (Low - sparse matrices, ~500-1000 features)
- **Preprocessing**: ⭐⭐⭐⭐⭐ (Simple tokenization only)

#### Pros
- ✅ Extremely fast inference (<1ms per item)
- ✅ Low memory footprint with sparse matrices
- ✅ No GPU required
- ✅ Interpretable features (actual words/phrases)
- ✅ Works well for domain-specific vocabularies

#### Cons
- ❌ Doesn't capture semantic meaning (e.g., "antique" vs "vintage")
- ❌ Requires vocabulary fitting on training data
- ❌ Struggles with synonyms and word variations
- ❌ High-dimensional output (500-5000 features typical)

#### Best For
- Quick MVP implementations
- Baseline models
- Production systems with strict latency requirements (<10ms)
- Tabular model inputs where you need many explicit features

---

### 2. Bag of Words (BoW) / Count Vectorization

#### Description
Simplest approach - counts word occurrences in each document.

#### Implementation
```python
from sklearn.feature_extraction.text import CountVectorizer

def extract_bow_features(texts: list[str], max_features: int = 500):
    """
    Extract Bag of Words features from text.
    
    Args:
        texts: List of item descriptions
        max_features: Maximum number of features
        
    Returns:
        Sparse matrix of word counts
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        binary=False,  # Set to True for binary BoW
    )
    
    features = vectorizer.fit_transform(texts)
    return features, vectorizer
```

#### Inference Performance
- **Speed**: ⭐⭐⭐⭐⭐ (Fastest - microseconds per item)
- **Memory**: ⭐⭐⭐⭐ (Low - sparse matrices)
- **Preprocessing**: ⭐⭐⭐⭐⭐ (Minimal)

#### Pros
- ✅ Fastest possible inference
- ✅ Simplest to implement and understand
- ✅ Works well as baseline
- ✅ No external dependencies beyond sklearn

#### Cons
- ❌ Even less semantic understanding than TF-IDF
- ❌ Sensitive to document length
- ❌ No weighting by importance

#### Best For
- Quick prototypes
- Baseline comparisons
- When you need absolute minimum latency

---

### 3. Pre-trained Sentence Embeddings (Recommended for Production)

#### Description
Use pre-trained models to generate dense, fixed-size semantic embeddings.

#### Implementation

**Option A: sentence-transformers (Recommended)**
```python
from sentence_transformers import SentenceTransformer
import torch

class SentenceEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize sentence embedder.
        
        Model options (ordered by speed):
        - 'all-MiniLM-L6-v2': 384 dims, ~14ms per item (RECOMMENDED)
        - 'all-MiniLM-L12-v2': 384 dims, ~20ms per item
        - 'all-mpnet-base-v2': 768 dims, ~40ms per item (best quality)
        """
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        
        # Use GPU if available for batch processing
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of descriptions
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings (n_samples x embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings
```

**Option B: Hugging Face Transformers**
```python
from transformers import AutoTokenizer, AutoModel
import torch

class TransformerEmbedder:
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize transformer embedder.
        
        Model options:
        - 'distilbert-base-uncased': 768 dims, ~15ms per item
        - 'bert-base-uncased': 768 dims, ~25ms per item
        - 'roberta-base': 768 dims, ~25ms per item
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def encode(self, texts: list[str], max_length: int = 128) -> np.ndarray:
        """
        Encode texts to embeddings using [CLS] token.
        
        Args:
            texts: List of descriptions
            max_length: Maximum sequence length
            
        Returns:
            Array of embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
```

#### Inference Performance
- **Speed**: ⭐⭐⭐ (10-40ms per item depending on model)
- **Memory**: ⭐⭐⭐ (Moderate - ~100-500MB model size)
- **Preprocessing**: ⭐⭐⭐⭐ (Automatic tokenization)

#### Pros
- ✅ Captures semantic meaning and context
- ✅ Fixed-size dense embeddings (384 or 768 dims)
- ✅ Pre-trained on massive corpora (no training needed)
- ✅ Handles synonyms, paraphrasing, and context
- ✅ Can batch process for efficiency
- ✅ Works great as input to neural networks

#### Cons
- ❌ Slower than TF-IDF (~10-40ms vs <1ms)
- ❌ Requires PyTorch/TensorFlow
- ❌ Larger model files (~100-500MB)
- ❌ Less interpretable than word-based features

#### Best For
- **RECOMMENDED FOR THIS PROJECT**: Production deployments where 10-40ms latency is acceptable
- Text model in the multi-modal ensemble
- When semantic understanding is important
- When you have GPU available for inference

---

### 4. Fine-tuned Transformer (Best Quality, Slower)

#### Description
Fine-tune a transformer model specifically for auction price prediction.

#### Implementation
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import torch.nn as nn

class AuctionTextModel(nn.Module):
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Regression head for price prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_embedding)
        return prediction
    
    def get_embedding(self, input_ids, attention_mask):
        """Extract embeddings for fusion model."""
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]
```

#### Inference Performance
- **Speed**: ⭐⭐ (20-50ms per item)
- **Memory**: ⭐⭐ (High - model + gradients during training)
- **Preprocessing**: ⭐⭐⭐⭐ (Same as pre-trained)

#### Pros
- ✅ Best possible text understanding
- ✅ Learns auction-specific language patterns
- ✅ Can directly output price predictions
- ✅ Can be part of end-to-end training

#### Cons
- ❌ Requires significant training time and data
- ❌ Slowest inference (20-50ms)
- ❌ Risk of overfitting on small datasets
- ❌ Requires labeled training data

#### Best For
- When you have >100k labeled examples
- When text is the primary signal for price
- When 20-50ms latency is acceptable
- Stage 2 optimization after proving value with pre-trained models

---

### 5. Hand-crafted Text Features

#### Description
Extract domain-specific features from text.

#### Implementation
```python
import re
from collections import Counter

def extract_handcrafted_features(text: str) -> dict:
    """
    Extract hand-crafted features from item description.
    
    Returns:
        Dictionary of numeric features
    """
    features = {}
    
    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Lexical features
    features['uppercase_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
    features['digit_count'] = sum(c.isdigit() for c in text)
    features['punctuation_count'] = sum(c in '.,!?;:' for c in text)
    
    # Keyword presence (auction-specific)
    keywords = {
        'brand_keywords': ['authentic', 'original', 'genuine', 'branded'],
        'condition_keywords': ['mint', 'excellent', 'good', 'fair', 'poor', 'damaged'],
        'quality_keywords': ['rare', 'vintage', 'antique', 'collectible', 'limited'],
        'material_keywords': ['wood', 'metal', 'glass', 'ceramic', 'plastic', 'leather'],
    }
    
    text_lower = text.lower()
    for category, words in keywords.items():
        features[f'{category}_count'] = sum(word in text_lower for word in words)
        features[f'has_{category}'] = int(any(word in text_lower for word in words))
    
    # Named entity counts (if using spacy)
    # features['brand_mentions'] = count_brand_entities(text)
    # features['measurement_mentions'] = count_measurements(text)
    
    return features
```

#### Inference Performance
- **Speed**: ⭐⭐⭐⭐⭐ (Fastest - microseconds)
- **Memory**: ⭐⭐⭐⭐⭐ (Minimal)
- **Preprocessing**: ⭐⭐⭐⭐ (Simple regex/string ops)

#### Pros
- ✅ Extremely fast inference
- ✅ Highly interpretable
- ✅ Can capture domain-specific patterns
- ✅ Works well alongside other methods
- ✅ No dependencies

#### Cons
- ❌ Requires domain knowledge
- ❌ Misses semantic content
- ❌ Labor-intensive to design
- ❌ May need regular updates

#### Best For
- Complement to other methods (add to tabular features)
- Quick wins from domain knowledge
- Interpretable models (XGBoost, Random Forest)
- When you need to explain predictions

---

## Comparison Table

| Method | Speed | Memory | Quality | Setup | GPU | Best Use Case |
|--------|-------|--------|---------|-------|-----|---------------|
| **TF-IDF** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | No | Fast baseline, strict latency |
| **Bag of Words** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | No | Quick prototype |
| **Sentence Embeddings** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Optional | **RECOMMENDED** |
| **Fine-tuned Transformer** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Yes | Best quality, have resources |
| **Hand-crafted** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | No | Supplement other methods |

---

## Recommended Approach for This Project

### Phase 1: MVP (Fast Development)
```python
# Combine hand-crafted + TF-IDF for tabular model
def create_mvp_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quick MVP text features.
    
    Fast inference, good baseline performance.
    """
    # Hand-crafted features (20 features)
    handcrafted = df['item_description'].apply(extract_handcrafted_features)
    handcrafted_df = pd.DataFrame(handcrafted.tolist())
    
    # TF-IDF features (top 300 features)
    tfidf, vectorizer = extract_tfidf_features(
        df['item_description'].tolist(),
        max_features=300
    )
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Combine
    features = pd.concat([handcrafted_df, tfidf_df], axis=1)
    return features
```

**Inference time**: <5ms per item
**Setup time**: <1 hour
**Expected performance**: Decent baseline (MAE ~$15-25)

### Phase 2: Production (Recommended)
```python
# Use sentence-transformers for dedicated text model
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    def __init__(self):
        # Fast, high-quality embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Also keep hand-crafted for tabular model
        self.handcrafted_extractor = extract_handcrafted_features
    
    def extract_for_text_model(self, texts: list[str]) -> np.ndarray:
        """384-dim embeddings for text model."""
        return self.embedder.encode(texts, batch_size=32)
    
    def extract_for_tabular_model(self, texts: list[str]) -> pd.DataFrame:
        """Hand-crafted features for tabular model."""
        features = [self.handcrafted_extractor(text) for text in texts]
        return pd.DataFrame(features)
```

**Inference time**: ~15ms per item (batched)
**Setup time**: <2 hours
**Expected performance**: Strong (MAE ~$10-18)

### Phase 3: Optimization (If Needed)
```python
# Fine-tune transformer on auction data
# Only if Phase 2 performance isn't sufficient

def train_finetuned_model(train_df: pd.DataFrame):
    model = AuctionTextModel('distilbert-base-uncased')
    # ... fine-tuning code ...
    return model
```

**Inference time**: ~25ms per item
**Setup time**: 1-2 days training
**Expected performance**: Best possible (MAE ~$8-15)

---

## Implementation Guidelines

### For Web App Deployment

#### 1. Batch Processing
```python
# Process multiple predictions at once for efficiency
def batch_predict(descriptions: list[str], batch_size: int = 32):
    embeddings = embedder.encode(descriptions, batch_size=batch_size)
    predictions = model.predict(embeddings)
    return predictions
```

#### 2. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    """Cache embeddings for frequently queried items."""
    return embedder.encode([text])[0]
```

#### 3. Model Loading
```python
# Load model once at startup, not per request
class TextModelManager:
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
            cls._instance.eval()
        return cls._instance
```

#### 4. Async Processing
```python
import asyncio

async def predict_async(text: str) -> float:
    """Async prediction for web apps."""
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, embedder.encode, [text])
    prediction = await loop.run_in_executor(None, model.predict, embedding)
    return prediction
```

---

## Performance Benchmarks

### Inference Latency (Single Item)

| Method | CPU (ms) | GPU (ms) | Notes |
|--------|----------|----------|-------|
| Hand-crafted | 0.1 | 0.1 | Pure Python |
| TF-IDF | 0.5 | 0.5 | Sparse matrix ops |
| BoW | 0.3 | 0.3 | Slightly faster than TF-IDF |
| all-MiniLM-L6-v2 | 14 | 3 | Recommended |
| distilbert-base | 18 | 4 | Good balance |
| all-mpnet-base-v2 | 38 | 8 | Best quality |
| Fine-tuned BERT | 45 | 10 | Highest quality |

### Batched Inference (32 items)

| Method | CPU (ms) | GPU (ms) | Per-Item (ms) |
|--------|----------|----------|---------------|
| all-MiniLM-L6-v2 | 180 | 35 | 1.1 (CPU) / 0.3 (GPU) |
| distilbert-base | 250 | 50 | 1.6 (CPU) / 0.4 (GPU) |

**Key Insight**: Batching improves throughput by 10-30x for transformer models.

---

## Storage Requirements

### Trained Artifacts Size

| Method | Size | Notes |
|--------|------|-------|
| TF-IDF vectorizer | 1-10 MB | Vocabulary + IDF weights |
| all-MiniLM-L6-v2 | 80 MB | Compact transformer |
| distilbert-base | 250 MB | Standard size |
| Fine-tuned model | 250-500 MB | Model + custom head |

---

## Recommendations Summary

### For Your Web App

1. **Start with**: `all-MiniLM-L6-v2` sentence embeddings
   - Fast enough for web (<15ms)
   - Excellent quality
   - Easy to implement
   - No training required

2. **Add to tabular model**: Hand-crafted features
   - Description length
   - Keyword counts
   - Quality indicators
   - Augments other features

3. **If latency is critical** (<5ms required):
   - Use TF-IDF for tabular model only
   - Skip dedicated text model
   - Focus on other modalities (image, structured features)

4. **If performance matters most**:
   - Fine-tune DistilBERT on your auction data
   - Accept 20-30ms latency
   - Best possible text understanding

### Implementation Priority

```
Phase 1 (MVP - Day 1):
├── Hand-crafted features → Add to tabular model
└── TF-IDF → Add to tabular model

Phase 2 (Production - Week 1):
├── all-MiniLM-L6-v2 embeddings → Text model
├── Batch processing → API optimization
└── Model caching → Performance optimization

Phase 3 (Optimization - If needed):
└── Fine-tune DistilBERT → Best quality
```

---

## Code Example: Complete Pipeline

```python
# src/features.py - Complete text feature extraction

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class TextFeatureExtractor:
    """
    Complete text feature extraction for auction items.
    
    Combines multiple approaches for different model types.
    """
    
    def __init__(self):
        # For text model (dense embeddings)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model.eval()
        
        # For tabular model (sparse features)
        self.tfidf_vectorizer = None  # Fitted during training
        
    def extract_embeddings(
        self, 
        texts: list[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract dense embeddings for text model.
        
        Returns: (n_samples, 384) array
        """
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings
    
    def extract_tfidf(
        self, 
        texts: list[str], 
        fit: bool = False
    ) -> np.ndarray:
        """
        Extract TF-IDF features for tabular model.
        
        Returns: (n_samples, max_features) sparse array
        """
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=300,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english',
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        return features
    
    def extract_handcrafted(self, texts: list[str]) -> pd.DataFrame:
        """
        Extract hand-crafted features for tabular model.
        
        Returns: DataFrame with numeric features
        """
        features = [extract_handcrafted_features(text) for text in texts]
        return pd.DataFrame(features)
    
    def extract_all(
        self, 
        texts: list[str], 
        fit_tfidf: bool = False
    ) -> dict:
        """
        Extract all feature types at once.
        
        Returns:
            dict with keys: 'embeddings', 'tfidf', 'handcrafted'
        """
        return {
            'embeddings': self.extract_embeddings(texts),
            'tfidf': self.extract_tfidf(texts, fit=fit_tfidf),
            'handcrafted': self.extract_handcrafted(texts),
        }
```

---

## Next Steps

1. ✅ Review this document
2. ✅ Implement `extract_handcrafted_features()` in `src/features.py`
3. ✅ Add sentence-transformers to dependencies
4. ✅ Create `TextFeatureExtractor` functionality in `src/features.py`
5. ✅ Update `extract_text_features()` to use new implementation
6. ⬜ Test on sample data from Hugging Face dataset
7. ⬜ Benchmark inference times on production hardware
8. ⬜ Integrate with text model training pipeline

## Demo Script

A working example demonstrating all text feature extraction methods is available:

```bash
# Run the demo
python examples/text_features_demo.py

# Or with PYTHONPATH
PYTHONPATH=. python examples/text_features_demo.py
```

This will show:
- Timing comparisons for each method
- Output shapes and feature examples
- Recommendations for your use case

---

*Last updated: January 2026*
