# Text Embeddings Feature Engineering

This document describes the text embeddings pipeline for auction item titles and descriptions.

## Overview

The text embeddings module converts auction item text (titles and descriptions) into compact numerical vectors using FastText. These embeddings can be used as features for downstream ML tasks like price prediction.

## Pipeline Steps

### 1. Data Ingestion

Data is loaded from the Hugging Face dataset `jpearce610/item_data`:

```python
from src.text_embeddings import load_item_data

# Load all data
df = load_item_data()

# Load with limit (for testing)
df = load_item_data(limit=1000)
```

**Source columns used:**
- `item_id`: Unique item identifier
- `auction_id`: Auction identifier
- `item_title`: Item title text
- `item_description`: Item description text

### 2. Text Preprocessing

Text is preprocessed with the following steps:

1. **Handle missing values**: None/NaN values return empty token list
2. **Lowercase**: All text converted to lowercase
3. **Remove HTML tags**: Strip any HTML markup
4. **Remove URLs**: Strip web addresses
5. **Remove email addresses**: Strip email patterns
6. **Remove punctuation**: Strip special characters (except hyphens in words)
7. **Remove numbers**: Pure numeric tokens removed (alphanumeric like "1920s" kept)
8. **Tokenize**: Split on whitespace
9. **Remove stopwords**: Using NLTK English stopwords
10. **Remove short tokens**: Single-character tokens removed

```python
from src.text_embeddings import preprocess_text, get_stopwords

stopwords = get_stopwords()
tokens = preprocess_text("Beautiful Antique Oak Table - 1920s!", stopwords)
# Result: ['beautiful', 'antique', 'oak', 'table', '1920s']
```

### 3. Embedding Model Training

FastText is used for embeddings due to its ability to handle:
- Short text (item titles)
- Noisy text (user-generated descriptions)
- Out-of-vocabulary words (via subword information)

**Default parameters:**
- Vector size: 100 dimensions
- Window size: 5
- Minimum word count: 2
- Algorithm: Skip-gram (better for sparse data)
- Training epochs: 10

```python
from src.text_embeddings import train_fasttext_model, prepare_training_corpus

# Prepare corpus from DataFrame
corpus = prepare_training_corpus(df)

# Train model with custom parameters
model = train_fasttext_model(
    corpus,
    vector_size=100,
    window=5,
    min_count=2,
    epochs=10,
    sg=1  # Skip-gram
)
```

### 4. Embedding Generation

Document embeddings are computed by averaging word vectors:

```python
from src.text_embeddings import get_embedding_for_item, load_model

model = load_model()
embedding = get_embedding_for_item(
    "Vintage Oak Table",
    "Beautiful wooden table from the 1920s",
    model
)
# Result: numpy array of shape (100,)
```

### 5. Export & Upload

Embeddings are exported to Parquet format:

```python
from src.text_embeddings import (
    generate_item_embeddings,
    save_embeddings_parquet,
    save_model,
    upload_to_huggingface
)

# Generate embeddings for all items
embeddings_df = generate_item_embeddings(df, model)

# Save to Parquet
save_embeddings_parquet(embeddings_df, "text_embeddings.parquet")

# Save model
save_model(model, "fasttext_model.model")

# Upload to Hugging Face
upload_to_huggingface("text_embeddings.parquet", repo_id="jpearce610/text_embeddings")
```

**Output schema:**
| Column | Type | Description |
|--------|------|-------------|
| `auction_id` | int | Auction identifier |
| `item_id` | int | Item identifier |
| `embedding` | list[float] | 100-dimensional embedding vector |

## Usage

### Full Training Pipeline

```bash
# Train on full dataset
python -m src.text_embeddings --train

# Train on subset (for testing)
python -m src.text_embeddings --train --limit 10000

# Train with custom parameters
python -m src.text_embeddings --train --vector-size 200 --epochs 20

# Train and upload to Hugging Face
python -m src.text_embeddings --train --upload --hf-repo-id username/text_embeddings
```

### Inference on New Items

```bash
# Using CLI
python -m src.text_embeddings --inference \
    --title "Vintage Chair" \
    --description "Beautiful Victorian wooden chair"
```

```python
# Using Python API
from src.text_embeddings import get_embedding_for_item, load_model

# Load pretrained model
model = load_model()

# Get embedding for new item
embedding = get_embedding_for_item(
    item_title="Antique Clock",
    item_description="Brass pendulum clock from 1890s. Working condition.",
    model=model
)

# Use embedding as feature for prediction
print(f"Embedding shape: {embedding.shape}")  # (100,)
```

### Loading Saved Embeddings

```python
import pandas as pd

# Load from Parquet
embeddings_df = pd.read_parquet("data/processed/text_embeddings.parquet")

# Access embedding for specific item
item_embedding = embeddings_df[embeddings_df['item_id'] == 12345]['embedding'].iloc[0]

# Convert list back to numpy array
import numpy as np
embedding_array = np.array(item_embedding)
```

## File Locations

| File | Path | Description |
|------|------|-------------|
| Embeddings | `data/processed/text_embeddings.parquet` | Item embeddings |
| Model | `models/fasttext_text_embeddings.model` | Trained FastText model |
| Source | `src/text_embeddings.py` | Pipeline code |
| Tests | `tests/test_text_embeddings.py` | Unit tests |

## Requirements

Additional dependencies needed:
- `nltk`: For stopwords and tokenization
- `gensim`: For FastText model

Install with:
```bash
pip install nltk gensim
```

NLTK data will be downloaded automatically on first use.

## Edge Cases

| Case | Handling |
|------|----------|
| Missing title | Use description only |
| Missing description | Use title only |
| Both missing | Return zero vector |
| Empty text after preprocessing | Return zero vector |
| Unknown words | FastText generates vectors using subwords |

## Performance

- **Training time**: ~5-10 minutes on 3M items (standard laptop)
- **Memory usage**: ~2GB RAM for full dataset
- **Embedding size**: 100 floats × 4 bytes = 400 bytes per item
- **Model size**: ~50-100MB depending on vocabulary

## Evaluation

Embedding quality can be evaluated by checking nearest neighbors:

```python
from src.text_embeddings import evaluate_embeddings, load_model

model = load_model()
evaluate_embeddings(model, sample_words=['furniture', 'antique', 'vintage'])

# Output:
# 'furniture': table (0.89), chair (0.87), wooden (0.85), ...
# 'antique': vintage (0.92), collectible (0.88), rare (0.85), ...
```

## Integration with ML Models

The embeddings can be used as input features for the tabular or fusion models:

```python
import numpy as np
import pandas as pd

# Load embeddings
embeddings_df = pd.read_parquet("data/processed/text_embeddings.parquet")

# Convert to feature matrix
embedding_cols = [f"text_emb_{i}" for i in range(100)]
embedding_matrix = np.vstack(embeddings_df['embedding'].values)
text_features_df = pd.DataFrame(embedding_matrix, columns=embedding_cols)
text_features_df['item_id'] = embeddings_df['item_id']

# Merge with other features
full_features = other_features.merge(text_features_df, on='item_id')
```
