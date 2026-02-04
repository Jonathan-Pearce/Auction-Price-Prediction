# =============================================================================
# Auction Price Prediction - Text Embeddings Feature Engineering
# =============================================================================
"""
Text embeddings pipeline for auction item titles and descriptions.

This module implements:
1. Data ingestion from Hugging Face item_data dataset
2. Text preprocessing (stopwords, lowercasing, punctuation, tokenization)
3. FastText embedding model training
4. Embedding export to Parquet format
5. Hugging Face upload functionality

Usage:
    # Train and export embeddings
    python -m src.text_embeddings --train --export

    # Load pretrained model for inference
    python -m src.text_embeddings --inference --item_id 12345
"""

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, settings

# Constants for embedding configuration
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MIN_COUNT = 2
DEFAULT_EPOCHS = 10


# =============================================================================
# Text Preprocessing
# =============================================================================


def get_stopwords() -> set[str]:
    """
    Get English stopwords from NLTK.

    Returns:
        Set of English stopwords.
    """
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))


def preprocess_text(
    text: str | None,
    stopwords_set: set[str] | None = None,
    remove_numbers: bool = True,
) -> list[str]:
    """
    Preprocess text for embedding model training.

    Steps:
    1. Handle None/NaN values
    2. Lowercase all text
    3. Remove punctuation and special symbols
    4. Remove numbers (configurable)
    5. Tokenize
    6. Remove stopwords

    Args:
        text: Raw text string (title or description).
        stopwords_set: Set of stopwords to remove. If None, loads NLTK stopwords.
        remove_numbers: Whether to remove numeric tokens.

    Returns:
        List of preprocessed tokens.
    """
    # Handle missing text
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []

    text = str(text)

    # Skip if empty after conversion
    if not text.strip():
        return []

    # Lowercase
    text = text.lower()

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation and special symbols
    # Keep hyphens between words but remove standalone ones
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s-\s", " ", text)
    text = re.sub(r"^-|-$", " ", text)

    # Optionally remove numbers
    if remove_numbers:
        text = re.sub(r"\b\d+\b", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize (simple whitespace tokenization)
    tokens = text.split()

    # Load stopwords if not provided
    if stopwords_set is None:
        stopwords_set = get_stopwords()

    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in stopwords_set and len(t) > 1]

    return tokens


def combine_title_description(
    title: str | None,
    description: str | None,
    stopwords_set: set[str] | None = None,
) -> list[str]:
    """
    Combine and preprocess title and description into a single token list.

    Args:
        title: Item title.
        description: Item description.
        stopwords_set: Set of stopwords to remove.

    Returns:
        Combined list of preprocessed tokens.
    """
    title_tokens = preprocess_text(title, stopwords_set)
    desc_tokens = preprocess_text(description, stopwords_set)

    # Combine tokens, keeping title tokens first
    return title_tokens + desc_tokens


# =============================================================================
# Data Loading
# =============================================================================


def load_item_data(
    limit: int | None = None,
    streaming: bool = False,
) -> pd.DataFrame:
    """
    Load item data from Hugging Face dataset.

    Args:
        limit: Maximum number of items to load. None for all.
        streaming: Whether to use streaming mode (for large datasets).

    Returns:
        DataFrame with item_id, auction_id, item_title, item_description columns.
    """
    from datasets import load_dataset

    logger.info("Loading item data from Hugging Face...")

    if streaming and limit:
        # Stream mode with limit
        ds = load_dataset("jpearce610/item_data", split="train", streaming=True)
        items = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            items.append(
                {
                    "item_id": item["item_id"],
                    "auction_id": item["auction_id"],
                    "item_title": item["item_title"],
                    "item_description": item["item_description"],
                }
            )
        df = pd.DataFrame(items)
    else:
        # Full load
        ds = load_dataset("jpearce610/item_data", split="train")
        df = ds.to_pandas()[
            ["item_id", "auction_id", "item_title", "item_description"]
        ]
        if limit:
            df = df.head(limit)

    logger.info(f"Loaded {len(df)} items")
    return df


def prepare_training_corpus(df: pd.DataFrame) -> list[list[str]]:
    """
    Prepare corpus for FastText training.

    Args:
        df: DataFrame with item_title and item_description columns.

    Returns:
        List of token lists (one per item).
    """
    logger.info("Preparing training corpus...")
    stopwords_set = get_stopwords()

    corpus = []
    empty_count = 0

    for _, row in df.iterrows():
        tokens = combine_title_description(
            row.get("item_title"), row.get("item_description"), stopwords_set
        )
        if tokens:
            corpus.append(tokens)
        else:
            empty_count += 1

    if empty_count > 0:
        logger.warning(f"{empty_count} items had empty text after preprocessing")

    logger.info(f"Prepared corpus with {len(corpus)} documents")
    return corpus


# =============================================================================
# Embedding Model Training
# =============================================================================


def train_fasttext_model(
    corpus: list[list[str]],
    vector_size: int = DEFAULT_EMBEDDING_DIM,
    window: int = DEFAULT_WINDOW_SIZE,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    sg: int = 1,  # Skip-gram (1) vs CBOW (0)
    workers: int = 4,
) -> Any:
    """
    Train a FastText model on the corpus.

    Args:
        corpus: List of token lists.
        vector_size: Embedding dimension.
        window: Context window size.
        min_count: Minimum word frequency.
        epochs: Training epochs.
        sg: 1 for skip-gram, 0 for CBOW.
        workers: Number of worker threads.

    Returns:
        Trained FastText model.
    """
    from gensim.models import FastText

    logger.info(
        f"Training FastText model (dim={vector_size}, window={window}, "
        f"min_count={min_count}, epochs={epochs}, sg={'skip-gram' if sg else 'cbow'})..."
    )

    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=sg,
        workers=workers,
    )

    logger.info(f"Model trained. Vocabulary size: {len(model.wv)}")
    return model


def get_document_embedding(
    tokens: list[str],
    model: Any,
) -> np.ndarray:
    """
    Get document embedding by averaging word vectors.

    Args:
        tokens: List of preprocessed tokens.
        model: Trained FastText model.

    Returns:
        Document embedding vector.
    """
    if not tokens:
        # Return zero vector for empty documents
        return np.zeros(model.wv.vector_size, dtype=np.float32)

    vectors = []
    for token in tokens:
        # FastText can generate vectors for OOV words
        vectors.append(model.wv[token])

    # Average the vectors
    return np.mean(vectors, axis=0).astype(np.float32)


def evaluate_embeddings(model: Any, sample_words: list[str] | None = None) -> None:
    """
    Evaluate embedding quality by checking nearest neighbors.

    Args:
        model: Trained FastText model.
        sample_words: Words to check neighbors for.
    """
    if sample_words is None:
        # Default sample words relevant to auctions
        sample_words = [
            "furniture",
            "antique",
            "vintage",
            "table",
            "chair",
            "collectible",
        ]

    logger.info("Evaluating embedding quality...")

    for word in sample_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            neighbor_str = ", ".join([f"{w} ({s:.2f})" for w, s in neighbors])
            logger.info(f"  '{word}': {neighbor_str}")
        else:
            logger.info(f"  '{word}': (not in vocabulary)")


# =============================================================================
# Embedding Export
# =============================================================================


def generate_item_embeddings(
    df: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """
    Generate embeddings for all items.

    Args:
        df: DataFrame with item_id, auction_id, item_title, item_description.
        model: Trained FastText model.

    Returns:
        DataFrame with auction_id, item_id, and embedding columns.
    """
    logger.info("Generating item embeddings...")
    stopwords_set = get_stopwords()

    embeddings = []
    for _, row in df.iterrows():
        tokens = combine_title_description(
            row.get("item_title"), row.get("item_description"), stopwords_set
        )
        embedding = get_document_embedding(tokens, model)
        embeddings.append(
            {
                "auction_id": row["auction_id"],
                "item_id": row["item_id"],
                "embedding": embedding.tolist(),  # Convert to list for Parquet
            }
        )

    result_df = pd.DataFrame(embeddings)
    logger.info(f"Generated embeddings for {len(result_df)} items")
    return result_df


def save_embeddings_parquet(
    embeddings_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """
    Save embeddings to Parquet format.

    Args:
        embeddings_df: DataFrame with auction_id, item_id, embedding columns.
        output_path: Output file path. Defaults to processed data directory.

    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "text_embeddings.parquet"
    else:
        output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    embeddings_df.to_parquet(output_path, index=False)
    logger.info(f"Saved embeddings to {output_path}")

    return output_path


def save_model(
    model: Any,
    output_path: Path | str | None = None,
) -> Path:
    """
    Save trained FastText model.

    Args:
        model: Trained FastText model.
        output_path: Output file path. Defaults to models directory.

    Returns:
        Path to saved model.
    """
    if output_path is None:
        output_path = MODELS_DIR / "fasttext_text_embeddings.model"
    else:
        output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(output_path))
    logger.info(f"Saved model to {output_path}")

    return output_path


def load_model(model_path: Path | str | None = None) -> Any:
    """
    Load trained FastText model.

    Args:
        model_path: Path to saved model. Defaults to models directory.

    Returns:
        Loaded FastText model.
    """
    from gensim.models import FastText

    if model_path is None:
        model_path = MODELS_DIR / "fasttext_text_embeddings.model"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = FastText.load(str(model_path))
    logger.info(f"Loaded model from {model_path}")

    return model


# =============================================================================
# Hugging Face Upload
# =============================================================================


def upload_to_huggingface(
    embeddings_path: Path | str,
    repo_id: str = "jpearce610/text_embeddings",
    private: bool = False,
) -> None:
    """
    Upload embeddings to Hugging Face Hub.

    Args:
        embeddings_path: Path to embeddings Parquet file.
        repo_id: Hugging Face repository ID.
        private: Whether the repository should be private.
    """
    from huggingface_hub import HfApi

    embeddings_path = Path(embeddings_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

    logger.info(f"Uploading embeddings to Hugging Face: {repo_id}...")

    token = settings.huggingface.token
    if not token:
        logger.warning(
            "HF_TOKEN not set. Upload may fail if repo doesn't exist or is private."
        )

    api = HfApi(token=token)

    # Create dataset repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Upload the file
    api.upload_file(
        path_or_fileobj=str(embeddings_path),
        path_in_repo="text_embeddings.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

    logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


# =============================================================================
# Inference Functions
# =============================================================================


def get_embedding_for_item(
    item_title: str | None,
    item_description: str | None,
    model: Any | None = None,
) -> np.ndarray:
    """
    Get embedding for a single item (inference).

    Args:
        item_title: Item title text.
        item_description: Item description text.
        model: Trained FastText model. If None, loads from default path.

    Returns:
        Embedding vector as numpy array.

    Example:
        >>> model = load_model()
        >>> embedding = get_embedding_for_item(
        ...     "Vintage Oak Dining Table",
        ...     "Beautiful antique oak table from 1920s. Seats 6.",
        ...     model
        ... )
        >>> print(embedding.shape)  # (100,)
    """
    if model is None:
        model = load_model()

    stopwords_set = get_stopwords()
    tokens = combine_title_description(item_title, item_description, stopwords_set)
    return get_document_embedding(tokens, model)


# =============================================================================
# Main Training Pipeline
# =============================================================================


def run_training_pipeline(
    limit: int | None = None,
    vector_size: int = DEFAULT_EMBEDDING_DIM,
    window: int = DEFAULT_WINDOW_SIZE,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    upload: bool = False,
    hf_repo_id: str = "jpearce610/text_embeddings",
) -> tuple[Any, pd.DataFrame, Path]:
    """
    Run the complete training pipeline.

    Args:
        limit: Maximum number of items to load (None for all).
        vector_size: Embedding dimension.
        window: Context window size.
        min_count: Minimum word frequency.
        epochs: Training epochs.
        upload: Whether to upload embeddings to Hugging Face.
        hf_repo_id: Hugging Face repository ID for upload.

    Returns:
        Tuple of (model, embeddings_df, embeddings_path).
    """
    logger.info("Starting text embeddings training pipeline...")

    # 1. Load data
    df = load_item_data(limit=limit)

    # 2. Prepare corpus
    corpus = prepare_training_corpus(df)

    # 3. Train model
    model = train_fasttext_model(
        corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
    )

    # 4. Evaluate model
    evaluate_embeddings(model)

    # 5. Generate embeddings for all items
    embeddings_df = generate_item_embeddings(df, model)

    # 6. Save model and embeddings
    save_model(model)
    embeddings_path = save_embeddings_parquet(embeddings_df)

    # 7. Optionally upload to Hugging Face
    if upload:
        upload_to_huggingface(embeddings_path, repo_id=hf_repo_id)

    logger.info("Training pipeline complete!")
    return model, embeddings_df, embeddings_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Text embeddings pipeline for auction items"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train embedding model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Context window size (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help=f"Minimum word frequency (default: {DEFAULT_MIN_COUNT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload embeddings to Hugging Face",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="jpearce610/text_embeddings",
        help="Hugging Face repository ID for upload",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference mode with sample text",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Item title for inference",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Item description for inference",
    )

    args = parser.parse_args()

    if args.train:
        run_training_pipeline(
            limit=args.limit,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            upload=args.upload,
            hf_repo_id=args.hf_repo_id,
        )
    elif args.inference:
        if args.title is None and args.description is None:
            # Demo with sample text
            args.title = "Vintage Oak Dining Table"
            args.description = "Beautiful antique oak table from 1920s. Seats 6."
            logger.info(f"Using demo text - Title: '{args.title}'")

        model = load_model()
        embedding = get_embedding_for_item(args.title, args.description, model)
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info(f"Embedding (first 10 values): {embedding[:10]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
