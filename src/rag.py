# =============================================================================
# Auction Price Prediction - RAG (Retrieval-Augmented Generation) Module
# =============================================================================
"""
Retrieval-Augmented Generation (RAG) system for finding similar auction items.

This module implements a similarity-based retrieval system that:
1. Indexes pre-computed FastText embeddings of auction item descriptions
2. Given a query item (title + description), finds the most similar
   historical items using cosine similarity
3. Returns the top-k similar items with metadata and sale prices

This allows the UI to present users with not only a price prediction but
also concrete examples of similar items that have already sold.

Usage:
    from src.rag import SimilarItemRetriever

    # Load pre-computed embeddings from parquet
    retriever = SimilarItemRetriever.from_parquet("data/processed/text_embeddings.parquet")

    # Load the FastText model
    from src.text_embeddings import load_model
    model = load_model()

    # Retrieve similar items for a new query
    results = retriever.retrieve_by_text(
        title="Antique Oak Dining Table",
        description="Solid oak table from 1920s, seats 6.",
        model=model,
        k=5,
    )
    for r in results:
        print(r["item_id"], r["similarity"], r["winning_price"])
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROCESSED_DATA_DIR
from src.text_embeddings import combine_title_description, get_document_embedding


# =============================================================================
# Similar Item Result Type
# =============================================================================


class SimilarItem:
    """Represents a retrieved similar auction item."""

    def __init__(
        self,
        item_id: int,
        auction_id: int,
        similarity: float,
        item_title: str | None = None,
        item_description: str | None = None,
        winning_price: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.item_id = item_id
        self.auction_id = auction_id
        self.similarity = float(similarity)
        self.item_title = item_title
        self.item_description = item_description
        self.winning_price = winning_price
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "item_id": self.item_id,
            "auction_id": self.auction_id,
            "similarity": round(self.similarity, 4),
            "item_title": self.item_title,
            "item_description": self.item_description,
            "winning_price": self.winning_price,
        }

    def __repr__(self) -> str:
        return (
            f"SimilarItem(item_id={self.item_id}, "
            f"similarity={self.similarity:.4f}, "
            f"winning_price={self.winning_price})"
        )


# =============================================================================
# Retriever
# =============================================================================


class SimilarItemRetriever:
    """
    Retrieves the most similar past auction items for a given query item.

    The retriever builds an in-memory index of L2-normalised embedding
    vectors. Similarity queries are then just a matrix-vector dot product
    (equivalent to cosine similarity on unit vectors), which is fast even
    for millions of items with NumPy.

    Attributes:
        index_matrix: (N, D) float32 array of unit-normalised embeddings.
        metadata: DataFrame of item metadata aligned row-for-row with
            index_matrix (contains item_id, auction_id, and any other
            columns such as item_title or winning_price).
    """

    def __init__(self) -> None:
        self.index_matrix: np.ndarray | None = None
        self.metadata: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, embeddings_df: pd.DataFrame) -> None:
        """
        Build the similarity index from a DataFrame of pre-computed embeddings.

        The DataFrame must contain at minimum:
        - ``item_id``   – unique item identifier
        - ``auction_id`` – auction identifier
        - ``embedding``  – per-item embedding as a list or 1-D numpy array

        Any additional columns (e.g. ``item_title``, ``item_description``,
        ``winning_price``) are carried through and returned with results.

        Args:
            embeddings_df: DataFrame with embedding and metadata columns.
        """
        if embeddings_df.empty:
            raise ValueError("embeddings_df must not be empty")

        required_cols = {"item_id", "auction_id", "embedding"}
        missing = required_cols - set(embeddings_df.columns)
        if missing:
            raise ValueError(f"embeddings_df is missing columns: {missing}")

        logger.info(f"Building RAG index from {len(embeddings_df)} items...")

        # Stack embeddings into a (N, D) matrix
        raw = np.vstack(embeddings_df["embedding"].apply(np.asarray).values).astype(
            np.float32
        )

        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Avoid division by zero for all-zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        self.index_matrix = raw / norms

        # Store metadata (drop the embedding column to save memory)
        self.metadata = embeddings_df.drop(columns=["embedding"]).reset_index(drop=True)

        logger.info(
            f"RAG index built: {self.index_matrix.shape[0]} items, "
            f"embedding dim={self.index_matrix.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> "SimilarItemRetriever":
        """
        Create a retriever by loading embeddings from a Parquet file.

        Args:
            path: Path to the embeddings Parquet file.  Defaults to
                ``data/processed/text_embeddings.parquet``.
            metadata_path: Optional path to a separate metadata Parquet
                (e.g. with winning_price, item_title).  When provided its
                columns are merged into the index on ``item_id``.

        Returns:
            Initialised SimilarItemRetriever with index built.
        """
        if path is None:
            path = PROCESSED_DATA_DIR / "text_embeddings.parquet"
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")

        logger.info(f"Loading embeddings from {path}...")
        embeddings_df = pd.read_parquet(path)

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                extra_meta = pd.read_parquet(metadata_path)
                embeddings_df = embeddings_df.merge(extra_meta, on="item_id", how="left")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")

        retriever = cls()
        retriever.build_index(embeddings_df)
        return retriever

    @classmethod
    def from_dataframe(cls, embeddings_df: pd.DataFrame) -> "SimilarItemRetriever":
        """
        Create a retriever directly from an in-memory DataFrame.

        Args:
            embeddings_df: DataFrame with embedding and metadata columns.

        Returns:
            Initialised SimilarItemRetriever with index built.
        """
        retriever = cls()
        retriever.build_index(embeddings_df)
        return retriever

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_item_ids: list[int] | None = None,
    ) -> list[SimilarItem]:
        """
        Retrieve the top-k most similar items for a given embedding vector.

        Args:
            query_embedding: 1-D numpy array (the query item's embedding).
            k: Number of similar items to return (default 5, max 50).
            exclude_item_ids: Optional list of item IDs to exclude from results
                (e.g. the query item itself if it is already in the index).

        Returns:
            List of :class:`SimilarItem` objects ordered by descending
            similarity score.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index_matrix is None or self.metadata is None:
            raise RuntimeError(
                "Index has not been built. Call build_index() or use a "
                "factory constructor (from_parquet / from_dataframe)."
            )

        k = min(k, 50)  # hard cap to prevent abuse

        # Normalise query vector
        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Cosine similarities via dot product (index is pre-normalised)
        scores = self.index_matrix @ query  # shape (N,)

        # Apply exclusions
        exclude_set: set[int] = set(exclude_item_ids or [])
        if exclude_set:
            mask = self.metadata["item_id"].isin(exclude_set).values
            scores[mask] = -np.inf

        # Get top-k indices: use full argsort for small/medium indices or when
        # retrieving a large fraction of the index, otherwise use argpartition
        # (O(N) vs O(N log N)) for large sparse indices.
        n_candidates = min(k, len(scores))
        if n_candidates == 0:
            return []

        n_total = len(scores)
        if n_candidates >= n_total // 2:
            # Direct sort is simpler and avoids a second sort pass
            top_indices = np.argsort(scores)[::-1][:n_candidates]
        else:
            top_indices = np.argpartition(scores, -n_candidates)[-n_candidates:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[SimilarItem] = []
        for idx in top_indices:
            if scores[idx] == -np.inf:
                continue
            row = self.metadata.iloc[int(idx)]
            results.append(
                SimilarItem(
                    item_id=int(row["item_id"]),
                    auction_id=int(row["auction_id"]),
                    similarity=float(scores[idx]),
                    item_title=row.get("item_title"),
                    item_description=row.get("item_description"),
                    winning_price=row.get("winning_price"),
                )
            )

        return results

    def retrieve_by_text(
        self,
        title: str | None,
        description: str | None,
        model: Any,
        k: int = 5,
        exclude_item_ids: list[int] | None = None,
    ) -> list[SimilarItem]:
        """
        End-to-end retrieval: preprocess text, embed, then retrieve.

        Args:
            title: Item title string (may be None).
            description: Item description string (may be None).
            model: Trained FastText model (from :func:`src.text_embeddings.load_model`).
            k: Number of similar items to return.
            exclude_item_ids: Item IDs to exclude from results.

        Returns:
            List of :class:`SimilarItem` objects ordered by descending similarity.
        """
        tokens = combine_title_description(title, description)
        query_embedding = get_document_embedding(tokens, model)
        return self.retrieve(query_embedding, k=k, exclude_item_ids=exclude_item_ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Number of items in the index."""
        if self.index_matrix is None:
            return 0
        return int(self.index_matrix.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension of the index."""
        if self.index_matrix is None:
            return 0
        return int(self.index_matrix.shape[1])

    def __repr__(self) -> str:
        return (
            f"SimilarItemRetriever("
            f"index_size={self.index_size}, "
            f"embedding_dim={self.embedding_dim})"
        )
