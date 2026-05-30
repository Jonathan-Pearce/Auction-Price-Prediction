# =============================================================================
# Auction Price Prediction - FastAPI Application
# =============================================================================
"""
FastAPI application for serving auction price predictions.

Endpoints:
- POST /predict: Predict price for an auction item
- POST /predict/url: Predict price from MaxSold URL
- POST /similar: Retrieve similar past auction items (RAG)
- GET /health: Health check
- GET /models: List available models

Deployment:
- Local: uvicorn api.main:app --reload
- Production: uvicorn api.main:app --workers 4
- HF Spaces: Integrated with Gradio
"""

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from loguru import logger

from src.config import PROCESSED_DATA_DIR, settings
from src.rag import SimilarItemRetriever
from src.text_embeddings import combine_title_description


# =============================================================================
# Request/Response Models
# =============================================================================


class PredictRequest(BaseModel):
    """Request body for price prediction."""

    item_id: int = Field(..., description="MaxSold item ID")
    auction_id: int = Field(..., description="MaxSold auction ID")
    # Optional: provide data directly instead of fetching
    item_data: dict[str, Any] | None = Field(
        default=None, description="Optional pre-fetched item data"
    )


class PredictFromURLRequest(BaseModel):
    """Request body for prediction from URL."""

    url: HttpUrl = Field(..., description="MaxSold item URL")


class PredictResponse(BaseModel):
    """Response body for price prediction."""

    item_id: str
    predicted_price: float = Field(..., description="Predicted winning price in USD")
    confidence_interval: dict[str, float] = Field(
        ..., description="Lower and upper bounds of prediction"
    )
    model_predictions: dict[str, float | None] = Field(
        default_factory=dict, description="Individual model predictions"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    models_loaded: bool
    environment: str


class ModelsResponse(BaseModel):
    """Available models response."""

    models: list[str]
    fusion_model: str | None


# =============================================================================
# Similar Items (RAG) Models
# =============================================================================


class SimilarItemsRequest(BaseModel):
    """Request body for similar-item retrieval."""

    item_title: str | None = Field(default=None, description="Item title text")
    item_description: str | None = Field(
        default=None, description="Item description text"
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar items to return (1-20)",
    )
    exclude_item_ids: list[int] = Field(
        default_factory=list,
        description="Item IDs to exclude from results (e.g. the query item itself)",
    )


class SimilarItemResult(BaseModel):
    """A single similar auction item returned by the RAG retriever."""

    item_id: int
    auction_id: int
    similarity: float = Field(..., description="Cosine similarity score (0-1)")
    item_title: str | None = None
    item_description: str | None = None
    winning_price: float | None = Field(
        default=None, description="Historical winning price in USD"
    )


class SimilarItemsResponse(BaseModel):
    """Response body for similar-item retrieval."""

    similar_items: list[SimilarItemResult]
    query_title: str | None = None
    query_description: str | None = None
    retriever_index_size: int = Field(
        ..., description="Total number of items in the retrieval index"
    )


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting Auction Price Prediction API...")

    # Load prediction models
    # TODO: Initialize EnsemblePredictor and load models
    # app.state.predictor = EnsemblePredictor()
    logger.info("Models loaded (placeholder)")

    # Load RAG retriever (best-effort: skip if embeddings file is absent)
    embeddings_path = PROCESSED_DATA_DIR / "text_embeddings.parquet"
    if embeddings_path.exists():
        try:
            app.state.retriever = SimilarItemRetriever.from_parquet(embeddings_path)
            logger.info(
                f"RAG retriever loaded: {app.state.retriever.index_size} items indexed"
            )
        except Exception as exc:
            logger.warning(f"RAG retriever failed to load ({exc}); /similar disabled")
            app.state.retriever = None
    else:
        logger.info(
            "No embeddings file found at "
            f"{embeddings_path}; /similar endpoint will return empty results"
        )
        app.state.retriever = None

    # Load FastText model for query embedding (best-effort)
    app.state.fasttext_model = None
    try:
        from src.text_embeddings import load_model as load_fasttext_model

        app.state.fasttext_model = load_fasttext_model()
        logger.info("FastText embedding model loaded for RAG query encoding")
    except Exception as exc:
        logger.info(
            f"FastText model not available ({exc}); "
            "RAG queries will use zero-vector fallback"
        )

    yield

    # Shutdown
    logger.info("Shutting down API...")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Auction Price Prediction API",
    description=(
        "API for predicting winning prices of MaxSold auction items. "
        "Uses an ensemble of ML models (tabular, image, text, sequential) "
        "combined with a fusion model."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "Auction Price Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns API status and model availability.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=False,  # TODO: Check actual model status
        environment=settings.environment,
    )


@app.get("/models", response_model=ModelsResponse, tags=["System"])
async def list_models():
    """
    List available prediction models.

    Returns the names of loaded models and fusion model.
    """
    # TODO: Return actual loaded models
    return ModelsResponse(
        models=["tabular", "image", "text", "sequential"],
        fusion_model="neural_fusion",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict winning price for an auction item.

    Provide either:
    - item_id and auction_id (will fetch data from MaxSold API)
    - item_data directly (for faster inference)

    Returns predicted price with confidence interval and
    individual model predictions.
    """
    try:
        # TODO: Implement actual prediction
        # predictor = app.state.predictor
        # result = await predictor.predict(...)

        # Placeholder response
        logger.info(f"Predicting for item {request.item_id} in auction {request.auction_id}")

        return PredictResponse(
            item_id=str(request.item_id),
            predicted_price=0.0,
            confidence_interval={"lower": 0.0, "upper": 0.0},
            model_predictions={
                "tabular": None,
                "image": None,
                "text": None,
                "sequential": None,
            },
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/url", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_url(request: PredictFromURLRequest):
    """
    Predict winning price from MaxSold item URL.

    Parses the URL to extract auction_id and item_id,
    fetches item data from MaxSold API, and returns prediction.

    Example URL: https://maxsold.com/auction/12345/item/67890
    """
    try:
        # TODO: Parse URL and make prediction
        # predictor = app.state.predictor
        # result = await predictor.predict_from_url(str(request.url))

        logger.info(f"Predicting for URL: {request.url}")

        # Placeholder response
        return PredictResponse(
            item_id="from_url",
            predicted_price=0.0,
            confidence_interval={"lower": 0.0, "upper": 0.0},
            model_predictions={},
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid URL: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Prediction from URL failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/similar", response_model=SimilarItemsResponse, tags=["RAG"])
async def similar_items(request: SimilarItemsRequest):
    """
    Retrieve similar past auction items using RAG (embedding-based retrieval).

    Given an item title and/or description, returns the top-k most similar
    items from historical auctions ranked by cosine similarity of their
    FastText text embeddings.

    This endpoint is intended to complement price predictions by showing
    users concrete examples of comparable items that have already sold,
    along with their historical winning prices.

    The retriever is loaded at startup from the pre-computed embeddings
    file (``data/processed/text_embeddings.parquet``).  If the file is not
    present the endpoint returns an empty list without an error.

    Args:
        request: Contains item_title, item_description, k (1-20),
            and optional exclude_item_ids.

    Returns:
        Up to *k* similar items ordered by descending similarity.
    """
    retriever = getattr(app.state, "retriever", None)

    if retriever is None:
        logger.info("RAG retriever not available; returning empty similar items")
        return SimilarItemsResponse(
            similar_items=[],
            query_title=request.item_title,
            query_description=request.item_description,
            retriever_index_size=0,
        )

    if request.item_title is None and request.item_description is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of item_title or item_description must be provided.",
        )

    try:
        from src.text_embeddings import get_document_embedding

        tokens = combine_title_description(request.item_title, request.item_description)
        if not tokens:
            return SimilarItemsResponse(
                similar_items=[],
                query_title=request.item_title,
                query_description=request.item_description,
                retriever_index_size=retriever.index_size,
            )

        fasttext_model = getattr(app.state, "fasttext_model", None)
        if fasttext_model is not None:
            query_embedding = get_document_embedding(tokens, fasttext_model)
        else:
            # FastText model not loaded: return empty results rather than
            # returning meaningless similarity scores from a zero vector.
            logger.info(
                "FastText model not loaded; cannot embed query for /similar"
            )
            return SimilarItemsResponse(
                similar_items=[],
                query_title=request.item_title,
                query_description=request.item_description,
                retriever_index_size=retriever.index_size,
            )

        results = retriever.retrieve(
            query_embedding=query_embedding,
            k=request.k,
            exclude_item_ids=request.exclude_item_ids or [],
        )

        similar_items_out = [
            SimilarItemResult(
                item_id=r.item_id,
                auction_id=r.auction_id,
                similarity=r.similarity,
                item_title=r.item_title,
                item_description=r.item_description,
                winning_price=r.winning_price,
            )
            for r in results
        ]

        logger.info(
            f"Similar items retrieved: {len(similar_items_out)} results "
            f"for title='{request.item_title}'"
        )

        return SimilarItemsResponse(
            similar_items=similar_items_out,
            query_title=request.item_title,
            query_description=request.item_description,
            retriever_index_size=retriever.index_size,
        )

    except Exception as e:
        logger.error(f"Similar items retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar items retrieval failed: {str(e)}",
        )


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api.workers,
    )


if __name__ == "__main__":
    main()
