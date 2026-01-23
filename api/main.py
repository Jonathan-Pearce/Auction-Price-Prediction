# =============================================================================
# Auction Price Prediction - FastAPI Application
# =============================================================================
"""
FastAPI application for serving auction price predictions.

Endpoints:
- POST /predict: Predict price for an auction item
- POST /predict/url: Predict price from MaxSold URL
- GET /health: Health check
- GET /models: List available models

Deployment:
- Local: uvicorn api.main:app --reload
- Production: uvicorn api.main:app --workers 4
- HF Spaces: Integrated with Gradio
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from loguru import logger

from src.config import settings


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
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting Auction Price Prediction API...")

    # Load models
    # TODO: Initialize EnsemblePredictor and load models
    # app.state.predictor = EnsemblePredictor()
    logger.info("Models loaded (placeholder)")

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
