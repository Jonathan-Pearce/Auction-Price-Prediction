# =============================================================================
# Auction Price Prediction - API Package
# =============================================================================
"""
FastAPI backend for auction price prediction.

Components:
- main: FastAPI application and entry point
- routes: API endpoints
- gradio_app: Gradio interface for HF Spaces
"""

from api.main import app

__all__ = ["app"]
