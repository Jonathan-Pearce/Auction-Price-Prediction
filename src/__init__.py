# =============================================================================
# Auction Price Prediction - Source Package
# =============================================================================
"""
Auction Price Prediction package for MaxSold auctions.

This package provides:
- Data collection from MaxSold API
- Feature engineering pipelines
- ML model training (tabular, image, text, sequential)
- Model fusion and ensemble methods
"""

__version__ = "0.1.0"
__author__ = "Jonathan Pearce"

from src.config import settings

__all__ = ["settings", "__version__"]
