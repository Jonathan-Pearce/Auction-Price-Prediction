# =============================================================================
# Auction Price Prediction - Modeling Package
# =============================================================================
"""
Machine learning models for auction price prediction.

Model Types:
- Tabular: XGBoost, Random Forest, or Neural Network for structured features
- Image: CNN for item photos
- Text: Transformer for item descriptions
- Sequential: LSTM/GRU for bid history time series
- Fusion: Meta-model combining all model predictions
"""

from src.modeling.train import main as train_main
from src.modeling.predict import main as predict_main

__all__ = ["train_main", "predict_main"]
