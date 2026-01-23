# =============================================================================
# Auction Price Prediction - Data Collection Package
# =============================================================================
"""
Data collection from MaxSold API.

Components:
- maxsold_client: API wrapper for MaxSold endpoints
- scraper: Orchestration of data collection
- schemas: Pydantic models for data validation
"""

from src.data.schemas import Auction, Item, Bid, AuctionResponse
from src.data.maxsold_client import MaxSoldClient

__all__ = ["Auction", "Item", "Bid", "AuctionResponse", "MaxSoldClient"]
