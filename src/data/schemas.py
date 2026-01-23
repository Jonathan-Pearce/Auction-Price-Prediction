# =============================================================================
# Auction Price Prediction - Data Schemas
# =============================================================================
"""
Pydantic models for MaxSold API data validation.

These schemas define the structure of:
- Auction data
- Item data (including enriched info)
- Bid history data
- API responses
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# =============================================================================
# Bid Schema
# =============================================================================


class Bid(BaseModel):
    """Individual bid in an auction item's bidding history."""

    bid_id: str | None = Field(default=None, description="Unique bid identifier")
    bidder_id: str | None = Field(default=None, description="Anonymous bidder ID")
    amount: float = Field(..., description="Bid amount in dollars")
    bid_time: datetime | None = Field(default=None, description="Timestamp of bid")
    is_winning: bool = Field(default=False, description="Whether this is the winning bid")

    # Soft-close related
    extended_auction: bool = Field(
        default=False, description="Whether this bid triggered a soft-close extension"
    )

    class Config:
        extra = "allow"  # Allow additional fields from API


# =============================================================================
# Item Schema
# =============================================================================


class ItemImage(BaseModel):
    """Image associated with an auction item."""

    url: HttpUrl = Field(..., description="Image URL")
    thumbnail_url: HttpUrl | None = Field(default=None, description="Thumbnail URL")
    is_primary: bool = Field(default=False, description="Whether this is the main image")
    order: int = Field(default=0, description="Display order")


class Item(BaseModel):
    """Auction item with all associated data."""

    # Identifiers
    item_id: int = Field(..., description="Unique item identifier")
    auction_id: int = Field(..., description="Parent auction ID")
    lot_number: int | None = Field(default=None, description="Lot number in auction")

    # Basic info
    title: str = Field(..., description="Item title")
    description: str | None = Field(default=None, description="Item description")
    category: str | None = Field(default=None, description="Item category")
    subcategory: str | None = Field(default=None, description="Item subcategory")

    # Pricing
    starting_bid: float = Field(default=0.0, description="Starting bid amount")
    current_bid: float = Field(default=0.0, description="Current highest bid")
    winning_price: float | None = Field(
        default=None, description="Final winning price (if auction ended)"
    )
    estimated_value: float | None = Field(
        default=None, description="Estimated value (if provided)"
    )

    # Bidding stats
    num_bids: int = Field(default=0, description="Total number of bids")
    num_bidders: int | None = Field(default=None, description="Number of unique bidders")

    # Images
    images: list[ItemImage] = Field(default_factory=list, description="Item images")
    primary_image_url: HttpUrl | None = Field(default=None, description="Primary image URL")

    # Timing
    end_time: datetime | None = Field(default=None, description="Scheduled end time")
    actual_end_time: datetime | None = Field(
        default=None, description="Actual end time (may differ due to soft-close)"
    )

    # Status
    status: str | None = Field(default=None, description="Item status")
    has_bids: bool = Field(default=False, description="Whether item received any bids")

    # Bid history (populated separately)
    bids: list[Bid] = Field(default_factory=list, description="Bidding history")

    # Enriched data (from enriched API endpoint)
    enriched_data: dict[str, Any] | None = Field(
        default=None, description="Additional enriched data"
    )

    class Config:
        extra = "allow"

    @property
    def is_no_bid(self) -> bool:
        """Check if item received no bids."""
        return self.num_bids == 0 or self.winning_price == 0


# =============================================================================
# Auction Schema
# =============================================================================


class AuctionLocation(BaseModel):
    """Location information for an auction."""

    city: str | None = Field(default=None)
    state: str | None = Field(default=None, description="State or province")
    country: str | None = Field(default=None)
    postal_code: str | None = Field(default=None)


class Auction(BaseModel):
    """Auction event containing multiple items."""

    # Identifiers
    auction_id: int = Field(..., description="Unique auction identifier")
    url: HttpUrl | None = Field(default=None, description="Auction URL")

    # Basic info
    title: str = Field(..., description="Auction title")
    description: str | None = Field(default=None, description="Auction description")
    auction_type: str | None = Field(
        default=None, description="Type: estate_sale, moving, reseller, etc."
    )

    # Location
    location: AuctionLocation | None = Field(default=None, description="Auction location")

    # Timing
    start_time: datetime | None = Field(default=None, description="Auction start time")
    end_time: datetime | None = Field(default=None, description="Scheduled end time")
    actual_end_time: datetime | None = Field(
        default=None, description="Actual end time (soft-close)"
    )

    # Stats
    num_items: int = Field(default=0, description="Total number of items")
    num_items_with_bids: int | None = Field(
        default=None, description="Items that received bids"
    )
    total_bids: int | None = Field(default=None, description="Total bids across all items")

    # Status
    status: str = Field(default="unknown", description="Auction status")

    # Items (populated separately)
    items: list[Item] = Field(default_factory=list, description="Items in auction")

    class Config:
        extra = "allow"

    @property
    def is_completed(self) -> bool:
        """Check if auction is completed."""
        return self.status.lower() in ("completed", "closed", "ended")


# =============================================================================
# API Response Schemas
# =============================================================================


class AuctionResponse(BaseModel):
    """Response from auction items API endpoint."""

    auction_id: int
    items: list[Item] = Field(default_factory=list)
    total_items: int | None = Field(default=None)
    page: int | None = Field(default=None)
    limit: int | None = Field(default=None)

    class Config:
        extra = "allow"


class ItemDetailResponse(BaseModel):
    """Response from item detail API endpoint (includes bid history)."""

    item: Item
    bids: list[Bid] = Field(default_factory=list)

    class Config:
        extra = "allow"


class EnrichedItemResponse(BaseModel):
    """Response from enriched item API endpoint."""

    item_id: int
    enriched_data: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# =============================================================================
# Database Models (for DuckDB storage)
# =============================================================================


class AuctionRecord(BaseModel):
    """Flattened auction record for database storage."""

    auction_id: int
    title: str
    auction_type: str | None
    city: str | None
    state: str | None
    country: str | None
    start_time: datetime | None
    end_time: datetime | None
    num_items: int
    status: str
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class ItemRecord(BaseModel):
    """Flattened item record for database storage."""

    item_id: int
    auction_id: int
    lot_number: int | None
    title: str
    description: str | None
    category: str | None
    starting_bid: float
    winning_price: float | None
    num_bids: int
    num_bidders: int | None
    primary_image_url: str | None
    end_time: datetime | None
    has_bids: bool
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class BidRecord(BaseModel):
    """Flattened bid record for database storage."""

    bid_id: str | None
    item_id: int
    auction_id: int
    bidder_id: str | None
    amount: float
    bid_time: datetime | None
    is_winning: bool
    extended_auction: bool
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
