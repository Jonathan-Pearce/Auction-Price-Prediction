# =============================================================================
# Enriched Auction Scraper Tests
# =============================================================================
"""
Unit tests for the enriched auction scraper module.

This module tests the enriched auction data fetching, processing, and
transformation logic, as well as progress tracking functionality.
"""


import pandas as pd
import pytest

from src.data.enriched_auction_scraper import (
    EnrichedAuctionDataFetcher,
    ProgressTracker,
    transform_enriched_auction_data,
)

# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def mock_enriched_auction_data():
    """Sample enriched auction data from API."""
    return [
        {
            "amAuctionId": 103293,
            "type": "estate_sale",
            "category": "Home & Garden",
            "displayRegion": "Toronto, ON",
            "approxLocation": {
                "city": "Toronto",
                "countryCode": "CA",
                "regionCode": "ON",
                "postalCode": "M5V",
                "latLng": {"lat": 43.6426, "lng": -79.3871},
            },
        },
        {
            "amAuctionId": 99941,
            "type": "moving_sale",
            "category": "Furniture",
            "displayRegion": "Ottawa, ON",
            "approxLocation": {
                "city": "Ottawa",
                "countryCode": "CA",
                "regionCode": "ON",
                "postalCode": "K1A",
                "latLng": {"lat": 45.4215, "lng": -75.6972},
            },
        },
    ]


@pytest.fixture
def mock_processed_enriched_data():
    """Sample processed enriched auction data."""
    return [
        {
            "auction_id": 103293,
            "type": "estate_sale",
            "category": "Home & Garden",
            "displayRegion": "Toronto, ON",
            "approxLocation_city": "Toronto",
            "approxLocation_countryCode": "CA",
            "approxLocation_regionCode": "ON",
            "approxLocation_postalCode": "M5V",
            "approxLocation_lat": 43.6426,
            "approxLocation_lng": -79.3871,
        },
        {
            "auction_id": 99941,
            "type": "moving_sale",
            "category": "Furniture",
            "displayRegion": "Ottawa, ON",
            "approxLocation_city": "Ottawa",
            "approxLocation_countryCode": "CA",
            "approxLocation_regionCode": "ON",
            "approxLocation_postalCode": "K1A",
            "approxLocation_lat": 45.4215,
            "approxLocation_lng": -75.6972,
        },
    ]


# =============================================================================
# Data Processing Tests
# =============================================================================


def test_process_enriched_auction_data(mock_enriched_auction_data):
    """Test processing of raw API response."""
    fetcher = EnrichedAuctionDataFetcher()
    api_response = mock_enriched_auction_data[0]

    result = fetcher.process_enriched_auction_data(103293, api_response)

    # Check auction_id is set correctly
    assert result["auction_id"] == 103293

    # Check top-level fields
    assert result["type"] == "estate_sale"
    assert result["category"] == "Home & Garden"
    assert result["displayRegion"] == "Toronto, ON"

    # Check flattened location fields
    assert result["approxLocation_city"] == "Toronto"
    assert result["approxLocation_countryCode"] == "CA"
    assert result["approxLocation_regionCode"] == "ON"
    assert result["approxLocation_postalCode"] == "M5V"

    # Check lat/lng extraction
    assert result["approxLocation_lat"] == 43.6426
    assert result["approxLocation_lng"] == -79.3871


def test_process_enriched_auction_data_missing_location():
    """Test processing when location data is missing."""
    fetcher = EnrichedAuctionDataFetcher()
    api_response = {
        "amAuctionId": 103293,
        "type": "estate_sale",
        "category": "Home & Garden",
        "displayRegion": "Toronto, ON",
        # approxLocation is missing
    }

    result = fetcher.process_enriched_auction_data(103293, api_response)

    # Should still have auction_id and other fields
    assert result["auction_id"] == 103293
    assert result["type"] == "estate_sale"

    # Location fields should be None
    assert result["approxLocation_city"] is None
    assert result["approxLocation_lat"] is None


# =============================================================================
# Transformation Tests
# =============================================================================


def test_transform_enriched_auction_data(mock_processed_enriched_data):
    """Test transformation (renaming and prefixing)."""
    df = transform_enriched_auction_data(mock_processed_enriched_data)

    # Check DataFrame shape
    assert df.shape == (2, 10)

    # Check auction_id doesn't have prefix
    assert "auction_id" in df.columns
    assert "enriched_auction_auction_id" not in df.columns

    # Check other fields have enriched_auction_ prefix
    expected_columns = [
        "auction_id",
        "enriched_auction_type",
        "enriched_auction_category",
        "enriched_auction_displayRegion",
        "enriched_auction_approxLocation_city",
        "enriched_auction_approxLocation_countryCode",
        "enriched_auction_approxLocation_regionCode",
        "enriched_auction_approxLocation_postalCode",
        "enriched_auction_approxLocation_lat",
        "enriched_auction_approxLocation_lng",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # Check values are preserved
    assert df.loc[0, "auction_id"] == 103293
    assert df.loc[0, "enriched_auction_type"] == "estate_sale"
    assert df.loc[0, "enriched_auction_approxLocation_city"] == "Toronto"
    assert df.loc[0, "enriched_auction_approxLocation_lat"] == 43.6426

    assert df.loc[1, "auction_id"] == 99941
    assert df.loc[1, "enriched_auction_approxLocation_city"] == "Ottawa"


def test_transform_enriched_auction_data_empty():
    """Test transformation with empty input."""
    df = transform_enriched_auction_data([])

    assert df.empty
    assert isinstance(df, pd.DataFrame)


# =============================================================================
# Progress Tracker Tests
# =============================================================================


def test_progress_tracker_initialization(tmp_path):
    """Test progress tracker initialization."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    assert tracker.completed_ids == set()
    assert tracker.failed_ids == set()


def test_progress_tracker_mark_completed(tmp_path):
    """Test marking auctions as completed."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_completed(103293)
    tracker.mark_completed(99941)

    assert 103293 in tracker.completed_ids
    assert 99941 in tracker.completed_ids
    assert tracker.is_completed(103293)


def test_progress_tracker_filter_pending(tmp_path):
    """Test filtering pending auctions."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_completed(103293)

    auction_ids = [103293, 99941, 99942]
    pending = tracker.filter_pending(auction_ids)

    assert 103293 not in pending
    assert 99941 in pending
    assert 99942 in pending
    assert len(pending) == 2


def test_progress_tracker_save_and_load(tmp_path):
    """Test saving and loading progress."""
    progress_file = tmp_path / "test_progress.json"

    # Create and save progress
    tracker1 = ProgressTracker(progress_file=progress_file)
    tracker1.mark_completed(103293)
    tracker1.mark_failed(99941)
    tracker1.save()

    # Load progress in new tracker
    tracker2 = ProgressTracker(progress_file=progress_file)

    assert 103293 in tracker2.completed_ids
    assert 99941 in tracker2.failed_ids
