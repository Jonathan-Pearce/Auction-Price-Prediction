# =============================================================================
# Tests for Item Scraper
# =============================================================================
"""
Tests for the item scraper module.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.item_scraper import (
    ItemDataFetcher,
    ProgressTracker,
    transform_item_data,
)

# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def mock_item_data():
    """Sample item data from API."""
    return [
        {
            "id": 1001,
            "auction_id": 99941,
            "title": "Antique Chair",
            "description": "Beautiful wooden chair",
            "viewed": 150,
            "starting_bid": 10.0,
            "current_bid": 45.0,
            "proxy_bid": 50.0,
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-05T18:00:00",
            "bid_count": 8,
            "bidding_extended": False,
            "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
        },
        {
            "id": 1002,
            "auction_id": 99941,
            "title": "Vintage Table",
            "description": "Solid oak table",
            "viewed": 200,
            "starting_bid": 25.0,
            "current_bid": 0.0,
            "proxy_bid": 0.0,
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-05T18:00:00",
            "bid_count": 0,
            "bidding_extended": False,
            "images": ["img1.jpg"],
        },
    ]


@pytest.fixture
def mock_api_response():
    """Mock API response structure."""
    return {
        "auction": {
            "items": {
                0: {
                    "id": 1001,
                    "title": "Antique Chair",
                    "description": "Beautiful wooden chair",
                    "viewed": 150,
                    "starting_bid": 10.0,
                    "current_bid": 45.0,
                    "proxy_bid": 50.0,
                    "start_time": "2024-01-01T10:00:00",
                    "end_time": "2024-01-05T18:00:00",
                    "bid_count": 8,
                    "bidding_extended": False,
                    "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
                },
                1: {
                    "id": 1002,
                    "title": "Vintage Table",
                    "description": "Solid oak table",
                    "viewed": 200,
                    "starting_bid": 25.0,
                    "current_bid": 0.0,
                    "proxy_bid": 0.0,
                    "start_time": "2024-01-01T10:00:00",
                    "end_time": "2024-01-05T18:00:00",
                    "bid_count": 0,
                    "bidding_extended": False,
                    "images": ["img1.jpg"],
                },
            }
        }
    }


# =============================================================================
# Data Processing Tests
# =============================================================================


def test_process_item_data():
    """Test item data processing."""
    fetcher = ItemDataFetcher()

    item = {
        "id": 1001,
        "title": "Antique Chair",
        "description": "Beautiful wooden chair",
        "viewed": 150,
        "starting_bid": 10.0,
        "current_bid": 45.0,
        "proxy_bid": 50.0,
        "start_time": "2024-01-01T10:00:00",
        "end_time": "2024-01-05T18:00:00",
        "bid_count": 8,
        "bidding_extended": False,
        "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
    }

    result = fetcher.process_item_data(item, 99941)

    assert result["auction_id"] == 99941
    assert result["id"] == 1001
    assert result["title"] == "Antique Chair"
    assert result["number_of_images"] == 3


def test_process_item_data_zero_bid():
    """Test processing item with no bids."""
    fetcher = ItemDataFetcher()

    item = {
        "id": 1002,
        "title": "Vintage Table",
        "viewed": 200,
        "starting_bid": 25.0,
        "current_bid": 0.0,
        "bid_count": 0,
        "images": ["img1.jpg"],
    }

    result = fetcher.process_item_data(item, 99941)

    assert result["current_bid"] == 0.0
    assert result["bid_count"] == 0
    assert result["number_of_images"] == 1


def test_transform_item_data(mock_item_data):
    """Test item data transformation with prefix."""
    df = transform_item_data(mock_item_data)

    # Check that all columns have 'item_' prefix except auction_id
    non_prefixed_cols = [col for col in df.columns if not col.startswith("item_")]
    assert non_prefixed_cols == [
        "auction_id"
    ], f"Expected only auction_id without prefix, got: {non_prefixed_cols}"

    # Check expected columns exist
    assert "item_id" in df.columns
    assert "auction_id" in df.columns  # No prefix for auction_id
    assert "item_title" in df.columns
    assert "item_description" in df.columns
    assert "item_viewed" in df.columns
    assert "item_starting_bid" in df.columns
    assert "item_current_bid" in df.columns
    assert "item_bid_count" in df.columns

    # Check data integrity
    assert len(df) == 2
    assert df["item_title"].iloc[0] == "Antique Chair"
    assert df["item_title"].iloc[1] == "Vintage Table"


def test_transform_item_data_empty():
    """Test transformation with empty list."""
    df = transform_item_data([])

    assert df.empty
    assert isinstance(df, pd.DataFrame)


# =============================================================================
# Progress Tracker Tests
# =============================================================================


def test_progress_tracker_initialization(tmp_path):
    """Test progress tracker initialization."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    assert len(tracker.completed_ids) == 0
    assert len(tracker.failed_ids) == 0


def test_progress_tracker_mark_completed(tmp_path):
    """Test marking auctions as completed."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_completed(99941)
    tracker.mark_completed(99942)

    assert 99941 in tracker.completed_ids
    assert 99942 in tracker.completed_ids
    assert len(tracker.completed_ids) == 2


def test_progress_tracker_mark_failed(tmp_path):
    """Test marking auctions as failed."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_failed(99943)

    assert 99943 in tracker.failed_ids
    assert 99943 not in tracker.completed_ids


def test_progress_tracker_filter_pending(tmp_path):
    """Test filtering pending auctions."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_completed(99941)
    tracker.mark_completed(99942)

    all_ids = [99941, 99942, 99943, 99944]
    pending = tracker.filter_pending(all_ids)

    assert pending == [99943, 99944]
    assert 99941 not in pending
    assert 99942 not in pending


def test_progress_tracker_persistence(tmp_path):
    """Test progress tracker saves and loads correctly."""
    progress_file = tmp_path / "test_progress.json"

    # Create tracker and mark some items
    tracker1 = ProgressTracker(progress_file)
    tracker1.mark_completed(99941)
    tracker1.mark_completed(99942)
    tracker1.mark_failed(99943)
    tracker1.save()

    # Load progress in new tracker
    tracker2 = ProgressTracker(progress_file)

    assert tracker2.completed_ids == tracker1.completed_ids
    assert tracker2.failed_ids == tracker1.failed_ids


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_item_data_fetcher_context_manager():
    """Test ItemDataFetcher context manager."""
    async with ItemDataFetcher() as fetcher:
        assert fetcher._client is not None

    # After exit, client should be closed
    assert fetcher._client is None or fetcher._client.is_closed


@pytest.mark.asyncio
async def test_fetch_auction_items_dict_response(mock_api_response):
    """Test fetching items with dict response structure."""
    async with ItemDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            items = await fetcher.fetch_auction_items(99941)

            assert len(items) == 2
            assert items[0]["id"] == 1001
            assert items[1]["id"] == 1002


@pytest.mark.asyncio
async def test_fetch_auction_items_list_response(mock_item_data):
    """Test fetching items with list response structure."""
    async with ItemDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_item_data
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            items = await fetcher.fetch_auction_items(99941)

            assert len(items) == 2
            assert items[0]["id"] == 1001


@pytest.mark.asyncio
async def test_fetch_and_process_auction(mock_item_data):
    """Test fetching and processing a complete auction."""
    async with ItemDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_item_data
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await fetcher.fetch_and_process_auction(99941)

            assert result is not None
            assert len(result) == 2
            assert result[0]["auction_id"] == 99941
            assert result[0]["number_of_images"] == 3
            assert result[1]["number_of_images"] == 1


# =============================================================================
# Field Extraction Tests
# =============================================================================


def test_number_of_images_list():
    """Test image counting with list of images."""
    fetcher = ItemDataFetcher()
    item = {"images": ["img1.jpg", "img2.jpg", "img3.jpg"]}

    result = fetcher.process_item_data(item, 99941)

    assert result["number_of_images"] == 3


def test_number_of_images_int():
    """Test image counting when images is an integer."""
    fetcher = ItemDataFetcher()
    item = {"images": 5}

    result = fetcher.process_item_data(item, 99941)

    assert result["number_of_images"] == 5


def test_number_of_images_missing():
    """Test image counting when images field is missing."""
    fetcher = ItemDataFetcher()
    item = {"id": 1001, "title": "Test Item"}

    result = fetcher.process_item_data(item, 99941)

    assert result["number_of_images"] == 0


def test_missing_optional_fields():
    """Test handling of missing optional fields."""
    fetcher = ItemDataFetcher()
    item = {
        "id": 1001,
        "title": "Test Item",
        # Missing description, viewed, etc.
    }

    result = fetcher.process_item_data(item, 99941)

    assert result["auction_id"] == 99941
    assert result["id"] == 1001
    assert result["title"] == "Test Item"
    assert result.get("description") is None
    assert result.get("viewed") is None
