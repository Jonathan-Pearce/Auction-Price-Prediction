# =============================================================================
# Tests for Bid Scraper
# =============================================================================
"""
Tests for the bid scraper module.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.bid_scraper import (
    BidDataFetcher,
    ProgressTracker,
    transform_bid_data,
)

# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def mock_bid_data():
    """Sample bid data from API."""
    return [
        {
            "time_of_bid": "2024-01-05T17:45:00",
            "amount": 45.0,
            "isproxy": False,
        },
        {
            "time_of_bid": "2024-01-05T17:50:00",
            "amount": 50.0,
            "isproxy": True,
        },
        {
            "time_of_bid": "2024-01-05T17:55:00",
            "amount": 55.0,
            "isproxy": False,
        },
    ]


@pytest.fixture
def mock_api_response():
    """Mock API response structure with bid history."""
    return {
        "auction": {
            "items": {
                0: {
                    "id": 7433850,
                    "title": "Antique Chair",
                    "bid_history": [
                        {
                            "time_of_bid": "2024-01-05T17:45:00",
                            "amount": 45.0,
                            "isproxy": False,
                        },
                        {
                            "time_of_bid": "2024-01-05T17:50:00",
                            "amount": 50.0,
                            "isproxy": True,
                        },
                        {
                            "time_of_bid": "2024-01-05T17:55:00",
                            "amount": 55.0,
                            "isproxy": False,
                        },
                    ],
                }
            }
        }
    }


# =============================================================================
# Data Processing Tests
# =============================================================================


def test_process_bid_data():
    """Test bid data processing."""
    fetcher = BidDataFetcher()

    bids = [
        {
            "time_of_bid": "2024-01-05T17:45:00",
            "amount": 45.0,
            "isproxy": False,
        },
        {
            "time_of_bid": "2024-01-05T17:50:00",
            "amount": 50.0,
            "isproxy": True,
        },
        {
            "time_of_bid": "2024-01-05T17:55:00",
            "amount": 55.0,
            "isproxy": False,
        },
    ]

    result = fetcher.process_bid_data(bids, 103293, 7433850)

    assert len(result) == 3
    assert result[0]["auction_id"] == 103293
    assert result[0]["item_id"] == 7433850
    assert result[0]["id"] == 3  # First bid gets count
    assert result[1]["id"] == 2  # Second bid
    assert result[2]["id"] == 1  # Last bid gets 1
    assert result[0]["count"] == 3
    assert result[0]["amount"] == 45.0


def test_process_bid_data_no_bids():
    """Test processing item with no bids."""
    fetcher = BidDataFetcher()

    bids = []

    result = fetcher.process_bid_data(bids, 103293, 7433850)

    assert len(result) == 0


def test_transform_bid_data(mock_bid_data):
    """Test bid data transformation with prefix."""
    # First process the data to add auction_id, item_id, and bid_id
    fetcher = BidDataFetcher()
    processed_bids = fetcher.process_bid_data(mock_bid_data, 103293, 7433850)

    # Then transform
    df = transform_bid_data(processed_bids)

    # Check that all columns have 'bid_' prefix except auction_id and item_id
    non_prefixed_cols = [col for col in df.columns if not col.startswith("bid_")]
    assert set(non_prefixed_cols) == {
        "auction_id",
        "item_id",
    }, f"Expected only auction_id and item_id without prefix, got: {non_prefixed_cols}"

    # Check expected columns exist
    assert "bid_id" in df.columns
    assert "bid_count" in df.columns
    assert "auction_id" in df.columns  # No prefix for auction_id
    assert "item_id" in df.columns  # No prefix for item_id
    assert "bid_time" in df.columns  # Renamed from time_of_bid to time, then prefixed
    assert "bid_amount" in df.columns
    assert (
        "bid_is_proxy" in df.columns
    )  # Renamed from isproxy to is_proxy, then prefixed

    # Check data integrity
    assert len(df) == 3
    assert df["bid_id"].iloc[0] == 3  # First bid has highest id
    assert df["bid_id"].iloc[1] == 2
    assert df["bid_id"].iloc[2] == 1  # Last bid has id of 1


def test_transform_bid_data_empty():
    """Test transformation with empty list."""
    df = transform_bid_data([])

    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_bid_id_counting_order():
    """Test that bid_id counts downward correctly."""
    fetcher = BidDataFetcher()

    bids = [
        {"time_of_bid": "2024-01-05T17:45:00", "amount": 10.0, "isproxy": False},
        {"time_of_bid": "2024-01-05T17:46:00", "amount": 15.0, "isproxy": False},
        {"time_of_bid": "2024-01-05T17:47:00", "amount": 20.0, "isproxy": False},
        {"time_of_bid": "2024-01-05T17:48:00", "amount": 25.0, "isproxy": False},
        {"time_of_bid": "2024-01-05T17:49:00", "amount": 30.0, "isproxy": False},
    ]

    result = fetcher.process_bid_data(bids, 103293, 7433850)

    # First bid should have id = count (5)
    assert result[0]["id"] == 5
    assert result[0]["count"] == 5

    # Last bid should have id = 1
    assert result[-1]["id"] == 1
    assert result[-1]["count"] == 5


# =============================================================================
# Progress Tracker Tests
# =============================================================================


def test_progress_tracker_initialization(tmp_path):
    """Test progress tracker initialization."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    assert len(tracker.completed_items) == 0
    assert len(tracker.failed_items) == 0


def test_progress_tracker_mark_completed(tmp_path):
    """Test marking items as completed."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_completed(103293, 7433850)
    tracker.mark_completed(103293, 7433851)

    assert (103293, 7433850) in tracker.completed_items
    assert (103293, 7433851) in tracker.completed_items
    assert len(tracker.completed_items) == 2


def test_progress_tracker_mark_failed(tmp_path):
    """Test marking items as failed."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_failed(103293, 7433852)

    assert (103293, 7433852) in tracker.failed_items
    assert (103293, 7433852) not in tracker.completed_items


def test_progress_tracker_filter_pending(tmp_path):
    """Test filtering pending items."""
    progress_file = tmp_path / "test_progress.json"
    tracker = ProgressTracker(progress_file)

    tracker.mark_completed(103293, 7433850)
    tracker.mark_completed(103293, 7433851)

    all_pairs = [
        (103293, 7433850),
        (103293, 7433851),
        (103293, 7433852),
        (103293, 7433853),
    ]
    pending = tracker.filter_pending(all_pairs)

    assert pending == [(103293, 7433852), (103293, 7433853)]
    assert (103293, 7433850) not in pending
    assert (103293, 7433851) not in pending


def test_progress_tracker_persistence(tmp_path):
    """Test progress tracker saves and loads correctly."""
    progress_file = tmp_path / "test_progress.json"

    # Create tracker and mark some items
    tracker1 = ProgressTracker(progress_file)
    tracker1.mark_completed(103293, 7433850)
    tracker1.mark_completed(103293, 7433851)
    tracker1.mark_failed(103293, 7433852)
    tracker1.save()

    # Load progress in new tracker
    tracker2 = ProgressTracker(progress_file)

    assert tracker2.completed_items == tracker1.completed_items
    assert tracker2.failed_items == tracker1.failed_items


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_bid_data_fetcher_context_manager():
    """Test BidDataFetcher context manager."""
    async with BidDataFetcher() as fetcher:
        assert fetcher._client is not None

    # After exit, client should be closed
    assert fetcher._client is None or fetcher._client.is_closed


@pytest.mark.asyncio
async def test_fetch_item_bids_dict_response(mock_api_response):
    """Test fetching bids with dict response structure."""
    async with BidDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            bids = await fetcher.fetch_item_bids(103293, 7433850)

            assert len(bids) == 3
            assert bids[0]["amount"] == 45.0
            assert bids[1]["amount"] == 50.0
            assert bids[2]["amount"] == 55.0


@pytest.mark.asyncio
async def test_fetch_item_bids_list_response(mock_bid_data):
    """Test fetching bids with list response structure."""
    async with BidDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = [{"bid_history": mock_bid_data}]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            bids = await fetcher.fetch_item_bids(103293, 7433850)

            assert len(bids) == 3
            assert bids[0]["amount"] == 45.0


@pytest.mark.asyncio
async def test_fetch_and_process_item(mock_api_response):
    """Test fetching and processing complete item bid history."""
    async with BidDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await fetcher.fetch_and_process_item(103293, 7433850)

            assert result is not None
            assert len(result) == 3
            assert result[0]["auction_id"] == 103293
            assert result[0]["item_id"] == 7433850
            assert result[0]["id"] == 3  # First bid
            assert result[2]["id"] == 1  # Last bid


@pytest.mark.asyncio
async def test_fetch_item_with_no_bids():
    """Test fetching item that has no bids."""
    async with BidDataFetcher() as fetcher:
        # Mock the HTTP request
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "auction": {"items": {0: {"id": 7433850, "bid_history": []}}}
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await fetcher.fetch_and_process_item(103293, 7433850)

            assert result is not None
            assert len(result) == 0  # No bids


# =============================================================================
# Field Extraction Tests
# =============================================================================


def test_bid_field_extraction():
    """Test extraction of bid fields."""
    fetcher = BidDataFetcher()
    bids = [
        {
            "time_of_bid": "2024-01-05T17:45:00",
            "amount": 45.0,
            "isproxy": False,
            "extra_field": "should_be_ignored",
        }
    ]

    result = fetcher.process_bid_data(bids, 103293, 7433850)

    assert len(result) == 1
    assert "time_of_bid" in result[0]
    assert "amount" in result[0]
    assert "isproxy" in result[0]
    assert "extra_field" not in result[0]  # Extra fields not in config are ignored


def test_missing_optional_fields():
    """Test handling of missing optional fields."""
    fetcher = BidDataFetcher()
    bids = [
        {
            "amount": 45.0,
            # Missing time_of_bid and isproxy
        }
    ]

    result = fetcher.process_bid_data(bids, 103293, 7433850)

    assert len(result) == 1
    assert result[0]["auction_id"] == 103293
    assert result[0]["item_id"] == 7433850
    assert result[0]["amount"] == 45.0
    assert result[0].get("time_of_bid") is None
    assert result[0].get("isproxy") is None


def test_field_renaming_in_transform():
    """Test that field renaming works correctly."""
    fetcher = BidDataFetcher()
    bids = [
        {
            "time_of_bid": "2024-01-05T17:45:00",
            "amount": 45.0,
            "isproxy": True,
        }
    ]

    processed = fetcher.process_bid_data(bids, 103293, 7433850)
    df = transform_bid_data(processed)

    # Check renamed fields
    assert "bid_time" in df.columns  # time_of_bid -> time -> bid_time
    assert "bid_is_proxy" in df.columns  # isproxy -> is_proxy -> bid_is_proxy
    assert "time_of_bid" not in df.columns
    assert "isproxy" not in df.columns
