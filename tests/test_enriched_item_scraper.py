# =============================================================================
# Tests for Enriched Item Scraper
# =============================================================================
"""
Tests for the enriched item scraper module.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.data.enriched_item_scraper import (
    EnrichedItemDataFetcher,
    ProgressTracker,
    transform_enriched_item_data,
)

# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def mock_enriched_data():
    """Sample enriched item data from API."""
    return {
        "amLotId": 7433915,
        "amAuctionId": 99941,
        "generatedDescription": {
            "title": "Antique Wooden Chair",
            "slug": "antique-wooden-chair",
            "description": "Beautiful antique wooden chair in excellent condition",
            "qualitativeDescription": "Excellent condition with minor wear",
            "brand": "Vintage Furniture Co.",
            "seriesLine": "Classic Collection",
            "condition": "Excellent",
            "working": True,
            "singleKeyItem": True,
            "numItems": 1,
            "brands": [{"name": "Vintage Furniture Co.", "confidence": 0.95}],
            "categories": [
                {"name": "Furniture", "confidence": 0.98},
                {"name": "Chairs", "confidence": 0.92},
            ],
            "items": [{"name": "Chair", "quantity": 1}],
            "attributes": [
                {"name": "Material", "value": "Wood"},
                {"name": "Style", "value": "Antique"},
            ],
            "photosTaken": ["photo1.jpg", "photo2.jpg"],
        },
    }


@pytest.fixture
def mock_multiple_enriched_items():
    """Multiple enriched items for testing batch processing."""
    return [
        {
            "amLotId": 7433915,
            "amAuctionId": 99941,
            "generatedDescription": {
                "title": "Antique Chair",
                "description": "Beautiful chair",
                "brand": "Vintage Co.",
                "condition": "Excellent",
                "working": True,
                "numItems": 1,
                "brands": [{"name": "Vintage Co."}],
                "categories": [{"name": "Furniture"}],
                "items": [{"name": "Chair"}],
                "attributes": [{"name": "Material", "value": "Wood"}],
                "photosTaken": ["photo1.jpg"],
            },
        },
        {
            "amLotId": 7433916,
            "amAuctionId": 99941,
            "generatedDescription": {
                "title": "Vintage Table",
                "description": "Solid oak table",
                "brand": None,
                "condition": "Good",
                "working": True,
                "numItems": 1,
                "brands": [],
                "categories": [{"name": "Furniture"}],
                "items": [{"name": "Table"}],
                "attributes": [],
                "photosTaken": [],
            },
        },
    ]


# =============================================================================
# Test EnrichedItemDataFetcher
# =============================================================================


def test_enriched_item_data_fetcher_initialization():
    """Test EnrichedItemDataFetcher initialization."""
    fetcher = EnrichedItemDataFetcher(rate_limit=5, max_concurrent=3, timeout=15.0)

    assert fetcher.rate_limit == 5
    assert fetcher.max_concurrent == 3
    assert fetcher.timeout == 15.0
    assert fetcher.min_interval == 0.2  # 1.0 / 5


def test_process_enriched_item_data(mock_enriched_data):
    """Test processing of raw enriched item data."""
    fetcher = EnrichedItemDataFetcher()
    result = fetcher.process_enriched_item_data(mock_enriched_data, 7433915)

    # Check basic fields
    assert result["amLotId"] == 7433915
    assert result["amAuctionId"] == 99941

    # Check generatedDescription fields
    assert result["generatedDescription_title"] == "Antique Wooden Chair"
    assert result["generatedDescription_description"] == "Beautiful antique wooden chair in excellent condition"
    assert result["generatedDescription_brand"] == "Vintage Furniture Co."
    assert result["generatedDescription_condition"] == "Excellent"
    assert result["generatedDescription_working"] is True
    assert result["generatedDescription_numItems"] == 1

    # Check nested JSON fields are stored as strings
    assert isinstance(result["generatedDescription_brands"], str)
    assert isinstance(result["generatedDescription_categories"], str)
    assert isinstance(result["generatedDescription_items"], str)
    assert isinstance(result["generatedDescription_attributes"], str)
    assert isinstance(result["generatedDescription_photosTaken"], str)


def test_process_enriched_item_data_missing_fields():
    """Test processing when some fields are missing."""
    fetcher = EnrichedItemDataFetcher()
    minimal_data = {
        "amLotId": 7433915,
        "amAuctionId": 99941,
        "generatedDescription": {
            "title": "Test Item",
            # Missing other fields
        },
    }

    result = fetcher.process_enriched_item_data(minimal_data, 7433915)

    assert result["amLotId"] == 7433915
    assert result["amAuctionId"] == 99941
    assert result["generatedDescription_title"] == "Test Item"
    assert result["generatedDescription_description"] is None
    assert result["generatedDescription_brands"] is None


def test_process_enriched_item_data_empty_generated_description():
    """Test processing when generatedDescription is missing or empty."""
    fetcher = EnrichedItemDataFetcher()
    data = {
        "amLotId": 7433915,
        "amAuctionId": 99941,
        # No generatedDescription
    }

    result = fetcher.process_enriched_item_data(data, 7433915)

    assert result["amLotId"] == 7433915
    assert result["amAuctionId"] == 99941
    # All generatedDescription fields should be None
    assert result.get("generatedDescription_title") is None
    assert result.get("generatedDescription_description") is None


# =============================================================================
# Test Data Transformation
# =============================================================================


def test_transform_enriched_item_data(mock_multiple_enriched_items):
    """Test transformation of enriched item data."""
    fetcher = EnrichedItemDataFetcher()
    processed_items = [
        fetcher.process_enriched_item_data(item, item["amLotId"])
        for item in mock_multiple_enriched_items
    ]

    df = transform_enriched_item_data(processed_items)

    # Check renaming (amLotId -> item_id, amAuctionId -> auction_id)
    assert "item_id" in df.columns
    assert "auction_id" in df.columns
    assert "amLotId" not in df.columns
    assert "amAuctionId" not in df.columns

    # Check that item_id and auction_id don't have the prefix
    assert "enriched_item_item_id" not in df.columns
    assert "enriched_item_auction_id" not in df.columns

    # Check that other columns have the enriched_item_ prefix
    for col in df.columns:
        if col not in ("item_id", "auction_id"):
            assert col.startswith("enriched_item_generatedDescription_")

    # Check data values
    assert df["item_id"].tolist() == [7433915, 7433916]
    assert df["auction_id"].tolist() == [99941, 99941]


def test_transform_enriched_item_data_empty():
    """Test transformation with empty data."""
    df = transform_enriched_item_data([])

    assert df.empty
    assert len(df) == 0


# =============================================================================
# Test Progress Tracking
# =============================================================================


def test_progress_tracker_initialization(tmp_path):
    """Test ProgressTracker initialization."""
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    assert tracker.progress_file == progress_file
    assert len(tracker.completed_ids) == 0
    assert len(tracker.failed_ids) == 0


def test_progress_tracker_mark_completed(tmp_path):
    """Test marking items as completed."""
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_completed(7433915)
    tracker.mark_completed(7433916)

    assert 7433915 in tracker.completed_ids
    assert 7433916 in tracker.completed_ids
    assert len(tracker.completed_ids) == 2


def test_progress_tracker_mark_failed(tmp_path):
    """Test marking items as failed."""
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_failed(7433915)

    assert 7433915 in tracker.failed_ids
    assert len(tracker.failed_ids) == 1


def test_progress_tracker_is_completed(tmp_path):
    """Test checking if item is completed."""
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_completed(7433915)

    assert tracker.is_completed(7433915) is True
    assert tracker.is_completed(7433916) is False


def test_progress_tracker_filter_pending(tmp_path):
    """Test filtering pending items."""
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(progress_file=progress_file)

    tracker.mark_completed(7433915)
    tracker.mark_completed(7433916)

    item_ids = [7433915, 7433916, 7433917, 7433918]
    pending = tracker.filter_pending(item_ids)

    assert pending == [7433917, 7433918]


def test_progress_tracker_save_and_load(tmp_path):
    """Test saving and loading progress."""
    progress_file = tmp_path / "progress.json"

    # Create and save progress
    tracker1 = ProgressTracker(progress_file=progress_file)
    tracker1.mark_completed(7433915)
    tracker1.mark_completed(7433916)
    tracker1.mark_failed(7433917)
    tracker1.save()

    # Load progress in new tracker
    tracker2 = ProgressTracker(progress_file=progress_file)

    assert 7433915 in tracker2.completed_ids
    assert 7433916 in tracker2.completed_ids
    assert 7433917 in tracker2.failed_ids
    assert len(tracker2.completed_ids) == 2
    assert len(tracker2.failed_ids) == 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_and_process_item_mock(mock_enriched_data):
    """Test fetch_and_process_item with mocked HTTP client."""
    with patch("httpx.AsyncClient.get") as mock_get:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_enriched_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        async with EnrichedItemDataFetcher() as fetcher:
            result = await fetcher.fetch_and_process_item(7433915)

        assert result is not None
        assert result["amLotId"] == 7433915
        assert result["amAuctionId"] == 99941
        assert "generatedDescription_title" in result


@pytest.mark.asyncio
async def test_fetch_and_process_item_404():
    """Test fetch_and_process_item when item not found."""
    with patch("httpx.AsyncClient.get") as mock_get:
        # Setup mock 404 response
        mock_response = MagicMock()
        mock_response.status_code = 404

        def raise_404():
            from httpx import HTTPStatusError
            raise HTTPStatusError(
                message="Not Found",
                request=MagicMock(),
                response=mock_response
            )

        mock_response.raise_for_status.side_effect = raise_404
        mock_get.return_value = mock_response

        async with EnrichedItemDataFetcher() as fetcher:
            result = await fetcher.fetch_and_process_item(9999999)

        # Should return None for 404
        assert result is None
