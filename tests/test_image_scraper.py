# =============================================================================
# Tests for Image Scraper
# =============================================================================
"""
Tests for the image scraper module.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data import scraper_config as config
from src.data.image_scraper import (
    ImageDataFetcher,
    ImageEmbeddingExtractor,
    ProgressTracker,
    transform_image_data,
)

# =============================================================================
# Test Data
# =============================================================================


@pytest.fixture
def sample_image_bytes():
    """Generate sample image bytes for testing."""
    # Create a simple test image
    image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes()


@pytest.fixture
def mock_auction_response():
    """Mock API response with image data."""
    return {
        "auction": {
            "items": {
                0: {
                    "id": 1001,
                    "title": "Antique Chair",
                    "images": [
                        {"url": "https://example.com/img1.jpg"},
                        {"url": "https://example.com/img2.jpg"},
                    ],
                },
                1: {
                    "id": 1002,
                    "title": "Vintage Table",
                    "images": ["https://example.com/img3.jpg"],
                },
            }
        }
    }


@pytest.fixture
def mock_image_records():
    """Sample image records with embeddings."""
    return [
        {
            "auction_id": 99941,
            "item_id": 1001,
            "image_index": 0,
            "image_url": "https://example.com/img1.jpg",
            "embedding": list(np.random.randn(576).astype(np.float32)),
        },
        {
            "auction_id": 99941,
            "item_id": 1001,
            "image_index": 1,
            "image_url": "https://example.com/img2.jpg",
            "embedding": list(np.random.randn(576).astype(np.float32)),
        },
    ]


# =============================================================================
# Image Embedding Extractor Tests
# =============================================================================


def test_embedding_extractor_initialization():
    """Test embedding extractor initialization."""
    extractor = ImageEmbeddingExtractor()

    assert extractor.embedding_dim == 576
    assert extractor.target_size == (224, 224)
    assert len(extractor.mean) == 3
    assert len(extractor.std) == 3


def test_preprocess_image_shape(sample_image_bytes):
    """Test image preprocessing produces correct shape."""
    extractor = ImageEmbeddingExtractor()

    result = extractor.preprocess_image(sample_image_bytes)

    assert result is not None
    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32


def test_preprocess_image_wide_image():
    """Test preprocessing wide image maintains aspect ratio."""
    extractor = ImageEmbeddingExtractor()

    # Create wide image (300x100)
    wide_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", wide_image)

    result = extractor.preprocess_image(encoded.tobytes())

    assert result.shape == (1, 3, 224, 224)


def test_preprocess_image_tall_image():
    """Test preprocessing tall image maintains aspect ratio."""
    extractor = ImageEmbeddingExtractor()

    # Create tall image (100x300)
    tall_image = np.random.randint(0, 255, (300, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", tall_image)

    result = extractor.preprocess_image(encoded.tobytes())

    assert result.shape == (1, 3, 224, 224)


def test_preprocess_image_invalid_bytes():
    """Test preprocessing with invalid image bytes."""
    extractor = ImageEmbeddingExtractor()

    result = extractor.preprocess_image(b"invalid image data")

    assert result is None


def test_extract_embedding_shape(sample_image_bytes):
    """Test embedding extraction produces correct shape."""
    extractor = ImageEmbeddingExtractor()

    embedding = extractor.extract_embedding(sample_image_bytes)

    assert embedding is not None
    assert embedding.shape == (576,)
    assert embedding.dtype == np.float32


def test_extract_embedding_invalid_image():
    """Test embedding extraction with invalid image."""
    extractor = ImageEmbeddingExtractor()

    embedding = extractor.extract_embedding(b"invalid data")

    assert embedding is None


# =============================================================================
# Image Data Fetcher Tests
# =============================================================================


def test_extract_image_records_dict_format():
    """Test extracting image records from dict format."""
    fetcher = ImageDataFetcher()

    item = {
        "id": 1001,
        "images": [
            {"url": "https://example.com/img1.jpg"},
            {"url": "https://example.com/img2.jpg"},
        ],
    }

    records = fetcher.extract_image_records(item, 99941)

    assert len(records) == 2
    assert records[0]["auction_id"] == 99941
    assert records[0]["item_id"] == 1001
    assert records[0]["image_index"] == 0
    assert records[0]["image_url"] == "https://example.com/img1.jpg"


def test_extract_image_records_string_format():
    """Test extracting image records from string URL format."""
    fetcher = ImageDataFetcher()

    item = {
        "id": 1002,
        "images": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
    }

    records = fetcher.extract_image_records(item, 99941)

    assert len(records) == 2
    assert records[0]["image_url"] == "https://example.com/img1.jpg"
    assert records[1]["image_url"] == "https://example.com/img2.jpg"


def test_extract_image_records_no_images():
    """Test extracting from item with no images."""
    fetcher = ImageDataFetcher()

    item = {"id": 1003, "title": "No Images Item"}

    records = fetcher.extract_image_records(item, 99941)

    assert len(records) == 0


def test_extract_image_records_empty_images():
    """Test extracting from item with empty images list."""
    fetcher = ImageDataFetcher()

    item = {"id": 1004, "images": []}

    records = fetcher.extract_image_records(item, 99941)

    assert len(records) == 0


# =============================================================================
# Data Transformation Tests
# =============================================================================


def test_transform_image_data(mock_image_records):
    """Test image data transformation."""
    df = transform_image_data(mock_image_records)

    assert len(df) == 2
    assert "auction_id" in df.columns
    assert "item_id" in df.columns
    assert "image_embedding" in df.columns
    assert "image_url" in df.columns


def test_transform_image_data_empty():
    """Test transformation with empty list."""
    df = transform_image_data([])

    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_transform_image_data_preserves_values(mock_image_records):
    """Test that transformation preserves data values."""
    df = transform_image_data(mock_image_records)

    assert df["auction_id"].iloc[0] == 99941
    assert df["item_id"].iloc[0] == 1001
    assert df["image_index"].iloc[0] == 0


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


def test_progress_tracker_persistence(tmp_path):
    """Test progress tracker saves and loads correctly."""
    progress_file = tmp_path / "test_progress.json"

    # Create tracker and mark some items
    tracker1 = ProgressTracker(progress_file)
    tracker1.mark_completed(99941)
    tracker1.mark_completed(99942)
    tracker1.mark_failed(99943)
    tracker1.save()

    # Load in new tracker
    tracker2 = ProgressTracker(progress_file)

    assert tracker2.completed_ids == tracker1.completed_ids
    assert tracker2.failed_ids == tracker1.failed_ids


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_image_data_fetcher_context_manager():
    """Test ImageDataFetcher context manager."""
    async with ImageDataFetcher() as fetcher:
        assert fetcher._client is not None

    assert fetcher._client is None or fetcher._client.is_closed


@pytest.mark.asyncio
async def test_fetch_auction_items_mock(mock_auction_response):
    """Test fetching auction items with mock response."""
    async with ImageDataFetcher() as fetcher:
        with patch.object(fetcher.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_auction_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            items = await fetcher.fetch_auction_items(99941)

            assert len(items) == 2
            assert items[0]["id"] == 1001
            assert items[1]["id"] == 1002


# =============================================================================
# Embedding Dimension Tests
# =============================================================================


def test_embedding_dimension_matches_config():
    """Test that embedding dimension matches configuration."""
    extractor = ImageEmbeddingExtractor()

    assert extractor.embedding_dim == config.IMAGE_MODEL_CONFIG["embedding_dim"]
    assert extractor.embedding_dim == 576


def test_model_input_size_matches_config():
    """Test that model input size matches configuration."""
    extractor = ImageEmbeddingExtractor()

    assert extractor.target_size[0] == config.IMAGE_PREPROCESSING_CONFIG["max_dimension"]
