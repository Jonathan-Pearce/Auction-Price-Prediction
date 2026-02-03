# =============================================================================
# Tests for Auction Feature Engineering Pipeline
# =============================================================================
"""
Tests for the auction-level feature engineering pipeline.
"""

import pandas as pd
import pytest

from src.features_auction import (
    add_auction_length_features,
    add_auction_partner_feature,
    add_geospatial_features,
    add_normalized_auction_totals,
    add_pickup_window_features,
    encode_categorical_features,
    select_final_columns,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_auction_df():
    """Create sample auction data for testing."""
    return pd.DataFrame(
        {
            "auction_id": [1, 2, 3],
            "auction_starts": [
                "2024-01-01T10:00:00-05:00",
                "2024-02-15T14:00:00-05:00",
                "2024-03-20T09:00:00-04:00",
            ],
            "auction_ends": [
                "2024-01-05T18:00:00-05:00",
                "2024-02-20T14:00:00-05:00",
                "2024-03-25T09:00:00-04:00",
            ],
            "auction_last_item_closes": [
                "2024-01-05T18:30:00-05:00",
                "2024-02-20T14:15:00-05:00",
                "2024-03-25T09:45:00-04:00",
            ],
            "auction_item_count": [50, 100, 25],
            "auction_total_viewed": [5000, 20000, 2500],
            "auction_total_winning_price": [2500.0, 10000.0, 1000.0],
            "auction_total_bid_count": [500, 2000, 250],
            "auction_total_images": [300, 800, 150],
            "auction_partner_url": ["https://partner.com", "", None],
            "auction_removal_info": [
                "<strong>Pickup: Friday, March 13 EDT, 4PM - 7PM</strong>",
                "<strong>Pickup: Saturday, March 14 EDT, 9AM - 12 NOON</strong>",
                None,
            ],
        }
    )


@pytest.fixture
def sample_enriched_df():
    """Create sample enriched auction data for testing."""
    return pd.DataFrame(
        {
            "auction_id": [1, 2, 3],
            "enriched_auction_category": ["Estate_Sale", "Moving", None],
            "enriched_auction_type": ["maxsoldManaged", "sellerManaged", None],
            "enriched_auction_displayRegion": ["GTA", "Ottawa", "Kingston"],
            "enriched_auction_approxLocation_city": ["Toronto", "Ottawa", "Kingston"],
            "enriched_auction_approxLocation_countryCode": ["ca", "ca", "ca"],
            "enriched_auction_approxLocation_regionCode": ["on", "on", "on"],
            "enriched_auction_approxLocation_postalCode": ["M5V1A1", "K1A0B1", "K7L3M9"],
            "enriched_auction_approxLocation_lat": [43.6532, 45.4215, 44.2312],
            "enriched_auction_approxLocation_lng": [-79.3832, -75.6972, -76.4860],
        }
    )


# =============================================================================
# Test Auction Length Features
# =============================================================================


class TestAuctionLengthFeatures:
    """Test auction length feature engineering."""

    def test_auction_length_hours_calculated(self, sample_auction_df):
        """Test that auction length in hours is calculated correctly."""
        result = add_auction_length_features(sample_auction_df)

        assert "auction_length_hours" in result.columns
        # First auction: Jan 1 10:00 to Jan 5 18:00 = 4 days 8 hours = 104 hours
        assert abs(result["auction_length_hours"].iloc[0] - 104) < 0.1

    def test_actual_length_calculated(self, sample_auction_df):
        """Test that actual auction length accounts for extensions."""
        result = add_auction_length_features(sample_auction_df)

        assert "auction_actual_length_hours" in result.columns
        # First auction: Jan 1 10:00 to Jan 5 18:30 = 104.5 hours
        assert abs(result["auction_actual_length_hours"].iloc[0] - 104.5) < 0.1

    def test_extension_hours_calculated(self, sample_auction_df):
        """Test that extension hours are calculated."""
        result = add_auction_length_features(sample_auction_df)

        assert "auction_extension_hours" in result.columns
        # First auction: 104.5 - 104 = 0.5 hours extension
        assert abs(result["auction_extension_hours"].iloc[0] - 0.5) < 0.1


# =============================================================================
# Test Pickup Window Features
# =============================================================================


class TestPickupWindowFeatures:
    """Test pickup window feature engineering."""

    def test_pickup_features_added(self, sample_auction_df):
        """Test that pickup window features are added."""
        result = add_pickup_window_features(sample_auction_df)

        expected_cols = [
            "auction_num_pickup_windows",
            "auction_total_pickup_hours",
            "auction_first_pickup_start_hour",
            "auction_last_pickup_end_hour",
            "auction_pickup_day_of_week",
            "auction_has_category_pickup_windows",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_pickup_values_correct(self, sample_auction_df):
        """Test that pickup values are extracted correctly."""
        result = add_pickup_window_features(sample_auction_df)

        # First auction: 4PM - 7PM = 3 hours, Friday
        assert result["auction_num_pickup_windows"].iloc[0] == 1
        assert result["auction_total_pickup_hours"].iloc[0] == 3.0
        assert result["auction_pickup_day_of_week"].iloc[0] == 4  # Friday

    def test_none_pickup_info_handled(self, sample_auction_df):
        """Test that None pickup info is handled."""
        result = add_pickup_window_features(sample_auction_df)

        # Third auction has None removal_info
        assert result["auction_num_pickup_windows"].iloc[2] == 0
        assert result["auction_total_pickup_hours"].iloc[2] == 0.0


# =============================================================================
# Test Auction Partner Feature
# =============================================================================


class TestAuctionPartnerFeature:
    """Test auction partner boolean feature."""

    def test_partner_feature_added(self, sample_auction_df):
        """Test that partner feature is added."""
        result = add_auction_partner_feature(sample_auction_df)

        assert "auction_has_partner" in result.columns

    def test_partner_feature_values(self, sample_auction_df):
        """Test that partner feature values are correct."""
        result = add_auction_partner_feature(sample_auction_df)

        # First has partner URL, second is empty, third is None
        assert result["auction_has_partner"].iloc[0] == 1
        assert result["auction_has_partner"].iloc[1] == 0
        assert result["auction_has_partner"].iloc[2] == 0


# =============================================================================
# Test Normalized Auction Totals
# =============================================================================


class TestNormalizedAuctionTotals:
    """Test normalized auction total features."""

    def test_normalized_features_added(self, sample_auction_df):
        """Test that normalized features are added."""
        result = add_normalized_auction_totals(sample_auction_df)

        expected_cols = [
            "auction_avg_views_per_item",
            "auction_avg_price_per_item",
            "auction_avg_bids_per_item",
            "auction_avg_images_per_item",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_normalized_values_correct(self, sample_auction_df):
        """Test that normalized values are calculated correctly."""
        result = add_normalized_auction_totals(sample_auction_df)

        # First auction: 5000 views / 50 items = 100 views per item
        assert result["auction_avg_views_per_item"].iloc[0] == 100.0
        # 2500 / 50 = 50.0 price per item
        assert result["auction_avg_price_per_item"].iloc[0] == 50.0


# =============================================================================
# Test Geospatial Features
# =============================================================================


class TestGeospatialFeatures:
    """Test geospatial feature engineering."""

    def test_fsa_extracted(self, sample_enriched_df):
        """Test that FSA is extracted from postal code."""
        result = add_geospatial_features(sample_enriched_df)

        assert "auction_fsa" in result.columns
        assert result["auction_fsa"].iloc[0] == "M5V"
        assert result["auction_fsa"].iloc[1] == "K1A"

    def test_postal_zone_extracted(self, sample_enriched_df):
        """Test that postal zone (first char) is extracted."""
        result = add_geospatial_features(sample_enriched_df)

        assert "auction_postal_zone" in result.columns
        assert result["auction_postal_zone"].iloc[0] == "M"
        assert result["auction_postal_zone"].iloc[1] == "K"

    def test_lat_lng_features_added(self, sample_enriched_df):
        """Test that lat/lng based features are added."""
        result = add_geospatial_features(sample_enriched_df)

        assert "auction_latitude" in result.columns
        assert "auction_longitude" in result.columns
        assert "auction_distance_from_toronto" in result.columns
        assert "auction_is_northern" in result.columns


# =============================================================================
# Test Categorical Encoding
# =============================================================================


class TestCategoricalEncoding:
    """Test categorical variable encoding."""

    def test_low_cardinality_one_hot(self, sample_enriched_df):
        """Test that low cardinality variables are one-hot encoded."""
        result = encode_categorical_features(sample_enriched_df)

        # Check for one-hot encoded columns for auction_type (3 unique values)
        type_cols = [c for c in result.columns if c.startswith("auction_type_")]
        assert len(type_cols) > 0

    def test_high_cardinality_frequency(self):
        """Test that high cardinality variables use frequency encoding."""
        # Create df with many unique cities
        df = pd.DataFrame(
            {
                "enriched_auction_approxLocation_city": [
                    f"City_{i}" for i in range(100)
                ],
            }
        )
        result = encode_categorical_features(df)

        assert "auction_city_frequency" in result.columns
        assert "auction_city_count" in result.columns


# =============================================================================
# Test Column Selection
# =============================================================================


class TestSelectFinalColumns:
    """Test final column selection."""

    def test_only_auction_prefix_columns(self):
        """Test that only columns with auction_ prefix are kept."""
        df = pd.DataFrame(
            {
                "auction_id": [1, 2],
                "auction_item_count": [50, 100],
                "other_column": ["a", "b"],
                "enriched_data": [1.0, 2.0],
            }
        )
        result = select_final_columns(df)

        assert "auction_id" in result.columns
        assert "auction_item_count" in result.columns
        assert "other_column" not in result.columns
        assert "enriched_data" not in result.columns

    def test_raw_text_columns_dropped(self):
        """Test that raw text columns are dropped."""
        df = pd.DataFrame(
            {
                "auction_id": [1, 2],
                "auction_title": ["Title 1", "Title 2"],
                "auction_intro": ["<p>Intro 1</p>", "<p>Intro 2</p>"],
                "auction_item_count": [50, 100],
            }
        )
        result = select_final_columns(df)

        assert "auction_id" in result.columns
        assert "auction_item_count" in result.columns
        assert "auction_title" not in result.columns
        assert "auction_intro" not in result.columns

    def test_auction_id_first(self):
        """Test that auction_id is the first column."""
        df = pd.DataFrame(
            {
                "auction_item_count": [50],
                "auction_id": [1],
                "auction_length_hours": [100.0],
            }
        )
        result = select_final_columns(df)

        assert result.columns[0] == "auction_id"
