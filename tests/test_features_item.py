# =============================================================================
# Tests for Item Feature Engineering Pipeline
# =============================================================================
"""
Tests for the item-level feature engineering pipeline.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.features_item import (
    add_boolean_features,
    add_item_closing_order,
    add_list_count_features,
    add_price_features,
    add_text_length_features,
    encode_categorical_features,
    select_final_columns,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_item_df():
    """Create sample item data for testing."""
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "auction_id": [100, 100, 101, 101],
            "item_title": ["Vintage Chair", "Antique Table", None, ""],
            "item_description": [
                "A beautiful vintage chair from the 1920s",
                "Oak dining table with intricate carvings",
                None,
                "",
            ],
            "item_current_bid": [50.0, 150.0, 0.0, 25.0],
            "item_closes": [
                "2024-01-05T18:00:00",
                "2024-01-05T17:00:00",
                "2024-01-06T10:00:00",
                "2024-01-06T12:00:00",
            ],
            "item_number_of_images": [3, 5, 1, 2],
            "item_bidding_extended": [True, False, False, True],
        }
    )


@pytest.fixture
def sample_enriched_df():
    """Create sample enriched item data for testing."""
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "enriched_item_title": ["Vintage Wood Chair", "Antique Oak Table", "", None],
            "enriched_item_description": [
                "A beautiful vintage wooden chair from the 1920s era",
                "An oak dining table with intricate hand-carved details",
                "",
                None,
            ],
            "enriched_item_qualitativeDescription": [
                "Excellent condition with minor wear",
                "Good condition with some scratches",
                None,
                "",
            ],
            "enriched_item_brand": ["Unknown", None, "", "IKEA"],
            "enriched_item_seriesLine": ["Heritage", "", None, "KALLAX"],
            "enriched_item_condition": ["Excellent", "Good", None, "Fair"],
            "enriched_item_working": ["Yes", "Yes", None, "Unknown"],
            "enriched_item_brands": [
                json.dumps(["BrandA", "BrandB"]),
                json.dumps([]),
                None,
                "[]",
            ],
            "enriched_item_categories": [
                json.dumps(["Furniture", "Antiques"]),
                json.dumps(["Furniture"]),
                json.dumps(["Electronics"]),
                "",
            ],
            "enriched_item_items": [
                json.dumps(["Chair", "Cushion"]),
                json.dumps(["Table"]),
                None,
                json.dumps([]),
            ],
            "enriched_item_attributes": [
                json.dumps({"color": "brown", "material": "wood"}),
                json.dumps({"color": "oak"}),
                None,
                "",
            ],
        }
    )


@pytest.fixture
def merged_df(sample_item_df, sample_enriched_df):
    """Create merged item and enriched data for testing."""
    return sample_item_df.merge(sample_enriched_df, on="item_id", how="left")


# =============================================================================
# Test Text Length Features
# =============================================================================


class TestTextLengthFeatures:
    """Test text length feature engineering."""

    def test_title_length_calculated(self, merged_df):
        """Test that item title length is calculated correctly."""
        result = add_text_length_features(merged_df)

        assert "item_title_length" in result.columns
        assert result["item_title_length"].iloc[0] == len("Vintage Chair")
        assert result["item_title_length"].iloc[1] == len("Antique Table")

    def test_description_length_calculated(self, merged_df):
        """Test that item description length is calculated correctly."""
        result = add_text_length_features(merged_df)

        assert "item_description_length" in result.columns
        assert result["item_description_length"].iloc[0] == len(
            "A beautiful vintage chair from the 1920s"
        )

    def test_enriched_title_length_calculated(self, merged_df):
        """Test that enriched title length is calculated correctly."""
        result = add_text_length_features(merged_df)

        assert "item_enriched_title_length" in result.columns
        assert result["item_enriched_title_length"].iloc[0] == len("Vintage Wood Chair")

    def test_enriched_description_length_calculated(self, merged_df):
        """Test that enriched description length is calculated correctly."""
        result = add_text_length_features(merged_df)

        assert "item_enriched_description_length" in result.columns
        expected = len("A beautiful vintage wooden chair from the 1920s era")
        assert result["item_enriched_description_length"].iloc[0] == expected

    def test_qualitative_description_length_calculated(self, merged_df):
        """Test that qualitative description length is calculated correctly."""
        result = add_text_length_features(merged_df)

        assert "item_enriched_qualitative_description_length" in result.columns
        expected = len("Excellent condition with minor wear")
        assert result["item_enriched_qualitative_description_length"].iloc[0] == expected

    def test_none_values_handled(self, merged_df):
        """Test that None values are handled (treated as empty string)."""
        result = add_text_length_features(merged_df)

        # Row 2 has None title
        assert result["item_title_length"].iloc[2] == 0

    def test_empty_string_handled(self, merged_df):
        """Test that empty strings return length 0."""
        result = add_text_length_features(merged_df)

        # Row 3 has empty title
        assert result["item_title_length"].iloc[3] == 0


# =============================================================================
# Test Boolean Features
# =============================================================================


class TestBooleanFeatures:
    """Test boolean feature engineering."""

    def test_has_brand_feature_added(self, merged_df):
        """Test that has_brand feature is added."""
        result = add_boolean_features(merged_df)

        assert "item_has_brand" in result.columns

    def test_has_brand_values(self, merged_df):
        """Test that has_brand values are correct."""
        result = add_boolean_features(merged_df)

        # Row 0: "Unknown" is populated (considered as having brand)
        assert result["item_has_brand"].iloc[0] == 1
        # Row 1: None
        assert result["item_has_brand"].iloc[1] == 0
        # Row 2: empty string
        assert result["item_has_brand"].iloc[2] == 0
        # Row 3: "IKEA" is populated
        assert result["item_has_brand"].iloc[3] == 1

    def test_has_series_line_feature_added(self, merged_df):
        """Test that has_series_line feature is added."""
        result = add_boolean_features(merged_df)

        assert "item_has_series_line" in result.columns

    def test_has_series_line_values(self, merged_df):
        """Test that has_series_line values are correct."""
        result = add_boolean_features(merged_df)

        # Row 0: "Heritage" is populated
        assert result["item_has_series_line"].iloc[0] == 1
        # Row 1: empty string
        assert result["item_has_series_line"].iloc[1] == 0
        # Row 2: None
        assert result["item_has_series_line"].iloc[2] == 0
        # Row 3: "KALLAX" is populated
        assert result["item_has_series_line"].iloc[3] == 1


# =============================================================================
# Test Categorical Encoding
# =============================================================================


class TestCategoricalEncoding:
    """Test categorical variable encoding."""

    def test_condition_encoded(self, merged_df):
        """Test that condition is one-hot encoded."""
        result = encode_categorical_features(merged_df)

        # Check for one-hot encoded columns for condition
        condition_cols = [c for c in result.columns if c.startswith("item_condition_")]
        assert len(condition_cols) > 0

    def test_working_encoded(self, merged_df):
        """Test that working is one-hot encoded."""
        result = encode_categorical_features(merged_df)

        # Check for one-hot encoded columns for working
        working_cols = [c for c in result.columns if c.startswith("item_working_")]
        assert len(working_cols) > 0

    def test_missing_values_handled(self, merged_df):
        """Test that missing values are filled with 'Unknown'."""
        result = encode_categorical_features(merged_df)

        # Check that Unknown column exists for condition
        assert "item_condition_unknown" in result.columns

    def test_empty_strings_converted_to_unknown(self):
        """Test that empty strings in categorical columns are converted to Unknown."""
        test_df = pd.DataFrame({
            "item_id": [1, 2, 3],
            "enriched_item_condition": ["new", "", None],
            "enriched_item_working": ["tested & working", "", None],
        })
        
        result = encode_categorical_features(test_df)
        
        # Empty strings should be converted to "Unknown", so no columns should end with just "_"
        condition_cols = [c for c in result.columns if c.startswith("item_condition_")]
        working_cols = [c for c in result.columns if c.startswith("item_working_")]
        
        # Should not have columns that end with just the prefix + "_"
        assert "item_condition_" not in condition_cols
        assert "item_working_" not in working_cols
        
        # Should have unknown columns
        assert "item_condition_unknown" in condition_cols
        assert "item_working_unknown" in working_cols
        
        # Verify row 2 (empty string) is encoded as Unknown
        assert result["item_condition_unknown"].iloc[1] == 1
        assert result["item_working_unknown"].iloc[1] == 1


# =============================================================================
# Test List Count Features
# =============================================================================


class TestListCountFeatures:
    """Test list count feature engineering."""

    def test_brands_count_added(self, merged_df):
        """Test that brands count is added."""
        result = add_list_count_features(merged_df)

        assert "item_brands_count" in result.columns

    def test_brands_count_values(self, merged_df):
        """Test that brands count values are correct."""
        result = add_list_count_features(merged_df)

        # Row 0: ["BrandA", "BrandB"] = 2
        assert result["item_brands_count"].iloc[0] == 2
        # Row 1: [] = 0
        assert result["item_brands_count"].iloc[1] == 0
        # Row 2: None = 0
        assert result["item_brands_count"].iloc[2] == 0

    def test_categories_count_added(self, merged_df):
        """Test that categories count is added."""
        result = add_list_count_features(merged_df)

        assert "item_categories_count" in result.columns
        # Row 0: ["Furniture", "Antiques"] = 2
        assert result["item_categories_count"].iloc[0] == 2

    def test_items_count_added(self, merged_df):
        """Test that items count is added."""
        result = add_list_count_features(merged_df)

        assert "item_items_count" in result.columns
        # Row 0: ["Chair", "Cushion"] = 2
        assert result["item_items_count"].iloc[0] == 2

    def test_attributes_count_added(self, merged_df):
        """Test that attributes count is added."""
        result = add_list_count_features(merged_df)

        assert "item_attributes_count" in result.columns
        # Row 0: {"color": "brown", "material": "wood"} has 2 keys
        # For dicts, we count the number of keys (attributes)
        assert result["item_attributes_count"].iloc[0] == 2


# =============================================================================
# Test Item Closing Order
# =============================================================================


class TestItemClosingOrder:
    """Test item closing order feature engineering."""

    def test_closing_order_added(self, merged_df):
        """Test that closing order is added."""
        result = add_item_closing_order(merged_df)

        assert "item_closing_order" in result.columns

    def test_closing_order_values(self, merged_df):
        """Test that closing order values are correct within auction."""
        result = add_item_closing_order(merged_df)

        # Auction 100: item 2 (17:00) closes before item 1 (18:00)
        # So item 2 should be order 1, item 1 should be order 2
        auction_100 = result[result["auction_id"] == 100]
        item_1_order = auction_100[auction_100["item_id"] == 1]["item_closing_order"].iloc[0]
        item_2_order = auction_100[auction_100["item_id"] == 2]["item_closing_order"].iloc[0]
        assert item_2_order < item_1_order

    def test_closing_order_independent_per_auction(self, merged_df):
        """Test that closing order is independent per auction."""
        result = add_item_closing_order(merged_df)

        # Auction 101: item 3 (10:00) closes before item 4 (12:00)
        auction_101 = result[result["auction_id"] == 101]
        item_3_order = auction_101[auction_101["item_id"] == 3]["item_closing_order"].iloc[0]
        item_4_order = auction_101[auction_101["item_id"] == 4]["item_closing_order"].iloc[0]
        assert item_3_order == 1
        assert item_4_order == 2


# =============================================================================
# Test Price Features
# =============================================================================


class TestPriceFeatures:
    """Test price feature engineering."""

    def test_winning_price_raw_added(self, merged_df):
        """Test that winning price raw is added."""
        result = add_price_features(merged_df)

        assert "item_winning_price_raw" in result.columns
        assert result["item_winning_price_raw"].iloc[0] == 50.0

    def test_winning_price_log1p_added(self, merged_df):
        """Test that log1p transformed price is added."""
        result = add_price_features(merged_df)

        assert "item_winning_price_log1p" in result.columns
        # log1p(50) ≈ 3.93
        assert abs(result["item_winning_price_log1p"].iloc[0] - np.log1p(50)) < 0.01

    def test_zero_price_handled(self, merged_df):
        """Test that zero prices are handled correctly."""
        result = add_price_features(merged_df)

        # Row 2 has 0 price
        assert result["item_winning_price_raw"].iloc[2] == 0
        assert result["item_winning_price_log1p"].iloc[2] == 0  # log1p(0) = 0


# =============================================================================
# Test Column Selection
# =============================================================================


class TestSelectFinalColumns:
    """Test final column selection."""

    def test_only_item_prefix_and_auction_id(self):
        """Test that only columns with item_ prefix and auction_id are kept."""
        df = pd.DataFrame(
            {
                "item_id": [1, 2],
                "auction_id": [100, 101],
                "item_winning_price_raw": [50.0, 100.0],
                "other_column": ["a", "b"],
                "enriched_data": [1.0, 2.0],
            }
        )
        result = select_final_columns(df)

        assert "item_id" in result.columns
        assert "auction_id" in result.columns
        assert "item_winning_price_raw" in result.columns
        assert "other_column" not in result.columns
        assert "enriched_data" not in result.columns

    def test_raw_text_columns_dropped(self):
        """Test that raw text columns are dropped."""
        df = pd.DataFrame(
            {
                "item_id": [1, 2],
                "auction_id": [100, 101],
                "item_title": ["Title 1", "Title 2"],
                "item_description": ["Desc 1", "Desc 2"],
                "item_title_length": [7, 7],
            }
        )
        result = select_final_columns(df)

        assert "item_id" in result.columns
        assert "item_title_length" in result.columns
        assert "item_title" not in result.columns
        assert "item_description" not in result.columns

    def test_item_id_and_auction_id_first(self):
        """Test that item_id and auction_id are the first columns."""
        df = pd.DataFrame(
            {
                "item_winning_price_log1p": [3.93],
                "auction_id": [100],
                "item_id": [1],
                "item_title_length": [10],
            }
        )
        result = select_final_columns(df)

        assert result.columns[0] == "item_id"
        assert result.columns[1] == "auction_id"

    def test_enriched_raw_columns_dropped(self):
        """Test that enriched raw columns are dropped."""
        df = pd.DataFrame(
            {
                "item_id": [1],
                "auction_id": [100],
                "enriched_item_title": ["Title"],
                "enriched_item_description": ["Desc"],
                "enriched_item_condition": ["Good"],
                "item_enriched_title_length": [5],
                "item_condition_good": [1],
            }
        )
        result = select_final_columns(df)

        assert "enriched_item_title" not in result.columns
        assert "enriched_item_description" not in result.columns
        assert "enriched_item_condition" not in result.columns
        assert "item_enriched_title_length" in result.columns
        assert "item_condition_good" in result.columns

    def test_empty_string_columns_filtered(self):
        """Test that columns from empty string encoding are filtered out."""
        df = pd.DataFrame(
            {
                "item_id": [1, 2],
                "auction_id": [100, 101],
                "item_condition_good": [1, 0],
                "item_condition_": [0, 1],  # From empty string - should be removed
                "item_working_tested_&_working": [1, 0],
                "item_working_": [0, 1],  # From empty string - should be removed
            }
        )
        result = select_final_columns(df)

        # Valid columns should be kept
        assert "item_condition_good" in result.columns
        assert "item_working_tested_&_working" in result.columns
        
        # Problematic columns should be removed
        assert "item_condition_" not in result.columns
        assert "item_working_" not in result.columns
