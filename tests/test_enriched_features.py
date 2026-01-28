# =============================================================================
# Tests for Enriched Item Feature Engineering
# =============================================================================
"""
Tests for feature engineering functions for enriched item JSON columns.
"""

import json
import pandas as pd
import pytest

from src.features import (
    _safe_parse_json,
    engineer_enriched_brands_features,
    engineer_enriched_categories_features,
    engineer_enriched_items_features,
    engineer_enriched_attributes_features,
    engineer_enriched_photos_features,
    engineer_all_enriched_features,
)


class TestSafeParseJson:
    """Test the JSON parsing helper function."""

    def test_parse_list(self):
        """Test parsing list input."""
        assert _safe_parse_json(["a", "b"]) == ["a", "b"]

    def test_parse_dict(self):
        """Test parsing dict input."""
        assert _safe_parse_json({"key": "value"}) == {"key": "value"}

    def test_parse_json_string(self):
        """Test parsing JSON string."""
        assert _safe_parse_json('["a", "b"]') == ["a", "b"]
        assert _safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_parse_none(self):
        """Test parsing None returns empty list."""
        assert _safe_parse_json(None) == []

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        assert _safe_parse_json("") == []
        assert _safe_parse_json("  ") == []

    def test_parse_null_string(self):
        """Test parsing 'null' string returns empty list."""
        assert _safe_parse_json("null") == []

    def test_parse_empty_array_string(self):
        """Test parsing '[]' string returns empty list."""
        assert _safe_parse_json("[]") == []

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        assert _safe_parse_json("not valid json") == []


class TestBrandsFeatures:
    """Test brand feature engineering."""

    @pytest.fixture
    def sample_brands_df(self):
        """Create sample DataFrame with brands data."""
        return pd.DataFrame({
            "item_id": [1, 2, 3, 4, 5],
            "enriched_item_brands": [
                '["Royal Albert", "Wedgwood"]',  # Luxury brands
                '["Sony"]',
                '[]',  # Empty
                '["Unknown Brand"]',
                None,  # Null
            ]
        })

    def test_basic_brand_features(self, sample_brands_df):
        """Test basic brand feature creation."""
        result = engineer_enriched_brands_features(sample_brands_df)

        assert "has_brand" in result.columns
        assert "brand_count" in result.columns
        assert "primary_brand" in result.columns
        assert "is_luxury_brand" in result.columns

    def test_brand_count(self, sample_brands_df):
        """Test brand count is correct."""
        result = engineer_enriched_brands_features(sample_brands_df)

        assert result["brand_count"].iloc[0] == 2
        assert result["brand_count"].iloc[1] == 1
        assert result["brand_count"].iloc[2] == 0
        assert result["brand_count"].iloc[4] == 0

    def test_has_brand(self, sample_brands_df):
        """Test has_brand flag is correct."""
        result = engineer_enriched_brands_features(sample_brands_df)

        assert result["has_brand"].iloc[0] == True
        assert result["has_brand"].iloc[1] == True
        assert result["has_brand"].iloc[2] == False
        assert result["has_brand"].iloc[4] == False

    def test_primary_brand(self, sample_brands_df):
        """Test primary brand extraction."""
        result = engineer_enriched_brands_features(sample_brands_df)

        assert result["primary_brand"].iloc[0] == "Royal Albert"
        assert result["primary_brand"].iloc[1] == "Sony"
        assert pd.isna(result["primary_brand"].iloc[2])

    def test_luxury_brand_detection(self, sample_brands_df):
        """Test luxury brand detection."""
        result = engineer_enriched_brands_features(sample_brands_df)

        assert result["is_luxury_brand"].iloc[0] == True  # Royal Albert, Wedgwood
        assert result["is_luxury_brand"].iloc[1] == False  # Sony (not luxury)
        assert result["is_luxury_brand"].iloc[3] == False  # Unknown Brand

    def test_missing_column(self):
        """Test handling of missing brands column."""
        df = pd.DataFrame({"item_id": [1, 2]})
        result = engineer_enriched_brands_features(df)
        # Should return original DataFrame without error
        assert len(result) == 2


class TestCategoriesFeatures:
    """Test category feature engineering."""

    @pytest.fixture
    def sample_categories_df(self):
        """Create sample DataFrame with categories data."""
        return pd.DataFrame({
            "item_id": [1, 2, 3, 4],
            "enriched_item_categories": [
                '["furniture", "sofa", "living room"]',
                '["art", "painting"]',
                '["electronics"]',
                '[]',
            ]
        })

    def test_basic_category_features(self, sample_categories_df):
        """Test basic category feature creation."""
        result = engineer_enriched_categories_features(sample_categories_df)

        assert "category_depth" in result.columns
        assert "primary_category" in result.columns
        assert "secondary_category" in result.columns
        assert "category_path" in result.columns

    def test_category_depth(self, sample_categories_df):
        """Test category depth calculation."""
        result = engineer_enriched_categories_features(sample_categories_df)

        assert result["category_depth"].iloc[0] == 3
        assert result["category_depth"].iloc[1] == 2
        assert result["category_depth"].iloc[2] == 1
        assert result["category_depth"].iloc[3] == 0

    def test_primary_secondary_category(self, sample_categories_df):
        """Test primary and secondary category extraction."""
        result = engineer_enriched_categories_features(sample_categories_df)

        assert result["primary_category"].iloc[0] == "furniture"
        assert result["secondary_category"].iloc[0] == "sofa"
        assert result["primary_category"].iloc[2] == "electronics"
        assert pd.isna(result["secondary_category"].iloc[2])

    def test_super_category_flags(self, sample_categories_df):
        """Test super-category flag creation."""
        result = engineer_enriched_categories_features(sample_categories_df)

        assert result["is_furniture"].iloc[0] == True
        assert result["is_art"].iloc[1] == True
        assert result["is_electronics"].iloc[2] == True

    def test_category_path(self, sample_categories_df):
        """Test category path creation."""
        result = engineer_enriched_categories_features(sample_categories_df)

        assert result["category_path"].iloc[0] == "furniture > sofa > living room"
        assert result["category_path"].iloc[1] == "art > painting"


class TestItemsFeatures:
    """Test items feature engineering."""

    @pytest.fixture
    def sample_items_df(self):
        """Create sample DataFrame with items data."""
        return pd.DataFrame({
            "item_id": [1, 2, 3],
            "enriched_item_items": [
                '[{"title": "Wooden chair", "category": "chair"}, {"title": "Side table", "category": "table"}]',
                '[{"title": "Oil painting landscape", "category": "painting"}]',
                '[]',
            ]
        })

    def test_basic_items_features(self, sample_items_df):
        """Test basic items feature creation."""
        result = engineer_enriched_items_features(sample_items_df)

        assert "num_items_in_lot" in result.columns
        assert "is_single_item" in result.columns
        assert "is_multi_item" in result.columns
        assert "item_titles_combined" in result.columns

    def test_num_items_in_lot(self, sample_items_df):
        """Test item count calculation."""
        result = engineer_enriched_items_features(sample_items_df)

        assert result["num_items_in_lot"].iloc[0] == 2
        assert result["num_items_in_lot"].iloc[1] == 1
        assert result["num_items_in_lot"].iloc[2] == 0

    def test_single_multi_item_flags(self, sample_items_df):
        """Test single/multi item flags."""
        result = engineer_enriched_items_features(sample_items_df)

        assert result["is_single_item"].iloc[1] == True
        assert result["is_multi_item"].iloc[1] == False

    def test_item_categories_unique(self, sample_items_df):
        """Test unique category count."""
        result = engineer_enriched_items_features(sample_items_df)

        assert result["item_categories_unique"].iloc[0] == 2  # chair, table


class TestAttributesFeatures:
    """Test attributes feature engineering."""

    @pytest.fixture
    def sample_attributes_df(self):
        """Create sample DataFrame with attributes data."""
        return pd.DataFrame({
            "item_id": [1, 2, 3, 4],
            "enriched_item_attributes": [
                '[{"name": "material", "value": "wood"}, {"name": "dimensions", "value": "24x36 inches"}]',
                '[{"name": "year", "value": "1970s"}, {"name": "color", "value": "blue"}]',
                '[{"name": "year_manufactured", "value": "1985"}]',
                '[]',
            ]
        })

    def test_basic_attributes_features(self, sample_attributes_df):
        """Test basic attributes feature creation."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert "num_attributes" in result.columns
        assert "has_dimensions" in result.columns
        assert "has_material" in result.columns
        assert "primary_material" in result.columns

    def test_num_attributes(self, sample_attributes_df):
        """Test attribute count."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert result["num_attributes"].iloc[0] == 2
        assert result["num_attributes"].iloc[3] == 0

    def test_material_detection(self, sample_attributes_df):
        """Test material detection and classification."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert result["has_material"].iloc[0] == True
        assert result["primary_material"].iloc[0] == "wood"
        assert result["is_material_wood"].iloc[0] == True

    def test_year_extraction(self, sample_attributes_df):
        """Test year/decade extraction."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert result["has_year_info"].iloc[1] == True  # 1970s
        assert result["decade"].iloc[1] == 1970
        assert result["has_year_info"].iloc[2] == True  # 1985
        assert result["year_numeric"].iloc[2] == 1985
        assert result["decade"].iloc[2] == 1980

    def test_dimension_extraction(self, sample_attributes_df):
        """Test dimension detection."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert result["has_dimensions"].iloc[0] == True
        assert result["dimension_largest"].iloc[0] == 36.0

    def test_era_category(self, sample_attributes_df):
        """Test era category assignment."""
        result = engineer_enriched_attributes_features(sample_attributes_df)

        assert result["era_category"].iloc[1] == "1950_1980"  # 1970s
        assert result["era_category"].iloc[2] == "1980_2000"  # 1985


class TestPhotosFeatures:
    """Test photos feature engineering."""

    @pytest.fixture
    def sample_photos_df(self):
        """Create sample DataFrame with photos data."""
        return pd.DataFrame({
            "item_id": [1, 2, 3],
            "enriched_item_photosTaken": [
                '[{"description": "Front view", "reason": "Show overall condition", "imageId": "img1", "imagePath": "path/to/img1.jpg"}, '
                '{"description": "Back view", "reason": "Show maker mark", "imageId": "img2", "imagePath": "path/to/img2.jpg"}]',
                '[{"description": "Detail shot", "reason": "Show detail closeup", "imageId": "img3", "imagePath": "path/to/img3.jpg"}]',
                '[]',
            ]
        })

    def test_basic_photos_features(self, sample_photos_df):
        """Test basic photos feature creation."""
        result = engineer_enriched_photos_features(sample_photos_df)

        assert "num_photos" in result.columns
        assert "has_photos" in result.columns
        assert "photo_desc_length_avg" in result.columns
        assert "photo_coverage_score" in result.columns

    def test_num_photos(self, sample_photos_df):
        """Test photo count."""
        result = engineer_enriched_photos_features(sample_photos_df)

        assert result["num_photos"].iloc[0] == 2
        assert result["num_photos"].iloc[1] == 1
        assert result["num_photos"].iloc[2] == 0

    def test_photo_content_flags(self, sample_photos_df):
        """Test photo content flag detection."""
        result = engineer_enriched_photos_features(sample_photos_df)

        assert result["has_condition_photo"].iloc[0] == True
        assert result["has_brand_photo"].iloc[0] == True  # "maker mark"
        assert result["has_detail_photo"].iloc[1] == True

    def test_primary_photo_path(self, sample_photos_df):
        """Test primary photo path extraction."""
        result = engineer_enriched_photos_features(sample_photos_df)

        assert result["primary_photo_path"].iloc[0] == "path/to/img1.jpg"
        assert pd.isna(result["primary_photo_path"].iloc[2])


class TestEngineerAllEnrichedFeatures:
    """Test combined feature engineering function."""

    @pytest.fixture
    def sample_full_df(self):
        """Create sample DataFrame with all JSON columns."""
        return pd.DataFrame({
            "item_id": [1, 2],
            "enriched_item_brands": ['["Sony"]', '[]'],
            "enriched_item_categories": ['["electronics", "tv"]', '["furniture"]'],
            "enriched_item_items": [
                '[{"title": "TV", "category": "tv"}]',
                '[{"title": "Chair", "category": "chair"}]',
            ],
            "enriched_item_attributes": [
                '[{"name": "material", "value": "plastic"}]',
                '[{"name": "material", "value": "wood"}]',
            ],
            "enriched_item_photosTaken": [
                '[{"description": "Front", "reason": "overall", "imageId": "1", "imagePath": "a.jpg"}]',
                '[{"description": "Side", "reason": "detail", "imageId": "2", "imagePath": "b.jpg"}]',
            ],
        })

    def test_all_features_created(self, sample_full_df):
        """Test all feature groups are created."""
        result = engineer_all_enriched_features(sample_full_df)

        # Brands features
        assert "has_brand" in result.columns
        assert "brand_count" in result.columns

        # Categories features
        assert "category_depth" in result.columns
        assert "is_furniture" in result.columns

        # Items features
        assert "num_items_in_lot" in result.columns

        # Attributes features
        assert "has_material" in result.columns

        # Photos features
        assert "num_photos" in result.columns

    def test_selective_features(self, sample_full_df):
        """Test selective feature engineering."""
        result = engineer_all_enriched_features(
            sample_full_df,
            include_brands=True,
            include_categories=False,
            include_items=False,
            include_attributes=False,
            include_photos=False,
        )

        assert "has_brand" in result.columns
        assert "category_depth" not in result.columns
        assert "num_items_in_lot" not in result.columns

    def test_preserves_original_columns(self, sample_full_df):
        """Test original columns are preserved."""
        result = engineer_all_enriched_features(sample_full_df)

        assert "item_id" in result.columns
        assert result["item_id"].iloc[0] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "enriched_item_brands": pd.Series([], dtype="object"),
            "enriched_item_categories": pd.Series([], dtype="object"),
            "enriched_item_items": pd.Series([], dtype="object"),
            "enriched_item_attributes": pd.Series([], dtype="object"),
            "enriched_item_photosTaken": pd.Series([], dtype="object"),
        })
        # Should not raise an error
        result = engineer_all_enriched_features(df)
        assert len(result) == 0

    def test_all_null_values(self):
        """Test handling of all null values."""
        df = pd.DataFrame({
            "enriched_item_brands": [None, None],
            "enriched_item_categories": [None, None],
            "enriched_item_items": [None, None],
            "enriched_item_attributes": [None, None],
            "enriched_item_photosTaken": [None, None],
        })
        result = engineer_all_enriched_features(df)
        assert len(result) == 2
        assert result["has_brand"].sum() == 0
        assert result["num_photos"].sum() == 0
