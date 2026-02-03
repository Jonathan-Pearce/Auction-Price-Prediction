# =============================================================================
# Tests for Enriched Data Utilities
# =============================================================================
"""
Tests for enriched data loading and feature engineering functions.
"""

import pandas as pd
import pytest

from src.enriched import (
    add_fsa_column,
    compute_enriched_features,
    enrich_with_demographics,
    extract_fsa,
    load_all_enriched_data,
    load_population_data,
    load_tax_statistics,
)


class TestFSAExtraction:
    """Test Forward Sortation Area extraction from postal codes."""

    def test_extract_fsa_valid_with_space(self):
        """Test extraction from postal code with space."""
        assert extract_fsa("M5V 3A8") == "M5V"

    def test_extract_fsa_valid_without_space(self):
        """Test extraction from postal code without space."""
        assert extract_fsa("K1A0B1") == "K1A"

    def test_extract_fsa_lowercase(self):
        """Test extraction converts to uppercase."""
        assert extract_fsa("m5v 3a8") == "M5V"

    def test_extract_fsa_none(self):
        """Test extraction with None input."""
        assert extract_fsa(None) is None

    def test_extract_fsa_invalid(self):
        """Test extraction with invalid postal code."""
        assert extract_fsa("invalid") is None
        assert extract_fsa("123") is None
        assert extract_fsa("AB") is None

    def test_add_fsa_column(self):
        """Test adding FSA column to DataFrame."""
        df = pd.DataFrame({"postal_code": ["M5V 3A8", "K1A 0B1", None]})
        df_with_fsa = add_fsa_column(df)

        assert "fsa" in df_with_fsa.columns
        assert df_with_fsa["fsa"].iloc[0] == "M5V"
        assert df_with_fsa["fsa"].iloc[1] == "K1A"
        assert pd.isna(df_with_fsa["fsa"].iloc[2])


class TestDataLoading:
    """Test loading enriched datasets."""

    def test_load_population_data(self):
        """Test loading population/dwelling data."""
        df = load_population_data()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        expected_cols = [
            "GEO",
            "Population, 2021",
            "Total private dwellings, 2021",
            "Private dwellings occupied by usual residents, 2021",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check FSA format (uppercase, 3 chars)
        assert df["GEO"].str.len().eq(3).all()
        assert df["GEO"].str.isupper().all()

    def test_load_tax_statistics(self):
        """Test loading tax statistics data."""
        df = load_tax_statistics()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Check required columns
        expected_cols = [
            "FSA",
            "Number of Returns",
            "Total Income",
            "Median Total Income",
            "Average Total Income",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check FSA format
        assert df["FSA"].str.len().eq(3).all()
        assert df["FSA"].str.isupper().all()

    def test_load_all_enriched_data(self):
        """Test loading and merging all enriched data."""
        df = load_all_enriched_data()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.index.name == "FSA"

        # Check has columns from both datasets
        assert "Population, 2021" in df.columns
        assert "Median Total Income" in df.columns


class TestEnrichment:
    """Test data enrichment functions."""

    def test_enrich_with_demographics(self):
        """Test enriching DataFrame with demographic data."""
        # Create sample auction data
        sample_df = pd.DataFrame(
            {"auction_id": [1, 2, 3], "postal_code": ["A0A 1B2", "A0B 2C3", "A0E 4D5"]}
        )

        enriched_df = enrich_with_demographics(sample_df)

        # Check FSA column added
        assert "fsa" in enriched_df.columns

        # Check enriched columns added
        assert "Population, 2021" in enriched_df.columns
        assert "Median Total Income" in enriched_df.columns

        # Original columns preserved
        assert "auction_id" in enriched_df.columns
        assert "postal_code" in enriched_df.columns

    def test_compute_enriched_features(self):
        """Test computing derived features from enriched data."""
        # Create sample data with enriched columns
        sample_df = pd.DataFrame(
            {
                "auction_id": [1],
                "Population, 2021": [10000],
                "Total private dwellings, 2021": [4000],
                "Private dwellings occupied by usual residents, 2021": [3500],
                "Number of Returns": [5000],
                "Median Total Income": [50000],
                "Average Total Income": [60000],
            }
        )

        features_df = compute_enriched_features(sample_df)

        # Check derived features exist
        assert "persons_per_dwelling" in features_df.columns
        assert "occupancy_rate" in features_df.columns
        assert "income_skew" in features_df.columns
        assert "returns_per_capita" in features_df.columns
        assert "economic_capacity" in features_df.columns

        # Verify calculations
        assert features_df["persons_per_dwelling"].iloc[0] == pytest.approx(10000 / 3500, rel=0.01)
        assert features_df["occupancy_rate"].iloc[0] == pytest.approx(3500 / 4000, rel=0.01)
        assert features_df["income_skew"].iloc[0] == pytest.approx((60000 - 50000) / 50000, rel=0.01)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_enrich_with_missing_postal_codes(self):
        """Test enrichment handles missing postal codes."""
        sample_df = pd.DataFrame({"auction_id": [1, 2], "postal_code": ["M5V 3A8", None]})

        enriched_df = enrich_with_demographics(sample_df)

        # Should not crash, but second row will have NaN enriched values
        assert len(enriched_df) == 2
        assert pd.isna(enriched_df["fsa"].iloc[1])

    def test_enrich_with_invalid_postal_codes(self):
        """Test enrichment handles invalid postal codes."""
        sample_df = pd.DataFrame({"auction_id": [1], "postal_code": ["INVALID"]})

        enriched_df = enrich_with_demographics(sample_df)

        # Should not crash, enriched values will be NaN
        assert len(enriched_df) == 1
        assert pd.isna(enriched_df["fsa"].iloc[0])
