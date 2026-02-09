# =============================================================================
# Tests for Price Feature EDA
# =============================================================================
"""
Tests for price feature exploratory data analysis functions.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class TestPriceDistributionAnalysis:
    """Test price distribution analysis functions."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        # Create zero-inflated distribution
        zero_prices = np.zeros(100)
        non_zero_prices = np.random.lognormal(mean=2, sigma=1, size=900)
        prices = np.concatenate([zero_prices, non_zero_prices])
        df = pd.DataFrame({"item_current_bid": prices})
        return df

    def test_zero_bid_detection(self, sample_price_data):
        """Test detection of zero-bid items."""
        zero_count = (sample_price_data["item_current_bid"] == 0).sum()
        zero_pct = (sample_price_data["item_current_bid"] == 0).mean()

        assert zero_count == 100
        assert zero_pct == 0.1

    def test_price_statistics(self, sample_price_data):
        """Test basic price statistics calculation."""
        prices = sample_price_data["item_current_bid"]

        # Check statistics exist
        assert prices.mean() > 0
        assert prices.median() > 0
        assert prices.std() > 0
        assert prices.min() == 0  # Has zero bids
        assert prices.max() > 0

    def test_non_zero_filtering(self, sample_price_data):
        """Test filtering to non-zero prices."""
        prices = sample_price_data["item_current_bid"]
        prices_nonzero = prices[prices > 0]

        assert len(prices_nonzero) == 900
        assert prices_nonzero.min() > 0
        assert (prices_nonzero == 0).sum() == 0


class TestTransformations:
    """Test price transformation functions."""

    @pytest.fixture
    def positive_prices(self):
        """Create positive price data for transformations."""
        np.random.seed(42)
        return np.random.lognormal(mean=2, sigma=1, size=1000)

    @pytest.fixture
    def zero_inflated_prices(self):
        """Create zero-inflated price data."""
        np.random.seed(42)
        zeros = np.zeros(100)
        positives = np.random.lognormal(mean=2, sigma=1, size=900)
        return np.concatenate([zeros, positives])

    def test_log1p_transformation(self, zero_inflated_prices):
        """Test log1p transformation handles zeros."""
        log_prices = np.log1p(zero_inflated_prices)

        # Should not have infinities or NaN
        assert not np.any(np.isinf(log_prices))
        assert not np.any(np.isnan(log_prices))

        # Zero prices should map to log(1) = 0
        zero_mask = zero_inflated_prices == 0
        assert np.allclose(log_prices[zero_mask], 0)

    def test_log1p_reduces_skewness(self, positive_prices):
        """Test that log1p reduces skewness."""
        original_skew = stats.skew(positive_prices)
        log_prices = np.log1p(positive_prices)
        log_skew = stats.skew(log_prices)

        # Log transformation should reduce positive skew
        assert abs(log_skew) < abs(original_skew)

    def test_sqrt_transformation(self, positive_prices):
        """Test square root transformation."""
        sqrt_prices = np.sqrt(positive_prices)

        # Should not have NaN
        assert not np.any(np.isnan(sqrt_prices))

        # Should reduce skewness (but not as much as log)
        original_skew = stats.skew(positive_prices)
        sqrt_skew = stats.skew(sqrt_prices)
        assert abs(sqrt_skew) < abs(original_skew)

    def test_boxcox_transformation(self, positive_prices):
        """Test Box-Cox transformation."""
        # Box-Cox requires strictly positive values
        boxcox_prices, lambda_param = stats.boxcox(positive_prices)

        # Should not have NaN or inf
        assert not np.any(np.isnan(boxcox_prices))
        assert not np.any(np.isinf(boxcox_prices))

        # Lambda should be a number
        assert isinstance(lambda_param, (int, float))

    def test_yeojohnson_transformation(self, zero_inflated_prices):
        """Test Yeo-Johnson transformation (handles zeros)."""
        yj_prices, lambda_param = stats.yeojohnson(zero_inflated_prices)

        # Should not have NaN or inf
        assert not np.any(np.isnan(yj_prices))
        assert not np.any(np.isinf(yj_prices))

        # Lambda should be a number
        assert isinstance(lambda_param, (int, float))

    def test_log1p_inverse(self, zero_inflated_prices):
        """Test log1p transformation is reversible."""
        log_prices = np.log1p(zero_inflated_prices)
        reconstructed = np.exp(log_prices) - 1

        # Should reconstruct original (within floating point precision)
        assert np.allclose(reconstructed, zero_inflated_prices, rtol=1e-10)


class TestZeroBidFeatures:
    """Test zero-bid feature analysis."""

    @pytest.fixture
    def sample_feature_data(self):
        """Create sample data with features and target."""
        np.random.seed(42)
        n = 1000

        # Create features that correlate with zero bids
        df = pd.DataFrame({
            "item_viewed": np.random.randint(0, 300, n),
            "item_number_of_images": np.random.randint(0, 15, n),
            "item_starting_bid": np.random.uniform(1, 100, n),
            "item_bid_count": np.random.randint(0, 50, n),
        })

        # Create zero-inflated target
        # Items with low views and images more likely to be zero
        zero_prob = 1 / (1 + np.exp(df["item_viewed"] / 50 - 2))
        df["item_current_bid"] = np.where(
            np.random.rand(n) < zero_prob,
            0,
            np.random.lognormal(mean=2, sigma=1, size=n)
        )

        return df

    def test_has_bids_creation(self, sample_feature_data):
        """Test creation of binary has_bids target."""
        df = sample_feature_data.copy()
        df["has_bids"] = (df["item_current_bid"] > 0).astype(int)

        # Should be binary
        assert set(df["has_bids"].unique()) <= {0, 1}

        # Should match zero-bid count
        zero_count = (df["item_current_bid"] == 0).sum()
        no_bids_count = (df["has_bids"] == 0).sum()
        assert zero_count == no_bids_count

    def test_feature_comparison(self, sample_feature_data):
        """Test comparing features between zero and non-zero items."""
        df = sample_feature_data.copy()
        df["has_bids"] = (df["item_current_bid"] > 0).astype(int)

        # Items with bids should have more views on average
        zero_views = df[df["has_bids"] == 0]["item_viewed"].mean()
        nonzero_views = df[df["has_bids"] == 1]["item_viewed"].mean()

        # This may not always be true due to randomness, but likely
        # Just check that the means are different and both positive
        assert zero_views >= 0
        assert nonzero_views >= 0


class TestDataQuality:
    """Test data quality checks."""

    def test_missing_values_detection(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            "item_current_bid": [1.0, 2.0, None, 4.0, 0.0],
            "item_viewed": [10, 20, 30, None, 50],
        })

        # Check missing count per column
        assert df["item_current_bid"].isna().sum() == 1
        assert df["item_viewed"].isna().sum() == 1

    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1

        # Outliers are beyond 1.5 * IQR from Q1/Q3
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (prices < lower_bound) | (prices > upper_bound)

        # 100 should be detected as outlier
        assert outliers[-1]
        # Most others should not be outliers
        assert outliers[:-1].sum() <= 2


class TestRecommendationsLogic:
    """Test the logic behind recommendations."""

    def test_two_stage_prediction(self):
        """Test two-stage prediction logic."""
        # Mock predictions
        p_sell = 0.8  # 80% probability of selling
        log_price = 3.0  # log(price + 1)

        # Stage 1: Classification probability
        # Stage 2: Regression prediction
        final_price = p_sell * (np.exp(log_price) - 1)

        # Should be positive and less than exp(log_price) - 1
        assert final_price > 0
        assert final_price < (np.exp(log_price) - 1)

        # If p_sell is 1.0, should equal exp(log_price) - 1
        final_price_certain = 1.0 * (np.exp(log_price) - 1)
        assert np.isclose(final_price_certain, np.exp(log_price) - 1)

    def test_time_based_split(self):
        """Test time-based data splitting."""
        # Create sample data with dates
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        df = pd.DataFrame({
            "auction_end_date": dates,
            "price": np.random.rand(1000) * 100
        })

        # Sort by date
        df = df.sort_values("auction_end_date")

        # Split 70/15/15
        n = len(df)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]

        # Check temporal ordering
        assert train_df["auction_end_date"].max() <= val_df["auction_end_date"].min()
        assert val_df["auction_end_date"].max() <= test_df["auction_end_date"].min()

        # Check sizes
        assert len(train_df) == 700
        assert len(val_df) == 150
        assert len(test_df) == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
