# =============================================================================
# Tests for Bid Feature Engineering
# =============================================================================
"""
Tests for the sequential bid feature engineering module.

Tests verify:
1. Causal feature computation (no leakage from future bids)
2. Correctness of rolling window calculations
3. Edge cases (first bid, single bid items)
4. MaxSold bid increment rule validation
"""

import numpy as np
import pandas as pd
import pytest

from src.features_bid import (
    add_auction_progress_features,
    add_bid_amount_features,
    add_bid_count_features,
    add_temporal_activity_features,
    add_time_series_features,
    calculate_increment_deviation,
    engineer_bid_features,
    get_expected_increment,
    is_unusual_increment,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_bid_df():
    """Create sample bid data for testing."""
    return pd.DataFrame({
        "auction_id": [1, 1, 1, 1, 1],
        "item_id": [100, 100, 100, 100, 100],
        "bid_time": [
            "2024-01-05T10:00:00",
            "2024-01-05T10:05:00",
            "2024-01-05T10:06:00",
            "2024-01-05T10:30:00",
            "2024-01-05T11:00:00",
        ],
        "bid_number": [1, 2, 3, 4, 5],
        "bid_amount": [5.0, 10.0, 15.0, 25.0, 50.0],
        "is_proxy_bid": [False, False, True, False, True],
    })


@pytest.fixture
def multi_item_bid_df():
    """Create sample bid data for multiple items."""
    return pd.DataFrame({
        "auction_id": [1, 1, 1, 1, 2, 2, 2],
        "item_id": [100, 100, 100, 100, 200, 200, 200],
        "bid_time": [
            "2024-01-05T10:00:00",
            "2024-01-05T10:05:00",
            "2024-01-05T10:06:00",
            "2024-01-05T10:30:00",
            "2024-01-05T11:00:00",
            "2024-01-05T11:01:00",
            "2024-01-05T11:30:00",
        ],
        "bid_number": [1, 2, 3, 4, 1, 2, 3],
        "bid_amount": [5.0, 10.0, 15.0, 25.0, 100.0, 110.0, 150.0],
        "is_proxy_bid": [False, False, True, False, True, False, True],
    })


@pytest.fixture
def single_bid_df():
    """Create sample data with single-bid items."""
    return pd.DataFrame({
        "auction_id": [1, 1],
        "item_id": [100, 200],
        "bid_time": [
            "2024-01-05T10:00:00",
            "2024-01-05T11:00:00",
        ],
        "bid_number": [1, 1],
        "bid_amount": [10.0, 50.0],
        "is_proxy_bid": [False, True],
    })


@pytest.fixture
def rapid_bidding_df():
    """Create sample data with rapid bidding (cluster detection)."""
    return pd.DataFrame({
        "auction_id": [1] * 8,
        "item_id": [100] * 8,
        "bid_time": [
            "2024-01-05T10:00:00",  # First bid
            "2024-01-05T10:00:10",  # 10s later - cluster
            "2024-01-05T10:00:20",  # 10s later - cluster
            "2024-01-05T10:00:25",  # 5s later - cluster
            "2024-01-05T10:02:00",  # 95s later - new cluster
            "2024-01-05T10:02:15",  # 15s later - cluster
            "2024-01-05T10:10:00",  # Long gap - new cluster
            "2024-01-05T10:10:05",  # 5s later - cluster
        ],
        "bid_number": list(range(1, 9)),
        "bid_amount": [5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 15.0, 16.0],
        "is_proxy_bid": [False] * 8,
    })


# =============================================================================
# MaxSold Bid Increment Rule Tests
# =============================================================================


class TestBidIncrementRules:
    """Tests for MaxSold bid increment rule functions."""

    def test_get_expected_increment_low_bid(self):
        """Test expected increment for low bids ($0-$25)."""
        assert get_expected_increment(5) == 1
        assert get_expected_increment(20) == 1
        assert get_expected_increment(24.99) == 1

    def test_get_expected_increment_medium_bid(self):
        """Test expected increment for medium bids ($25-$100)."""
        assert get_expected_increment(25) == 5
        assert get_expected_increment(50) == 5
        assert get_expected_increment(99.99) == 5

    def test_get_expected_increment_high_bid(self):
        """Test expected increment for higher bids."""
        assert get_expected_increment(100) == 10
        assert get_expected_increment(500) == 25
        assert get_expected_increment(1000) == 50
        assert get_expected_increment(5000) == 100
        assert get_expected_increment(10000) == 100

    def test_is_unusual_increment_normal(self):
        """Test normal increments are not flagged."""
        # Normal $1 increment at $5
        assert not is_unusual_increment(5, 6)
        # Normal $5 increment at $50
        assert not is_unusual_increment(50, 55)
        # Normal $10 increment at $100
        assert not is_unusual_increment(100, 110)

    def test_is_unusual_increment_too_small(self):
        """Test that too-small increments are flagged."""
        # $0.50 increment when $1 expected - unusual
        # Note: In real auction this wouldn't be allowed, but we flag it
        assert is_unusual_increment(5, 5.50)

    def test_is_unusual_increment_very_large(self):
        """Test that very large increments (>3x expected) are flagged."""
        # $10 increment when $1 expected at $5 (10x) - unusual
        assert is_unusual_increment(5, 15)
        # $50 increment when $5 expected at $50 (10x) - unusual
        assert is_unusual_increment(50, 100)

    def test_is_unusual_increment_handles_nan(self):
        """Test that NaN values don't crash."""
        assert not is_unusual_increment(np.nan, 10)
        assert not is_unusual_increment(10, np.nan)
        assert not is_unusual_increment(np.nan, np.nan)

    def test_calculate_increment_deviation(self):
        """Test increment deviation ratio calculation."""
        # Exact expected increment
        deviation = calculate_increment_deviation(5, 6)  # $1 at $5 level
        assert abs(deviation - 1.0) < 0.01

        # Double expected increment
        deviation = calculate_increment_deviation(5, 7)  # $2 at $5 level
        assert abs(deviation - 2.0) < 0.01

    def test_calculate_increment_deviation_handles_nan(self):
        """Test deviation calculation handles edge cases."""
        assert np.isnan(calculate_increment_deviation(np.nan, 10))
        assert np.isnan(calculate_increment_deviation(10, np.nan))
        assert np.isnan(calculate_increment_deviation(0, 10))  # Zero previous bid


# =============================================================================
# Time-Series Features Tests
# =============================================================================


class TestTimeSeriesFeatures:
    """Tests for time-series feature engineering."""

    def test_time_since_first_bid(self, sample_bid_df):
        """Test time since first bid calculation."""
        result = add_time_series_features(sample_bid_df)

        # First bid should have 0 time since first bid
        assert result.iloc[0]["time_since_first_bid_seconds"] == 0

        # Second bid at 10:05 is 5 minutes (300s) after first
        assert result.iloc[1]["time_since_first_bid_seconds"] == 300

        # Third bid at 10:06 is 6 minutes (360s) after first
        assert result.iloc[2]["time_since_first_bid_seconds"] == 360

    def test_time_since_last_bid(self, sample_bid_df):
        """Test time since last bid calculation."""
        result = add_time_series_features(sample_bid_df)

        # First bid has 0 time since last (no previous bid)
        assert result.iloc[0]["time_since_last_bid_seconds"] == 0

        # Second bid is 5 minutes after first
        assert result.iloc[1]["time_since_last_bid_seconds"] == 300

        # Third bid is 1 minute after second
        assert result.iloc[2]["time_since_last_bid_seconds"] == 60

    def test_elapsed_time_fraction(self, sample_bid_df):
        """Test elapsed time fraction calculation."""
        result = add_time_series_features(sample_bid_df)

        # First bid has 0 elapsed time fraction
        assert result.iloc[0]["elapsed_time_fraction"] == 0

        # Last bid should have elapsed_time_fraction of 1.0
        assert result.iloc[-1]["elapsed_time_fraction"] == 1.0

        # Intermediate bids should be between 0 and 1
        for i in range(1, len(result) - 1):
            assert 0 <= result.iloc[i]["elapsed_time_fraction"] <= 1

    def test_time_features_per_item(self, multi_item_bid_df):
        """Test that time features are computed per item."""
        result = add_time_series_features(multi_item_bid_df)

        # Item 100: First bid should have 0 time since first bid
        item_100 = result[result["item_id"] == 100]
        assert item_100.iloc[0]["time_since_first_bid_seconds"] == 0

        # Item 200: First bid (at 11:00) should also have 0
        item_200 = result[result["item_id"] == 200]
        assert item_200.iloc[0]["time_since_first_bid_seconds"] == 0

    def test_time_features_single_bid(self, single_bid_df):
        """Test time features for single-bid items."""
        result = add_time_series_features(single_bid_df)

        # Single bid items should have 0 for all time-since features
        for _, row in result.iterrows():
            assert row["time_since_first_bid_seconds"] == 0
            assert row["time_since_last_bid_seconds"] == 0
            assert row["elapsed_time_fraction"] == 0


# =============================================================================
# Bid Count Features Tests
# =============================================================================


class TestBidCountFeatures:
    """Tests for bid count and intensity feature engineering."""

    def test_number_of_bids_so_far(self, sample_bid_df):
        """Test cumulative bid count."""
        result = add_bid_count_features(sample_bid_df)

        # Should be sequential 1, 2, 3, 4, 5
        expected = [1, 2, 3, 4, 5]
        assert list(result["number_of_bids_so_far"]) == expected

    def test_is_first_bid(self, sample_bid_df):
        """Test first bid detection."""
        result = add_bid_count_features(sample_bid_df)

        # Only first bid should be flagged
        assert result.iloc[0]["is_first_bid"] == 1
        for i in range(1, len(result)):
            assert result.iloc[i]["is_first_bid"] == 0

    def test_rolling_bid_count_1min(self, sample_bid_df):
        """Test rolling bid count in 1 minute window."""
        result = add_bid_count_features(sample_bid_df)

        # First bid: only itself in window
        assert result.iloc[0]["rolling_bid_count_1min"] == 1

        # Second bid (5 min later): only itself (first bid too old)
        assert result.iloc[1]["rolling_bid_count_1min"] == 1

        # Third bid (1 min after second): both second and third in window
        assert result.iloc[2]["rolling_bid_count_1min"] == 2

    def test_bid_velocity(self, sample_bid_df):
        """Test bid velocity calculation."""
        result = add_bid_count_features(sample_bid_df)

        # Check velocity is computed
        assert "bid_velocity_5min" in result.columns

        # All values should be non-negative
        assert all(result["bid_velocity_5min"] >= 0)

    def test_time_weighted_velocity(self, sample_bid_df):
        """Test time-weighted bid velocity."""
        result = add_bid_count_features(sample_bid_df)

        # Check feature exists
        assert "time_weighted_bid_velocity" in result.columns

        # First bid should have weighted count close to 1
        # (it weights itself with full weight)
        assert result.iloc[0]["time_weighted_bid_velocity"] >= 0.99

    def test_bid_count_per_item(self, multi_item_bid_df):
        """Test bid count features are computed per item."""
        result = add_bid_count_features(multi_item_bid_df)

        # Item 100 should have counts 1, 2, 3, 4
        item_100 = result[result["item_id"] == 100]
        assert list(item_100["number_of_bids_so_far"]) == [1, 2, 3, 4]

        # Item 200 should reset to 1, 2, 3
        item_200 = result[result["item_id"] == 200]
        assert list(item_200["number_of_bids_so_far"]) == [1, 2, 3]


# =============================================================================
# Bid Amount Features Tests
# =============================================================================


class TestBidAmountFeatures:
    """Tests for bid amount feature engineering."""

    def test_bid_increment(self, sample_bid_df):
        """Test bid increment calculation."""
        result = add_bid_amount_features(sample_bid_df)

        # First bid: increment equals the bid amount (no previous)
        assert result.iloc[0]["bid_increment"] == 5.0

        # Second bid: 10 - 5 = 5
        assert result.iloc[1]["bid_increment"] == 5.0

        # Third bid: 15 - 10 = 5
        assert result.iloc[2]["bid_increment"] == 5.0

        # Fourth bid: 25 - 15 = 10
        assert result.iloc[3]["bid_increment"] == 10.0

    def test_bid_increment_pct(self, sample_bid_df):
        """Test bid increment percentage calculation."""
        result = add_bid_amount_features(sample_bid_df)

        # First bid has no percentage (NaN expected)
        assert np.isnan(result.iloc[0]["bid_increment_pct"])

        # Second bid: (5/5) * 100 = 100%
        assert result.iloc[1]["bid_increment_pct"] == 100.0

        # Third bid: (5/10) * 100 = 50%
        assert result.iloc[2]["bid_increment_pct"] == 50.0

    def test_proxy_bid_flag(self, sample_bid_df):
        """Test proxy bid flag is correctly extracted."""
        result = add_bid_amount_features(sample_bid_df)

        expected = [0, 0, 1, 0, 1]
        assert list(result["proxy_bid_flag"]) == expected

    def test_bid_amount_percentile(self, sample_bid_df):
        """Test bid amount percentile calculation."""
        result = add_bid_amount_features(sample_bid_df)

        # First bid is 100% (only bid so far)
        assert result.iloc[0]["bid_amount_percentile_so_far"] == 100.0

        # Second bid ($10) is highest of $5, $10 -> 100%
        assert result.iloc[1]["bid_amount_percentile_so_far"] == 100.0

        # All subsequent bids are new highs, so 100%
        for i in range(2, len(result)):
            assert result.iloc[i]["bid_amount_percentile_so_far"] == 100.0

    def test_bid_amount_relative_to_max(self, sample_bid_df):
        """Test bid amount relative to running max."""
        result = add_bid_amount_features(sample_bid_df)

        # All bids are new highs, so ratio should be 1.0
        for _, row in result.iterrows():
            assert row["bid_amount_relative_to_max"] == 1.0

    def test_cumulative_increment_sum(self, sample_bid_df):
        """Test cumulative sum of increments."""
        result = add_bid_amount_features(sample_bid_df)

        # First bid: 5
        assert result.iloc[0]["cumulative_bid_increment_sum"] == 5.0

        # After second bid: 5 + 5 = 10
        assert result.iloc[1]["cumulative_bid_increment_sum"] == 10.0

        # After third bid: 10 + 5 = 15
        assert result.iloc[2]["cumulative_bid_increment_sum"] == 15.0

    def test_bid_amount_per_item(self, multi_item_bid_df):
        """Test bid amount features are computed per item."""
        result = add_bid_amount_features(multi_item_bid_df)

        # Item 200's first bid should have increment = bid_amount
        item_200 = result[result["item_id"] == 200]
        assert item_200.iloc[0]["bid_increment"] == 100.0


# =============================================================================
# Auction Progress Features Tests
# =============================================================================


class TestAuctionProgressFeatures:
    """Tests for auction progress feature engineering."""

    def test_bid_number_normalized(self, sample_bid_df):
        """Test normalized bid number."""
        result = add_auction_progress_features(sample_bid_df)

        # First bid: 1/1 = 1.0
        assert result.iloc[0]["bid_number_normalized"] == 1.0

        # Second bid: 2/2 = 1.0
        assert result.iloc[1]["bid_number_normalized"] == 1.0

        # Third bid: 3/3 = 1.0
        assert result.iloc[2]["bid_number_normalized"] == 1.0

    def test_current_to_starting_bid_ratio(self, sample_bid_df):
        """Test ratio to starting bid."""
        result = add_auction_progress_features(sample_bid_df)

        # First bid: 5/5 = 1.0
        assert result.iloc[0]["current_to_starting_bid_ratio"] == 1.0

        # Second bid: 10/5 = 2.0
        assert result.iloc[1]["current_to_starting_bid_ratio"] == 2.0

        # Last bid: 50/5 = 10.0
        assert result.iloc[-1]["current_to_starting_bid_ratio"] == 10.0

    def test_current_to_highest_ratio(self, sample_bid_df):
        """Test ratio to highest bid so far."""
        result = add_auction_progress_features(sample_bid_df)

        # All bids are new highs, so ratio should be 1.0
        for _, row in result.iterrows():
            assert row["current_to_highest_ratio"] == 1.0

    def test_is_new_high(self, sample_bid_df):
        """Test new high detection."""
        result = add_auction_progress_features(sample_bid_df)

        # All bids are new highs in this monotonically increasing series
        for _, row in result.iterrows():
            assert row["is_new_high"] == 1

    def test_is_new_high_with_lower_bid(self):
        """Test new high detection when a bid is not a new high."""
        df = pd.DataFrame({
            "item_id": [100, 100, 100],
            "bid_time": [
                "2024-01-05T10:00:00",
                "2024-01-05T10:05:00",
                "2024-01-05T10:10:00",
            ],
            "bid_amount": [10.0, 15.0, 12.0],  # Third bid is lower
        })

        result = add_auction_progress_features(df)

        # First two are new highs
        assert result.iloc[0]["is_new_high"] == 1
        assert result.iloc[1]["is_new_high"] == 1

        # Third bid is not a new high
        assert result.iloc[2]["is_new_high"] == 0

    def test_bid_momentum(self, sample_bid_df):
        """Test bid momentum calculation."""
        result = add_auction_progress_features(sample_bid_df)

        # Check feature exists
        assert "bid_momentum" in result.columns

        # First two bids don't have enough data for momentum
        assert result.iloc[0]["bid_momentum"] == 0.0
        assert result.iloc[1]["bid_momentum"] == 0.0


# =============================================================================
# Temporal Activity Features Tests
# =============================================================================


class TestTemporalActivityFeatures:
    """Tests for temporal activity/clustering feature engineering."""

    def test_bid_cluster_detection(self, rapid_bidding_df):
        """Test rapid bidding cluster detection."""
        result = add_temporal_activity_features(rapid_bidding_df)

        # First bid is not in a cluster (only one bid so far)
        assert result.iloc[0]["is_in_bid_cluster"] == 0

        # Second bid (10s later) should be in a cluster
        assert result.iloc[1]["is_in_bid_cluster"] == 1

        # Fifth bid starts a new cluster (95s gap)
        assert result.iloc[4]["is_in_bid_cluster"] == 0

        # Sixth bid (15s later) should be in the new cluster
        assert result.iloc[5]["is_in_bid_cluster"] == 1

    def test_cluster_size(self, rapid_bidding_df):
        """Test cluster size calculation."""
        result = add_temporal_activity_features(rapid_bidding_df)

        # First cluster: bids 1-4 (first 4 bids)
        # Bid 1: size 1
        assert result.iloc[0]["bid_cluster_size_so_far"] == 1
        # Bid 2: size 2
        assert result.iloc[1]["bid_cluster_size_so_far"] == 2
        # Bid 3: size 3
        assert result.iloc[2]["bid_cluster_size_so_far"] == 3
        # Bid 4: size 4
        assert result.iloc[3]["bid_cluster_size_so_far"] == 4

        # New cluster starts at bid 5
        assert result.iloc[4]["bid_cluster_size_so_far"] == 1

    def test_time_since_cluster_start(self, rapid_bidding_df):
        """Test time since cluster start calculation."""
        result = add_temporal_activity_features(rapid_bidding_df)

        # First bid of cluster: 0 seconds
        assert result.iloc[0]["time_since_cluster_start_seconds"] == 0

        # Second bid: 10 seconds since cluster start
        assert result.iloc[1]["time_since_cluster_start_seconds"] == 10

        # Third bid: 20 seconds since cluster start
        assert result.iloc[2]["time_since_cluster_start_seconds"] == 20

        # Fifth bid starts new cluster: 0 seconds
        assert result.iloc[4]["time_since_cluster_start_seconds"] == 0

    def test_cluster_detection_single_bid(self, single_bid_df):
        """Test cluster detection for single-bid items."""
        result = add_temporal_activity_features(single_bid_df)

        # Single bids are never in clusters
        for _, row in result.iterrows():
            assert row["is_in_bid_cluster"] == 0
            assert row["bid_cluster_size_so_far"] == 1


# =============================================================================
# Leakage Prevention Tests
# =============================================================================


class TestLeakagePrevention:
    """Tests to verify no data leakage from future bids."""

    def test_time_features_no_leakage(self, sample_bid_df):
        """Test that time features don't use future bid information."""
        result = add_time_series_features(sample_bid_df)

        # For each bid, elapsed_time_fraction should be <= 1.0
        # (can't know about future bids)
        for i in range(len(result)):
            assert result.iloc[i]["elapsed_time_fraction"] <= 1.0

    def test_count_features_no_leakage(self, sample_bid_df):
        """Test that count features don't use future bid information."""
        result = add_bid_count_features(sample_bid_df)

        # number_of_bids_so_far should only count current and past
        for i in range(len(result)):
            assert result.iloc[i]["number_of_bids_so_far"] == i + 1

    def test_amount_features_no_leakage(self, sample_bid_df):
        """Test that amount features don't use future bid information."""
        result = add_bid_amount_features(sample_bid_df)

        # bid_amount_relative_to_max should always be <= 1.0
        # (current bid can't exceed running max including itself)
        for _, row in result.iterrows():
            assert row["bid_amount_relative_to_max"] <= 1.0

    def test_percentile_no_leakage(self, sample_bid_df):
        """Test that percentile calculation uses only past/current bids."""
        result = add_bid_amount_features(sample_bid_df)

        # In monotonically increasing bids, each bid should be 100th percentile
        # because it's the highest so far
        for _, row in result.iterrows():
            assert row["bid_amount_percentile_so_far"] == 100.0

    def test_rolling_window_no_leakage(self, sample_bid_df):
        """Test that rolling windows don't include future bids."""
        result = add_bid_count_features(sample_bid_df)

        # At first bid, rolling count should be 1 (only itself)
        assert result.iloc[0]["rolling_bid_count_1min"] == 1
        assert result.iloc[0]["rolling_bid_count_5min"] == 1


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for the complete feature engineering pipeline."""

    def test_engineer_bid_features(self, sample_bid_df):
        """Test the full pipeline runs without error."""
        result = engineer_bid_features(sample_bid_df)

        # Should have more columns than input
        assert len(result.columns) > len(sample_bid_df.columns)

        # Should have same number of rows
        assert len(result) == len(sample_bid_df)

    def test_engineer_bid_features_required_columns(self):
        """Test that missing required columns raise an error."""
        df = pd.DataFrame({
            "item_id": [100],
            "bid_time": ["2024-01-05T10:00:00"],
            # Missing bid_amount
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            engineer_bid_features(df)

    def test_engineer_bid_features_feature_selection(self, sample_bid_df):
        """Test selective feature inclusion."""
        # Only time series features
        result = engineer_bid_features(
            sample_bid_df,
            include_time_series=True,
            include_bid_counts=False,
            include_bid_amounts=False,
            include_auction_progress=False,
            include_temporal_activity=False,
        )

        assert "time_since_first_bid_seconds" in result.columns
        assert "number_of_bids_so_far" not in result.columns
        assert "bid_increment" not in result.columns

    def test_pipeline_handles_unsorted_data(self, sample_bid_df):
        """Test that pipeline handles unsorted input data."""
        # Shuffle the data
        shuffled = sample_bid_df.sample(frac=1, random_state=42)

        result = engineer_bid_features(shuffled)

        # Result should be sorted by item_id and bid_time
        for item_id in result["item_id"].unique():
            item_data = result[result["item_id"] == item_id]
            # Bid times should be in ascending order
            bid_times = item_data["bid_time"].values
            assert all(bid_times[i] <= bid_times[i+1] for i in range(len(bid_times)-1))

    def test_pipeline_with_multi_item(self, multi_item_bid_df):
        """Test pipeline with multiple items."""
        result = engineer_bid_features(multi_item_bid_df)

        # Should preserve all rows
        assert len(result) == len(multi_item_bid_df)

        # Each item should have its own feature computations
        item_100 = result[result["item_id"] == 100]
        item_200 = result[result["item_id"] == 200]

        # First bid of each item should have similar base features
        assert item_100.iloc[0]["is_first_bid"] == 1
        assert item_200.iloc[0]["is_first_bid"] == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "item_id": [],
            "bid_time": [],
            "bid_amount": [],
        })

        result = engineer_bid_features(df)
        assert len(result) == 0

    def test_single_bid_item(self, single_bid_df):
        """Test handling of items with only one bid."""
        result = engineer_bid_features(single_bid_df)

        # Should not crash and produce valid features
        assert len(result) == 2

        # All first-bid indicators should be 1
        assert all(result["is_first_bid"] == 1)

        # Time since first bid should be 0
        assert all(result["time_since_first_bid_seconds"] == 0)

    def test_zero_bid_amount(self):
        """Test handling of zero bid amount."""
        df = pd.DataFrame({
            "item_id": [100, 100],
            "bid_time": ["2024-01-05T10:00:00", "2024-01-05T10:05:00"],
            "bid_amount": [0.0, 10.0],
        })

        # Should not crash
        result = engineer_bid_features(df)
        assert len(result) == 2

    def test_duplicate_timestamps(self):
        """Test handling of bids with same timestamp."""
        df = pd.DataFrame({
            "item_id": [100, 100],
            "bid_time": ["2024-01-05T10:00:00", "2024-01-05T10:00:00"],
            "bid_amount": [5.0, 10.0],
        })

        # Should not crash
        result = engineer_bid_features(df)
        assert len(result) == 2

    def test_large_time_gap(self):
        """Test handling of very large time gaps between bids."""
        df = pd.DataFrame({
            "item_id": [100, 100],
            "bid_time": [
                "2024-01-05T10:00:00",
                "2024-01-10T10:00:00",  # 5 days later
            ],
            "bid_amount": [5.0, 50.0],
        })

        result = engineer_bid_features(df)

        # Time since last bid should be 5 days in seconds
        expected_seconds = 5 * 24 * 60 * 60
        assert result.iloc[1]["time_since_last_bid_seconds"] == expected_seconds

    def test_string_datetime_parsing(self):
        """Test that string datetime columns are parsed correctly."""
        df = pd.DataFrame({
            "item_id": [100],
            "bid_time": ["2024-01-05T10:00:00"],  # String format
            "bid_amount": [10.0],
        })

        result = engineer_bid_features(df)

        # Should parse and process without error
        assert len(result) == 1
        assert pd.api.types.is_datetime64_any_dtype(result["bid_time"])
