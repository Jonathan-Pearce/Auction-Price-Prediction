# =============================================================================
# Tests for Datetime Feature Engineering
# =============================================================================
"""
Tests for the datetime feature engineering module.
"""

import pandas as pd
import pytest

from src.datetime_features import (
    DatetimeFeatureEngineer,
    add_auction_duration_features,
    add_bid_timing_features,
    engineer_all_datetime_features,
    get_part_of_day,
    get_season,
    is_canadian_holiday,
    is_us_holiday,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_datetime_df():
    """Sample DataFrame with datetime columns."""
    return pd.DataFrame(
        {
            "auction_id": [1, 2, 3, 4],
            "start_time": [
                "2024-01-01T10:00:00",
                "2024-06-15T14:00:00",
                "2024-12-25T09:00:00",
                "2024-07-04T15:30:00",
            ],
            "end_time": [
                "2024-01-05T18:00:00",
                "2024-06-20T14:00:00",
                "2024-12-28T09:00:00",
                "2024-07-07T15:30:00",
            ],
        }
    )


@pytest.fixture
def sample_bid_df():
    """Sample DataFrame with bid timing data."""
    return pd.DataFrame(
        {
            "bid_id": [1, 2, 3, 4],
            "bid_time": [
                "2024-01-02T10:00:00",
                "2024-01-05T17:58:30",  # Last minute bid
                "2024-01-03T14:00:00",
                "2024-01-05T17:00:00",
            ],
            "auction_start_time": ["2024-01-01T10:00:00"] * 4,
            "auction_end_time": ["2024-01-05T18:00:00"] * 4,
        }
    )


# =============================================================================
# Helper Functions Tests
# =============================================================================


def test_get_part_of_day():
    """Test part of day classification."""
    assert get_part_of_day(3) == "night"
    assert get_part_of_day(8) == "morning"
    assert get_part_of_day(14) == "afternoon"
    assert get_part_of_day(20) == "evening"


def test_get_season_northern_hemisphere():
    """Test season detection for northern hemisphere."""
    assert get_season(1, "north") == "winter"
    assert get_season(4, "north") == "spring"
    assert get_season(7, "north") == "summer"
    assert get_season(10, "north") == "fall"
    assert get_season(12, "north") == "winter"


def test_get_season_southern_hemisphere():
    """Test season detection for southern hemisphere."""
    assert get_season(1, "south") == "summer"  # Summer in south when winter in north
    assert get_season(7, "south") == "winter"  # Winter in south when summer in north


def test_is_us_holiday():
    """Test US holiday detection."""
    # New Year's Day
    assert is_us_holiday(pd.Timestamp("2024-01-01"))

    # Independence Day
    assert is_us_holiday(pd.Timestamp("2024-07-04"))

    # Christmas
    assert is_us_holiday(pd.Timestamp("2024-12-25"))

    # Not a holiday
    assert not is_us_holiday(pd.Timestamp("2024-03-15"))

    # Labor Day 2024 (Sept 2)
    assert is_us_holiday(pd.Timestamp("2024-09-02"))


def test_is_canadian_holiday():
    """Test Canadian holiday detection."""
    # New Year's Day
    assert is_canadian_holiday(pd.Timestamp("2024-01-01"))

    # Canada Day
    assert is_canadian_holiday(pd.Timestamp("2024-07-01"))

    # Christmas
    assert is_canadian_holiday(pd.Timestamp("2024-12-25"))

    # Not a holiday
    assert not is_canadian_holiday(pd.Timestamp("2024-03-15"))


# =============================================================================
# DatetimeFeatureEngineer Tests
# =============================================================================


def test_engineer_basic_features(sample_datetime_df):
    """Test basic datetime feature engineering."""
    engineer = DatetimeFeatureEngineer(
        include_cyclical=False,
        include_holidays=False,
    )

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check basic temporal features
    assert "start_time_year" in result.columns
    assert "start_time_month" in result.columns
    assert "start_time_day" in result.columns
    assert "start_time_hour" in result.columns
    assert "start_time_day_of_week" in result.columns
    assert "start_time_day_of_year" in result.columns
    assert "start_time_week_of_year" in result.columns
    assert "start_time_quarter" in result.columns

    # Check values
    assert result["start_time_year"].iloc[0] == 2024
    assert result["start_time_month"].iloc[0] == 1
    assert result["start_time_hour"].iloc[0] == 10


def test_engineer_weekend_features(sample_datetime_df):
    """Test weekend/weekday feature engineering."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check weekend/weekday features
    assert "start_time_is_weekend" in result.columns
    assert "start_time_is_weekday" in result.columns

    # Jan 1, 2024 is Monday (weekday)
    assert result["start_time_is_weekday"].iloc[0] == 1
    assert result["start_time_is_weekend"].iloc[0] == 0

    # June 15, 2024 is Saturday (weekend)
    assert result["start_time_is_weekend"].iloc[1] == 1
    assert result["start_time_is_weekday"].iloc[1] == 0


def test_engineer_part_of_day_features(sample_datetime_df):
    """Test part of day feature engineering."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check part of day features
    assert "start_time_part_of_day" in result.columns
    assert "start_time_is_morning" in result.columns
    assert "start_time_is_afternoon" in result.columns
    assert "start_time_is_evening" in result.columns
    assert "start_time_is_night" in result.columns

    # 10:00 is morning
    assert result["start_time_part_of_day"].iloc[0] == "morning"
    assert result["start_time_is_morning"].iloc[0] == 1

    # 14:00 is afternoon
    assert result["start_time_part_of_day"].iloc[1] == "afternoon"
    assert result["start_time_is_afternoon"].iloc[1] == 1


def test_engineer_working_hours_features(sample_datetime_df):
    """Test working hours feature engineering."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check working hours feature
    assert "start_time_is_working_hours" in result.columns

    # Jan 1, 2024 at 10am - Monday during work hours
    assert result["start_time_is_working_hours"].iloc[0] == 1

    # June 15, 2024 - Saturday (not working hours even if time is right)
    assert result["start_time_is_working_hours"].iloc[1] == 0


def test_engineer_season_features(sample_datetime_df):
    """Test season feature engineering."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check season features
    assert "start_time_season" in result.columns
    assert "start_time_is_winter" in result.columns
    assert "start_time_is_spring" in result.columns
    assert "start_time_is_summer" in result.columns
    assert "start_time_is_fall" in result.columns

    # January is winter
    assert result["start_time_season"].iloc[0] == "winter"
    assert result["start_time_is_winter"].iloc[0] == 1

    # June is summer
    assert result["start_time_season"].iloc[1] == "summer"
    assert result["start_time_is_summer"].iloc[1] == 1


def test_engineer_holiday_features(sample_datetime_df):
    """Test holiday feature engineering."""
    engineer = DatetimeFeatureEngineer(include_holidays=True, region="both")

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check holiday features
    assert "start_time_is_us_holiday" in result.columns
    assert "start_time_is_canadian_holiday" in result.columns
    assert "start_time_is_holiday" in result.columns

    # Dec 25 is Christmas (both US and Canada)
    assert result["start_time_is_us_holiday"].iloc[2] == 1
    assert result["start_time_is_canadian_holiday"].iloc[2] == 1
    assert result["start_time_is_holiday"].iloc[2] == 1

    # July 4 is US holiday only
    assert result["start_time_is_us_holiday"].iloc[3] == 1
    assert result["start_time_is_canadian_holiday"].iloc[3] == 0


def test_engineer_cyclical_features(sample_datetime_df):
    """Test cyclical encoding feature engineering."""
    engineer = DatetimeFeatureEngineer(include_cyclical=True)

    result = engineer.add_datetime_features(sample_datetime_df, "start_time")

    # Check cyclical features
    assert "start_time_hour_sin" in result.columns
    assert "start_time_hour_cos" in result.columns
    assert "start_time_day_of_week_sin" in result.columns
    assert "start_time_day_of_week_cos" in result.columns
    assert "start_time_month_sin" in result.columns
    assert "start_time_month_cos" in result.columns
    assert "start_time_day_of_year_sin" in result.columns
    assert "start_time_day_of_year_cos" in result.columns

    # Cyclical features should be between -1 and 1
    assert -1 <= result["start_time_hour_sin"].iloc[0] <= 1
    assert -1 <= result["start_time_hour_cos"].iloc[0] <= 1


def test_engineer_custom_prefix(sample_datetime_df):
    """Test custom prefix for feature names."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(
        sample_datetime_df,
        "start_time",
        prefix="custom_",
    )

    assert "custom_year" in result.columns
    assert "custom_month" in result.columns
    assert "custom_hour" in result.columns


def test_engineer_drop_original(sample_datetime_df):
    """Test dropping original datetime column."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_datetime_features(
        sample_datetime_df,
        "start_time",
        drop_original=True,
    )

    assert "start_time" not in result.columns
    assert "start_time_year" in result.columns


# =============================================================================
# Time-Since Features Tests
# =============================================================================


def test_add_time_since_features(sample_datetime_df):
    """Test time-since feature engineering."""
    engineer = DatetimeFeatureEngineer()

    result = engineer.add_time_since_features(
        sample_datetime_df,
        datetime_col="end_time",
        reference_col="start_time",
        prefix="duration_",
    )

    # Check time-since features
    assert "duration_total_seconds" in result.columns
    assert "duration_days" in result.columns
    assert "duration_hours" in result.columns
    assert "duration_minutes" in result.columns

    # Jan 1 10:00 to Jan 5 18:00 is 4 days 8 hours = 104 hours
    assert abs(result["duration_hours"].iloc[0] - 104) < 0.1


def test_add_time_until_features(sample_datetime_df):
    """Test time-until feature engineering."""
    engineer = DatetimeFeatureEngineer()

    reference = pd.Timestamp("2024-01-03T12:00:00")

    result = engineer.add_time_until_features(
        sample_datetime_df,
        datetime_col="end_time",
        reference_datetime=reference,
        prefix="time_left_",
    )

    # Check time-until features
    assert "time_left_total_seconds" in result.columns
    assert "time_left_days" in result.columns
    assert "time_left_hours" in result.columns
    assert "time_left_minutes" in result.columns
    assert "time_left_has_passed" in result.columns

    # Jan 3 12:00 to Jan 5 18:00 is about 2.25 days
    assert result["time_left_days"].iloc[0] > 2
    assert result["time_left_has_passed"].iloc[0] == 0


# =============================================================================
# Auction-Specific Features Tests
# =============================================================================


def test_add_auction_duration_features(sample_datetime_df):
    """Test auction duration feature engineering."""
    result = add_auction_duration_features(
        sample_datetime_df,
        start_col="start_time",
        end_col="end_time",
    )

    # Check duration features
    assert "auction_duration_total_seconds" in result.columns
    assert "auction_duration_days" in result.columns
    assert "auction_duration_hours" in result.columns
    assert "auction_duration_length_category" in result.columns

    # First auction: Jan 1 to Jan 5 = 4 days + 8 hours = 104 hours
    assert abs(result["auction_duration_hours"].iloc[0] - 104) < 0.1

    # Check length category
    assert result["auction_duration_length_category"].iloc[0] in [
        "short",
        "medium",
        "long",
        "very_long",
    ]


def test_add_bid_timing_features(sample_bid_df):
    """Test bid timing feature engineering."""
    result = add_bid_timing_features(sample_bid_df)

    # Check bid timing features
    assert "bid_timing_seconds_since_start" in result.columns
    assert "bid_timing_hours_since_start" in result.columns
    assert "bid_timing_seconds_until_end" in result.columns
    assert "bid_timing_hours_until_end" in result.columns
    assert "bid_timing_minutes_until_end" in result.columns
    assert "bid_timing_is_last_minute" in result.columns
    assert "bid_timing_pct_through_auction" in result.columns
    assert "bid_timing_position" in result.columns

    # First bid: 24 hours after start
    assert abs(result["bid_timing_hours_since_start"].iloc[0] - 24) < 0.1

    # Second bid: 90 seconds before end (last minute)
    assert result["bid_timing_is_last_minute"].iloc[1] == 1
    assert result["bid_timing_minutes_until_end"].iloc[1] < 2

    # Check bid positions
    assert result["bid_timing_position"].iloc[0] == "early"
    assert result["bid_timing_position"].iloc[1] == "last_minute"


# =============================================================================
# Convenience Functions Tests
# =============================================================================


def test_engineer_all_datetime_features(sample_datetime_df):
    """Test engineering features for multiple columns at once."""
    result = engineer_all_datetime_features(
        sample_datetime_df,
        datetime_columns=["start_time", "end_time"],
    )

    # Check that features were added for both columns
    assert "start_time_year" in result.columns
    assert "start_time_month" in result.columns
    assert "end_time_year" in result.columns
    assert "end_time_month" in result.columns

    # Original columns should still be present
    assert "start_time" in result.columns
    assert "end_time" in result.columns


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_handle_missing_values():
    """Test handling of missing datetime values."""
    df = pd.DataFrame(
        {
            "datetime_col": ["2024-01-01T10:00:00", None, "2024-06-15T14:00:00"],
        }
    )

    engineer = DatetimeFeatureEngineer()
    result = engineer.add_datetime_features(df, "datetime_col")

    # Should handle NaT values gracefully
    assert len(result) == 3
    assert pd.isna(result["datetime_col_year"].iloc[1])


def test_handle_string_datetimes():
    """Test automatic parsing of string datetime columns."""
    df = pd.DataFrame(
        {
            "datetime_col": ["2024-01-01T10:00:00", "2024-06-15T14:00:00"],
        }
    )

    # Column is string type (either 'object' or StringDtype)
    assert not pd.api.types.is_datetime64_any_dtype(df["datetime_col"])

    engineer = DatetimeFeatureEngineer()
    result = engineer.add_datetime_features(df, "datetime_col")

    # Should parse and create features
    assert "datetime_col_year" in result.columns
    assert result["datetime_col_year"].iloc[0] == 2024


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame(
        {
            "datetime_col": pd.Series([], dtype="datetime64[ns]"),
        }
    )

    engineer = DatetimeFeatureEngineer()
    result = engineer.add_datetime_features(df, "datetime_col")

    # Should not crash on empty DataFrame
    assert len(result) == 0
    assert "datetime_col_year" in result.columns
