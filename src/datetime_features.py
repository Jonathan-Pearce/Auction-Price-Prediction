# =============================================================================
# Auction Price Prediction - Datetime Feature Engineering
# =============================================================================
"""
Comprehensive datetime feature engineering for auction data.

This module provides unified datetime transformations for all datetime variables
in the auction dataset including:
- Auction start/end times
- Bid timestamps
- Pickup times
- Any other temporal variables

Features include:
- Basic temporal: hour, day, month, year, day_of_week, day_of_year, week_of_year
- Weekend/weekday indicators
- Part of day categorization (morning, afternoon, evening, night)
- Working hours indicators
- Season detection (spring, summer, fall, winter)
- Holiday detection
- Time-relative features (duration, time_until, time_since)
- Auction-specific timing features

Example:
    >>> import pandas as pd
    >>> from src.datetime_features import DatetimeFeatureEngineer
    >>>
    >>> df = pd.DataFrame({
    ...     'auction_end_time': ['2024-01-05T18:00:00', '2024-06-15T14:00:00']
    ... })
    >>>
    >>> engineer = DatetimeFeatureEngineer()
    >>> features = engineer.add_datetime_features(df, 'auction_end_time')
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from loguru import logger

# =============================================================================
# Holiday Detection
# =============================================================================


def is_us_holiday(date: pd.Timestamp | datetime) -> bool:
    """
    Check if a date is a major US holiday.

    Includes:
    - New Year's Day (Jan 1)
    - Memorial Day (Last Monday of May)
    - Independence Day (July 4)
    - Labor Day (First Monday of September)
    - Thanksgiving (Fourth Thursday of November)
    - Christmas (Dec 25)

    Args:
        date: Date to check

    Returns:
        True if date is a holiday
    """
    if isinstance(date, datetime):
        date = pd.Timestamp(date)

    month = date.month
    day = date.day

    # Fixed holidays
    if (
        (month == 1 and day == 1)
        or (month == 7 and day == 4)
        or (month == 12 and day == 25)
    ):
        return True

    # Memorial Day - Last Monday of May
    if month == 5 and date.dayofweek == 0:
        # Check if this is the last Monday
        next_week = date + pd.Timedelta(days=7)
        if next_week.month != 5:
            return True

    # Labor Day - First Monday of September
    if month == 9 and date.dayofweek == 0 and day <= 7:
        return True

    # Thanksgiving - Fourth Thursday of November
    if month == 11 and date.dayofweek == 3:
        # Check if this is the 4th Thursday (22-28)
        if 22 <= day <= 28:
            return True

    return False


def is_canadian_holiday(date: pd.Timestamp | datetime) -> bool:
    """
    Check if a date is a major Canadian holiday.

    Includes:
    - New Year's Day (Jan 1)
    - Canada Day (July 1)
    - Labour Day (First Monday of September)
    - Thanksgiving (Second Monday of October)
    - Christmas (Dec 25)

    Args:
        date: Date to check

    Returns:
        True if date is a holiday
    """
    if isinstance(date, datetime):
        date = pd.Timestamp(date)

    month = date.month
    day = date.day

    # Fixed holidays
    if (
        (month == 1 and day == 1)
        or (month == 7 and day == 1)
        or (month == 12 and day == 25)
    ):
        return True

    # Labour Day - First Monday of September
    if month == 9 and date.dayofweek == 0 and day <= 7:
        return True

    # Thanksgiving - Second Monday of October
    if month == 10 and date.dayofweek == 0 and 8 <= day <= 14:
        return True

    return False


# =============================================================================
# Part of Day Classification
# =============================================================================


def get_part_of_day(hour: int) -> str:
    """
    Classify hour into part of day.

    Classification:
    - night: 0-5 (12am-6am)
    - morning: 6-11 (6am-12pm)
    - afternoon: 12-17 (12pm-6pm)
    - evening: 18-23 (6pm-12am)

    Args:
        hour: Hour of day (0-23)

    Returns:
        Part of day string
    """
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def get_season(month: int, hemisphere: Literal["north", "south"] = "north") -> str:
    """
    Get season from month.

    Northern hemisphere:
    - winter: Dec, Jan, Feb (12, 1, 2)
    - spring: Mar, Apr, May (3, 4, 5)
    - summer: Jun, Jul, Aug (6, 7, 8)
    - fall: Sep, Oct, Nov (9, 10, 11)

    Southern hemisphere is offset by 6 months.

    Args:
        month: Month number (1-12)
        hemisphere: 'north' or 'south'

    Returns:
        Season name
    """
    if hemisphere == "south":
        # Shift by 6 months
        month = ((month + 6 - 1) % 12) + 1

    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:  # month in [9, 10, 11]
        return "fall"


# =============================================================================
# Main Feature Engineering Class
# =============================================================================


class DatetimeFeatureEngineer:
    """
    Engineer datetime features for auction data.

    Provides comprehensive datetime transformations that can be applied
    to any datetime column in the dataset.
    """

    def __init__(
        self,
        include_cyclical: bool = True,
        include_holidays: bool = True,
        region: Literal["us", "canada", "both"] = "both",
    ):
        """
        Initialize datetime feature engineer.

        Args:
            include_cyclical: Whether to include sin/cos cyclical encodings
            include_holidays: Whether to include holiday detection
            region: Holiday region ('us', 'canada', or 'both')
        """
        self.include_cyclical = include_cyclical
        self.include_holidays = include_holidays
        self.region = region

    def add_datetime_features(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        prefix: str | None = None,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Add comprehensive datetime features to a DataFrame.

        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column to engineer features from
            prefix: Prefix for new columns (default: datetime_col + '_')
            drop_original: Whether to drop the original datetime column

        Returns:
            DataFrame with additional datetime features
        """
        result = df.copy()

        # Ensure datetime column is parsed
        if not pd.api.types.is_datetime64_any_dtype(result[datetime_col]):
            result[datetime_col] = pd.to_datetime(result[datetime_col], errors="coerce")

        # Set prefix
        if prefix is None:
            prefix = f"{datetime_col}_"

        dt_series = result[datetime_col]

        # Basic temporal features
        result[f"{prefix}year"] = dt_series.dt.year
        result[f"{prefix}month"] = dt_series.dt.month
        result[f"{prefix}day"] = dt_series.dt.day
        result[f"{prefix}hour"] = dt_series.dt.hour
        result[f"{prefix}minute"] = dt_series.dt.minute
        result[f"{prefix}day_of_week"] = dt_series.dt.dayofweek  # 0=Monday, 6=Sunday
        result[f"{prefix}day_of_year"] = dt_series.dt.dayofyear
        result[f"{prefix}week_of_year"] = dt_series.dt.isocalendar().week
        result[f"{prefix}quarter"] = dt_series.dt.quarter

        # Weekend/weekday
        result[f"{prefix}is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
        result[f"{prefix}is_weekday"] = (dt_series.dt.dayofweek < 5).astype(int)

        # Part of day
        result[f"{prefix}part_of_day"] = dt_series.dt.hour.apply(get_part_of_day)
        result[f"{prefix}is_morning"] = (
            result[f"{prefix}part_of_day"] == "morning"
        ).astype(int)
        result[f"{prefix}is_afternoon"] = (
            result[f"{prefix}part_of_day"] == "afternoon"
        ).astype(int)
        result[f"{prefix}is_evening"] = (
            result[f"{prefix}part_of_day"] == "evening"
        ).astype(int)
        result[f"{prefix}is_night"] = (
            result[f"{prefix}part_of_day"] == "night"
        ).astype(int)

        # Working hours (9am-5pm, Mon-Fri)
        result[f"{prefix}is_working_hours"] = (
            (dt_series.dt.hour >= 9)
            & (dt_series.dt.hour < 17)
            & (dt_series.dt.dayofweek < 5)
        ).astype(int)

        # Business day (Mon-Fri, not holiday)
        result[f"{prefix}is_business_day"] = (dt_series.dt.dayofweek < 5).astype(int)

        # Season
        result[f"{prefix}season"] = dt_series.dt.month.apply(get_season)
        result[f"{prefix}is_winter"] = (result[f"{prefix}season"] == "winter").astype(
            int
        )
        result[f"{prefix}is_spring"] = (result[f"{prefix}season"] == "spring").astype(
            int
        )
        result[f"{prefix}is_summer"] = (result[f"{prefix}season"] == "summer").astype(
            int
        )
        result[f"{prefix}is_fall"] = (result[f"{prefix}season"] == "fall").astype(int)

        # Holiday detection
        if self.include_holidays:
            if self.region in ["us", "both"]:
                result[f"{prefix}is_us_holiday"] = dt_series.apply(
                    is_us_holiday
                ).astype(int)
            if self.region in ["canada", "both"]:
                result[f"{prefix}is_canadian_holiday"] = dt_series.apply(
                    is_canadian_holiday
                ).astype(int)
            if self.region == "both":
                result[f"{prefix}is_holiday"] = (
                    result[f"{prefix}is_us_holiday"]
                    | result[f"{prefix}is_canadian_holiday"]
                ).astype(int)

        # Cyclical encoding for periodic features
        if self.include_cyclical:
            import numpy as np

            # Hour (24-hour cycle)
            result[f"{prefix}hour_sin"] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
            result[f"{prefix}hour_cos"] = np.cos(2 * np.pi * dt_series.dt.hour / 24)

            # Day of week (7-day cycle)
            result[f"{prefix}day_of_week_sin"] = np.sin(
                2 * np.pi * dt_series.dt.dayofweek / 7
            )
            result[f"{prefix}day_of_week_cos"] = np.cos(
                2 * np.pi * dt_series.dt.dayofweek / 7
            )

            # Month (12-month cycle)
            result[f"{prefix}month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
            result[f"{prefix}month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)

            # Day of year (365-day cycle)
            result[f"{prefix}day_of_year_sin"] = np.sin(
                2 * np.pi * dt_series.dt.dayofyear / 365
            )
            result[f"{prefix}day_of_year_cos"] = np.cos(
                2 * np.pi * dt_series.dt.dayofyear / 365
            )

        # Drop original if requested
        if drop_original:
            result = result.drop(columns=[datetime_col])

        logger.debug(f"Added datetime features for {datetime_col} with prefix {prefix}")

        return result

    def add_time_since_features(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        reference_col: str,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Add time-since features (duration between two datetimes).

        Creates features for:
        - Total seconds elapsed
        - Days elapsed
        - Hours elapsed
        - Minutes elapsed

        Args:
            df: Input DataFrame
            datetime_col: Target datetime column
            reference_col: Reference datetime column (earlier time)
            prefix: Prefix for new columns

        Returns:
            DataFrame with time-since features
        """
        result = df.copy()

        # Ensure both columns are datetime
        for col in [datetime_col, reference_col]:
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors="coerce")

        # Set prefix
        if prefix is None:
            prefix = f"{datetime_col}_since_{reference_col}_"

        # Calculate time difference
        time_diff = result[datetime_col] - result[reference_col]

        result[f"{prefix}total_seconds"] = time_diff.dt.total_seconds()
        result[f"{prefix}days"] = time_diff.dt.total_seconds() / (24 * 3600)
        result[f"{prefix}hours"] = time_diff.dt.total_seconds() / 3600
        result[f"{prefix}minutes"] = time_diff.dt.total_seconds() / 60

        logger.debug(f"Added time-since features: {datetime_col} since {reference_col}")

        return result

    def add_time_until_features(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        reference_datetime: pd.Timestamp | datetime | None = None,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Add time-until features (duration until a target datetime).

        Useful for features like "time until auction closes" when evaluated
        at a specific point in time (e.g., when a bid is placed).

        Args:
            df: Input DataFrame
            datetime_col: Target datetime column (future time)
            reference_datetime: Reference point in time (default: now)
            prefix: Prefix for new columns

        Returns:
            DataFrame with time-until features
        """
        result = df.copy()

        # Ensure datetime column is parsed
        if not pd.api.types.is_datetime64_any_dtype(result[datetime_col]):
            result[datetime_col] = pd.to_datetime(result[datetime_col], errors="coerce")

        # Set reference datetime
        if reference_datetime is None:
            reference_datetime = pd.Timestamp.now()
        elif isinstance(reference_datetime, datetime):
            reference_datetime = pd.Timestamp(reference_datetime)

        # Set prefix
        if prefix is None:
            prefix = f"time_until_{datetime_col}_"

        # Calculate time difference
        time_diff = result[datetime_col] - reference_datetime

        result[f"{prefix}total_seconds"] = time_diff.dt.total_seconds()
        result[f"{prefix}days"] = time_diff.dt.total_seconds() / (24 * 3600)
        result[f"{prefix}hours"] = time_diff.dt.total_seconds() / 3600
        result[f"{prefix}minutes"] = time_diff.dt.total_seconds() / 60

        # Indicator if event has already passed
        result[f"{prefix}has_passed"] = (time_diff.dt.total_seconds() < 0).astype(int)

        logger.debug(f"Added time-until features for {datetime_col}")

        return result


# =============================================================================
# Auction-Specific Datetime Features
# =============================================================================


def add_auction_duration_features(
    df: pd.DataFrame,
    start_col: str = "start_time",
    end_col: str = "end_time",
    prefix: str = "auction_duration_",
) -> pd.DataFrame:
    """
    Add auction duration features.

    Args:
        df: Input DataFrame with auction data
        start_col: Column name for auction start time
        end_col: Column name for auction end time
        prefix: Prefix for new columns

    Returns:
        DataFrame with duration features
    """
    result = df.copy()

    # Ensure datetime columns are parsed
    for col in [start_col, end_col]:
        if not pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = pd.to_datetime(result[col], errors="coerce")

    duration = result[end_col] - result[start_col]

    result[f"{prefix}total_seconds"] = duration.dt.total_seconds()
    result[f"{prefix}days"] = duration.dt.total_seconds() / (24 * 3600)
    result[f"{prefix}hours"] = duration.dt.total_seconds() / 3600

    # Categorize auction length
    hours = duration.dt.total_seconds() / 3600
    result[f"{prefix}length_category"] = pd.cut(
        hours,
        bins=[0, 24, 72, 168, float("inf")],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
    )

    logger.debug("Added auction duration features")

    return result


def add_bid_timing_features(
    df: pd.DataFrame,
    bid_time_col: str = "bid_time",
    auction_start_col: str = "auction_start_time",
    auction_end_col: str = "auction_end_time",
    prefix: str = "bid_timing_",
) -> pd.DataFrame:
    """
    Add bid timing features relative to auction lifecycle.

    Features include:
    - Time since auction start
    - Time until auction end
    - Bid position (early, middle, late, last_minute)
    - Percentage through auction

    Args:
        df: Input DataFrame with bid data
        bid_time_col: Column name for bid timestamp
        auction_start_col: Column name for auction start time
        auction_end_col: Column name for auction end time (scheduled)
        prefix: Prefix for new columns

    Returns:
        DataFrame with bid timing features
    """
    result = df.copy()

    # Ensure datetime columns are parsed
    for col in [bid_time_col, auction_start_col, auction_end_col]:
        if col in result.columns and not pd.api.types.is_datetime64_any_dtype(
            result[col]
        ):
            result[col] = pd.to_datetime(result[col], errors="coerce")

    # Time since auction start
    if auction_start_col in result.columns:
        time_since_start = result[bid_time_col] - result[auction_start_col]
        result[f"{prefix}seconds_since_start"] = time_since_start.dt.total_seconds()
        result[f"{prefix}hours_since_start"] = (
            time_since_start.dt.total_seconds() / 3600
        )

    # Time until auction end
    if auction_end_col in result.columns:
        time_until_end = result[auction_end_col] - result[bid_time_col]
        result[f"{prefix}seconds_until_end"] = time_until_end.dt.total_seconds()
        result[f"{prefix}hours_until_end"] = time_until_end.dt.total_seconds() / 3600
        result[f"{prefix}minutes_until_end"] = time_until_end.dt.total_seconds() / 60

        # Last-minute bid detection (within 2 minutes of close - soft close trigger)
        result[f"{prefix}is_last_minute"] = (
            time_until_end.dt.total_seconds() <= 120
        ).astype(int)

        # Bid position in auction lifecycle
        if auction_start_col in result.columns:
            auction_duration = result[auction_end_col] - result[auction_start_col]
            time_elapsed = result[bid_time_col] - result[auction_start_col]

            # Percentage through auction (0-1)
            result[f"{prefix}pct_through_auction"] = (
                time_elapsed.dt.total_seconds() / auction_duration.dt.total_seconds()
            )

            # Categorize bid position
            pct = result[f"{prefix}pct_through_auction"]
            result[f"{prefix}position"] = pd.cut(
                pct,
                bins=[0, 0.25, 0.75, 0.95, 1.0],
                labels=["early", "middle", "late", "last_minute"],
                include_lowest=True,
            )

    logger.debug("Added bid timing features")

    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def engineer_all_datetime_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
    include_cyclical: bool = True,
    include_holidays: bool = True,
    region: Literal["us", "canada", "both"] = "both",
) -> pd.DataFrame:
    """
    Engineer datetime features for multiple columns at once.

    Args:
        df: Input DataFrame
        datetime_columns: List of datetime column names
        include_cyclical: Whether to include cyclical encodings
        include_holidays: Whether to include holiday detection
        region: Holiday region

    Returns:
        DataFrame with all datetime features added
    """
    engineer = DatetimeFeatureEngineer(
        include_cyclical=include_cyclical,
        include_holidays=include_holidays,
        region=region,
    )

    result = df.copy()

    for col in datetime_columns:
        if col in result.columns:
            logger.info(f"Engineering datetime features for: {col}")
            result = engineer.add_datetime_features(result, col)

    logger.info(
        f"Completed datetime feature engineering for {len(datetime_columns)} columns"
    )

    return result
