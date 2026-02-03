# Datetime Feature Engineering

This document describes the comprehensive datetime feature engineering capabilities for the Auction Price Prediction project.

## Overview

The datetime feature engineering module (`src/datetime_features.py`) provides unified transformations for all datetime variables in the auction dataset, including:
- Auction start/end times
- Bid timestamps  
- Pickup times
- Any other temporal variables

## Features

### 1. Basic Temporal Features

Extract standard datetime components:
- `year`, `month`, `day`, `hour`, `minute`
- `day_of_week` (0=Monday, 6=Sunday)
- `day_of_year` (1-365/366)
- `week_of_year` (1-52)
- `quarter` (1-4)

### 2. Weekend/Weekday Indicators

Binary flags for time categorization:
- `is_weekend`: 1 if Saturday/Sunday, 0 otherwise
- `is_weekday`: 1 if Monday-Friday, 0 otherwise
- `is_business_day`: 1 if weekday and not a holiday

### 3. Part of Day Classification

Categorizes hours into meaningful periods:
- **Night**: 0-5 (12am-6am)
- **Morning**: 6-11 (6am-12pm)
- **Afternoon**: 12-17 (12pm-6pm)
- **Evening**: 18-23 (6pm-12am)

Includes both categorical variable and one-hot encoded indicators:
- `part_of_day`: categorical string
- `is_morning`, `is_afternoon`, `is_evening`, `is_night`: binary flags

### 4. Working Hours

Detection of standard business hours:
- `is_working_hours`: 1 if Monday-Friday 9am-5pm, 0 otherwise

### 5. Season Detection

Categorizes dates by meteorological season:
- **Winter**: December, January, February
- **Spring**: March, April, May
- **Summer**: June, July, August
- **Fall**: September, October, November

Includes both categorical variable and one-hot encoded indicators:
- `season`: categorical string
- `is_winter`, `is_spring`, `is_summer`, `is_fall`: binary flags

Supports both Northern and Southern hemisphere seasons.

### 6. Holiday Detection

Identifies major holidays for US and Canadian markets:

**US Holidays:**
- New Year's Day (January 1)
- Memorial Day (Last Monday of May)
- Independence Day (July 4)
- Labor Day (First Monday of September)
- Thanksgiving (Fourth Thursday of November)
- Christmas (December 25)

**Canadian Holidays:**
- New Year's Day (January 1)
- Canada Day (July 1)
- Labour Day (First Monday of September)
- Thanksgiving (Second Monday of October)
- Christmas (December 25)

Features:
- `is_us_holiday`: binary flag for US holidays
- `is_canadian_holiday`: binary flag for Canadian holidays
- `is_holiday`: binary flag for either region (when region='both')

### 7. Cyclical Encoding

Sine/cosine transformations for periodic features, preserving the circular nature of time:

For each periodic feature:
- Hour (24-hour cycle): `hour_sin`, `hour_cos`
- Day of week (7-day cycle): `day_of_week_sin`, `day_of_week_cos`
- Month (12-month cycle): `month_sin`, `month_cos`
- Day of year (365-day cycle): `day_of_year_sin`, `day_of_year_cos`

**Why use cyclical encoding?**
- ML models can learn that hour 23 and hour 0 are close
- Avoids treating time as purely linear
- Prevents artificial distance between adjacent periods

### 8. Time Duration Features

Calculate elapsed time between two datetime points:
- `total_seconds`: Total seconds elapsed
- `days`: Days elapsed (fractional)
- `hours`: Hours elapsed (fractional)
- `minutes`: Minutes elapsed (fractional)

### 9. Time-Until Features

Calculate remaining time until a target datetime:
- `total_seconds`: Seconds remaining
- `days`: Days remaining
- `hours`: Hours remaining
- `minutes`: Minutes remaining
- `has_passed`: Binary flag if target time has passed

### 10. Auction-Specific Features

#### Auction Duration
Features describing total auction length:
- `auction_duration_total_seconds`
- `auction_duration_days`
- `auction_duration_hours`
- `auction_duration_length_category`: Categorized as 'short', 'medium', 'long', 'very_long'

#### Bid Timing
Features describing when a bid occurs relative to auction lifecycle:
- `bid_timing_seconds_since_start`: Time since auction start
- `bid_timing_hours_since_start`: Hours since auction start
- `bid_timing_seconds_until_end`: Time until auction closes
- `bid_timing_hours_until_end`: Hours until auction closes
- `bid_timing_minutes_until_end`: Minutes until auction closes
- `bid_timing_is_last_minute`: Binary flag for soft-close triggering bids (within 2 minutes)
- `bid_timing_pct_through_auction`: Percentage of auction elapsed (0-1)
- `bid_timing_position`: Categorical ('early', 'middle', 'late', 'last_minute')

## Usage

### Basic Usage

```python
from src.datetime_features import DatetimeFeatureEngineer
import pandas as pd

# Initialize engineer
engineer = DatetimeFeatureEngineer(
    include_cyclical=True,
    include_holidays=True,
    region='both',  # 'us', 'canada', or 'both'
)

# Add features to DataFrame
df = pd.DataFrame({
    'auction_id': [1, 2, 3],
    'end_time': ['2024-01-05T18:00:00', '2024-06-15T14:00:00', '2024-12-25T09:00:00']
})

df_with_features = engineer.add_datetime_features(df, 'end_time')
```

### Multiple Columns

```python
from src.datetime_features import engineer_all_datetime_features

df_with_features = engineer_all_datetime_features(
    df,
    datetime_columns=['start_time', 'end_time', 'bid_time'],
    include_cyclical=True,
    include_holidays=True,
)
```

### Auction Duration

```python
from src.datetime_features import add_auction_duration_features

df = pd.DataFrame({
    'auction_id': [1, 2],
    'start_time': ['2024-01-01T10:00:00', '2024-06-15T14:00:00'],
    'end_time': ['2024-01-05T18:00:00', '2024-06-20T14:00:00']
})

df_with_duration = add_auction_duration_features(df)
```

### Bid Timing

```python
from src.datetime_features import add_bid_timing_features

df = pd.DataFrame({
    'bid_id': [1, 2, 3],
    'bid_time': ['2024-01-02T10:00:00', '2024-01-05T17:58:30', '2024-01-03T14:00:00'],
    'auction_start_time': ['2024-01-01T10:00:00'] * 3,
    'auction_end_time': ['2024-01-05T18:00:00'] * 3,
})

df_with_timing = add_bid_timing_features(df)
```

### Integration with Feature Pipeline

```python
from src.features import engineer_auction_features, engineer_bid_features

# Auction features (includes datetime transformations)
auction_features = engineer_auction_features(auctions_df)

# Bid features (includes datetime transformations)
bid_features = engineer_bid_features(bids_df)
```

## Configuration Options

### DatetimeFeatureEngineer

```python
DatetimeFeatureEngineer(
    include_cyclical: bool = True,     # Include sin/cos encodings
    include_holidays: bool = True,     # Include holiday detection
    region: Literal["us", "canada", "both"] = "both"  # Holiday region
)
```

### add_datetime_features

```python
engineer.add_datetime_features(
    df: pd.DataFrame,
    datetime_col: str,              # Column to transform
    prefix: str | None = None,      # Custom prefix (default: column_name_)
    drop_original: bool = False,    # Whether to drop original column
)
```

## Examples

See `examples/datetime_features_demo.py` for comprehensive demonstrations of all features.

Run the demo:
```bash
python examples/datetime_features_demo.py
```

## Testing

Comprehensive test suite available in `tests/test_datetime_features.py`.

Run tests:
```bash
pytest tests/test_datetime_features.py -v
```

## Feature Count

For a single datetime column with all options enabled:
- Basic temporal: 9 features
- Weekend/weekday: 3 features
- Part of day: 5 features (1 categorical + 4 binary)
- Working hours: 1 feature
- Season: 5 features (1 categorical + 4 binary)
- Holidays: 2-3 features (depending on region)
- Cyclical: 8 features (4 pairs of sin/cos)

**Total: ~35-40 features per datetime column**

## Best Practices

1. **Use cyclical encoding for ML models**: Tree-based models can work with integer encodings, but neural networks benefit from cyclical encoding.

2. **Consider feature selection**: Not all features may be relevant for your specific use case. Use feature importance analysis to identify the most predictive features.

3. **Handle missing values**: The module gracefully handles NaT values, but consider your imputation strategy beforehand.

4. **Timezone awareness**: Ensure all datetime columns are in the same timezone before feature engineering.

5. **Custom prefixes**: Use descriptive prefixes when adding features for multiple datetime columns to avoid confusion.

## Performance

- Feature engineering is vectorized using pandas operations for efficiency
- Processing ~10,000 rows typically takes <1 second
- Cyclical encoding adds minimal overhead due to numpy vectorization

## Future Enhancements

Potential additions for future versions:
- Additional holidays (UK, EU, etc.)
- Lunar calendar features
- Fiscal year calculations
- Custom business calendars
- Time zone conversions
- Relative date features (days_until_weekend, days_since_holiday)
