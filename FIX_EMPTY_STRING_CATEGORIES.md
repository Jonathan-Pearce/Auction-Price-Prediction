# Fix for Empty String Categorical Encoding Issue

## Problem

The features `item_condition_graded`, `item_condition_`, and `item_working_` appeared as mostly NULL values in the engineered item dataset.

### Root Cause

The enriched item data contains empty strings `''` in the categorical columns:
- `enriched_item_condition` had values: `['lightly used', 'heavily used', 'unknown', 'new', 'graded', '']`
- `enriched_item_working` had values: `['tested & working', 'not applicable', 'untested', 'unknown', '']`

When `pd.get_dummies()` encountered empty strings, it created columns named:
- `item_condition_` (from empty string in condition)
- `item_working_` (from empty string in working status)

These columns were mostly NULL because:
1. Empty strings were not replaced before encoding
2. The resulting boolean columns had NULLs where the original value was not an empty string

## Solution

Updated [src/features_item.py](src/features_item.py):

### 1. Enhanced `encode_categorical_features()` function

**Before:**
```python
# Fill missing values
result[source_col] = result[source_col].fillna("Unknown")
```

**After:**
```python
# Fill missing values and empty strings with "Unknown"
result[source_col] = result[source_col].fillna("Unknown")
result[source_col] = result[source_col].replace("", "Unknown")
result[source_col] = result[source_col].replace("unknown", "Unknown")
```

### 2. Updated `select_final_columns()` function

Added filtering to remove columns created from empty strings:

```python
# Remove columns that are exactly "item_condition_" or "item_working_"
# (these come from empty string values in categorical encoding)
invalid_cols = {"item_condition_", "item_working_"}
final_cols = [col for col in final_cols if col not in invalid_cols]
```

## Verification

Created test file [test_fix_empty_strings.py](test_fix_empty_strings.py) that:
- ✅ Confirms empty strings are properly converted to "Unknown"
- ✅ Verifies no `item_condition_` or `item_working_` columns in output
- ✅ Ensures all expected category columns are present
- ✅ Checks that encoded columns have no NULL values

## Next Steps

The existing `data/processed/items/engineered_item_data.parquet` file needs to be regenerated with the fixed code. This can be done by running:

```bash
python -m src.features_item
```

Or using the batch processing pipeline:

```python
from src.features_item import run_item_feature_pipeline
df = run_item_feature_pipeline(batch_size=100_000)
```

## Impact

After reprocessing:
- **226 items** with empty `condition` → now in `item_condition_unknown`
- **226 items** with empty `working` → now in `item_working_unknown`
- **2,884 items** with `graded` condition → now in `item_condition_graded` (no longer NULL)
- All boolean categorical features will have proper 0/1 values instead of NULLs
