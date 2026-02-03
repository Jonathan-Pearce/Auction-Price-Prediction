# =============================================================================
# Auction Price Prediction - Feature Engineering
# =============================================================================
"""
Feature engineering pipeline for auction price prediction.

Transforms raw auction, item, and bid data into features for ML models:
- Tabular features: Numerical and categorical from auction/item metadata
- Datetime features: Temporal transformations for all datetime variables
- Text features: Processed item descriptions and titles
- Image features: Extracted from item photos (via pretrained models)
- Sequential features: Bid history time series
"""

import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from src.datetime_features import (
    DatetimeFeatureEngineer,
    add_auction_duration_features,
    add_bid_timing_features,
)

# =============================================================================
# Tabular Features
# =============================================================================


def extract_pickup_windows(auction_removal_info: str) -> dict[str, Any]:
    """
    Extract pickup window features from auction_removal_info HTML text.
    
    Parses the HTML-formatted pickup information to extract:
    - Number of pickup windows/categories
    - Total hours of pickup availability
    - Start hour of first pickup window
    - End hour of last pickup window
    - Day of week for pickup
    - Whether pickup is split by categories
    
    Args:
        auction_removal_info: HTML string containing pickup information
        
    Returns:
        Dictionary with pickup window features:
        - num_pickup_windows: int - Count of distinct pickup windows
        - total_pickup_hours: float - Total hours across all windows
        - first_pickup_start_hour: float - Hour of day (0-23.99) when first pickup starts
        - last_pickup_end_hour: float - Hour of day (0-23.99) when last pickup ends
        - pickup_day_of_week: int - Day of week (0=Monday, 6=Sunday), None if not found
        - has_category_windows: bool - True if pickup has category-based sub-windows
        
    Example input patterns:
        "Pickup: Saturday, March 14 EDT, 9AM - 12 NOON"
        "Pickup: Friday, March 13 EDT, 4PM - 7PM"
        "Category A: 9AM - 11AM" (with main pickup line)
    """
    # Default values
    result = {
        "num_pickup_windows": 0,
        "total_pickup_hours": 0.0,
        "first_pickup_start_hour": None,
        "last_pickup_end_hour": None,
        "pickup_day_of_week": None,
        "has_category_windows": False,
    }
    
    if not auction_removal_info or pd.isna(auction_removal_info):
        return result
    
    try:
        # Parse HTML to get clean text
        soup = BeautifulSoup(auction_removal_info, "html.parser")
        text = soup.get_text()
        
        # Extract day of week from main pickup line
        day_pattern = r"Pickup:\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        day_match = re.search(day_pattern, text, re.IGNORECASE)
        if day_match:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_name = day_match.group(1).lower()
            result["pickup_day_of_week"] = days.index(day_name)
        
        # Check for category-based windows
        has_categories = bool(re.search(r"Cat(egory)?\s+[A-Z]:", text, re.IGNORECASE))
        result["has_category_windows"] = has_categories
        
        # Extract all time ranges
        # Patterns to match:
        # - "9AM - 12 NOON", "4PM - 7PM", "12:00 Noon - 04:00 PM"
        # - "9AM - 11AM" (in category lines)
        time_ranges = []
        
        # Main pickup time pattern
        main_time_pattern = r"(\d{1,2}):?(\d{2})?\s*(AM|PM|NOON|Noon)\s*-\s*(\d{1,2}):?(\d{2})?\s*(AM|PM|NOON|Noon)"
        
        for match in re.finditer(main_time_pattern, text, re.IGNORECASE):
            start_hour = int(match.group(1))
            start_min = int(match.group(2)) if match.group(2) else 0
            start_period = match.group(3).upper()
            end_hour = int(match.group(4))
            end_min = int(match.group(5)) if match.group(5) else 0
            end_period = match.group(6).upper()
            
            # Validate hours are in valid 12-hour format (1-12)
            if not (1 <= start_hour <= 12) or not (1 <= end_hour <= 12):
                logger.debug(
                    f"Invalid hour values: start={start_hour}, end={end_hour}. "
                    f"Match: '{match.group(0)}'. Skipping."
                )
                continue
            
            # Validate minutes (0-59)
            if start_min >= 60 or end_min >= 60:
                logger.debug(
                    f"Invalid minute values: start_min={start_min}, end_min={end_min}. "
                    f"Match: '{match.group(0)}'. Skipping."
                )
                continue
            
            # Convert to 24-hour format
            if "NOON" in start_period:
                start_hour_24 = 12
            elif start_period == "PM" and start_hour != 12:
                start_hour_24 = start_hour + 12
            elif start_period == "AM" and start_hour == 12:
                start_hour_24 = 0
            else:
                start_hour_24 = start_hour
                
            if "NOON" in end_period:
                end_hour_24 = 12
            elif end_period == "PM" and end_hour != 12:
                end_hour_24 = end_hour + 12
            elif end_period == "AM" and end_hour == 12:
                end_hour_24 = 0
            else:
                end_hour_24 = end_hour
            
            # Convert to decimal hours
            start_decimal = start_hour_24 + start_min / 60.0
            end_decimal = end_hour_24 + end_min / 60.0
            
            # Handle cross-midnight time ranges (e.g., 11PM - 2AM next day)
            # If end time is before or equal to start time, check if it spans to next day
            if end_decimal <= start_decimal:
                # Skip if same time (zero duration)
                if end_decimal == start_decimal:
                    logger.debug(
                        f"Zero-duration time range: {start_decimal:.2f} to {end_decimal:.2f}. "
                        f"Match: '{match.group(0)}'. Skipping."
                    )
                    continue
                
                # For pickup windows, it's common to span midnight
                # Valid cross-midnight patterns:
                # 1. Late evening to next day: start ≥ 8PM AND (end is AM OR end ≤ 6PM)
                #    Examples: "11PM - 2AM", "11PM - 2PM", "9PM - 3PM"
                # 2. Morning/midday to early next morning: start < 8PM AND end ≤ 6AM
                #    Examples: "9AM - 1AM", "10AM - 2AM", "12PM - 3AM"
                # Invalid: Late evening to late evening (e.g., "9:30PM - 8PM")
                
                is_valid_cross_midnight = False
                
                if start_hour_24 >= 20:  # Late evening start
                    # End must be early (≤ 6PM) to be valid next-day
                    if end_hour_24 <= 18:
                        is_valid_cross_midnight = True
                elif end_hour_24 <= 6:  # Very early end (1-6 AM)
                    # Any start before 8PM going to early morning is valid
                    is_valid_cross_midnight = True
                
                if is_valid_cross_midnight:
                    # Add 24 hours to end time to represent next day
                    end_decimal += 24.0
                    logger.debug(
                        f"Cross-midnight range detected: {start_decimal:.2f} to {end_decimal:.2f}. "
                        f"Match: '{match.group(0)}'"
                    )
                else:
                    # Likely invalid - backwards time without crossing midnight
                    logger.debug(
                        f"Invalid time range (backwards, not cross-midnight): "
                        f"{start_decimal:.2f} to {end_decimal:.2f}. "
                        f"Match: '{match.group(0)}'. Skipping."
                    )
                    continue
            
            time_ranges.append((start_decimal, end_decimal))
        
        if time_ranges:
            result["num_pickup_windows"] = len(time_ranges)
            
            # Calculate total hours (sum of all windows)
            # Note: For cross-midnight ranges, end_time will be > 24
            total_hours = sum(end - start for start, end in time_ranges)
            result["total_pickup_hours"] = round(total_hours, 2)
            
            # First pickup start and last pickup end
            # Normalize times back to 0-24 range for these features
            all_starts = [start for start, _ in time_ranges]
            all_ends = [end % 24 for _, end in time_ranges]  # Modulo to get time of day
            result["first_pickup_start_hour"] = round(min(all_starts), 2)
            result["last_pickup_end_hour"] = round(max(all_ends), 2)
        
    except Exception as e:
        logger.warning(f"Error parsing pickup window info: {e}")
        # Return defaults on error
    
    return result


def engineer_auction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from auction-level data.

    Features include:
    - Auction type (estate sale, moving sale, reseller, etc.)
    - Location (city, state/province, region)
    - Datetime features: day of week, time of day, season, holidays
    - Total number of items in auction
    - Auction duration
    - Pickup window features (day, hours, number of windows)

    Args:
        df: Raw auction DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering auction features...")
    features = df.copy()

    # Extract pickup window features if auction_removal_info column exists
    if "auction_removal_info" in features.columns:
        logger.info("Extracting pickup window features...")
        pickup_features = features["auction_removal_info"].apply(extract_pickup_windows)
        
        # Convert list of dicts to DataFrame and join
        pickup_df = pd.DataFrame(pickup_features.tolist())
        features = pd.concat([features, pickup_df], axis=1)
        
        logger.info(
            f"Added pickup window features: {list(pickup_df.columns)}"
        )

    # TODO: Implement additional auction-level feature engineering
    # Example features:
    # Initialize datetime feature engineer
    dt_engineer = DatetimeFeatureEngineer(
        include_cyclical=True,
        include_holidays=True,
        region="both",
    )

    # Add datetime features for auction start time
    if "start_time" in features.columns:
        features = dt_engineer.add_datetime_features(
            features, "start_time", prefix="auction_start_"
        )

    # Add datetime features for auction end time
    if "end_time" in features.columns:
        features = dt_engineer.add_datetime_features(
            features, "end_time", prefix="auction_end_"
        )

    # Add auction duration features
    if "start_time" in features.columns and "end_time" in features.columns:
        features = add_auction_duration_features(
            features, start_col="start_time", end_col="end_time"
        )

    # Additional features can be added here:
    # - features['auction_item_count'] = ...
    # - features['location_encoded'] = ...

    logger.info(f"Auction features shape: {features.shape}")
    return features


def engineer_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from item-level data.

    Features to consider:
    - Category (hierarchical)
    - Starting bid / reserve price
    - Number of images
    - Description length
    - Keywords in title/description
    - Item position in auction (lot number)
    - Estimated value (if available)

    Args:
        df: Raw item DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering item features...")
    features = df.copy()

    # TODO: Implement item-level feature engineering
    # Example features:
    # - features['description_length'] = features['description'].str.len()
    # - features['num_images'] = ...
    # - features['has_reserve'] = ...

    return features


def engineer_bid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from bid history.

    Features include:
    - Number of bids
    - Number of unique bidders
    - Bid velocity (bids per hour)
    - Max bid increment
    - Average bid increment
    - Time since last bid
    - Soft-close extensions count
    - Early vs late bidding ratio
    - Bid timing features (time of day, day of week, etc.)

    Args:
        df: Raw bid DataFrame (for a single item or aggregated)

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering bid features...")
    features = df.copy()

    # Initialize datetime feature engineer
    dt_engineer = DatetimeFeatureEngineer(
        include_cyclical=True,
        include_holidays=True,
        region="both",
    )

    # Add datetime features for bid time
    if "bid_time" in features.columns:
        features = dt_engineer.add_datetime_features(
            features, "bid_time", prefix="bid_"
        )

    # Add bid timing features relative to auction lifecycle
    required_cols = {"bid_time", "auction_start_time", "auction_end_time"}
    if required_cols.issubset(features.columns):
        features = add_bid_timing_features(
            features,
            bid_time_col="bid_time",
            auction_start_col="auction_start_time",
            auction_end_col="auction_end_time",
        )

    # Additional features can be added here:
    # - features['num_bids'] = ...
    # - features['num_unique_bidders'] = ...
    # - features['bid_velocity'] = ...

    logger.info(f"Bid features shape: {features.shape}")
    return features


# =============================================================================
# Text Features
# =============================================================================


def preprocess_text(text: str) -> str:
    """
    Preprocess text for NLP models.

    Steps:
    - Lowercase
    - Remove special characters
    - Normalize whitespace
    - (Optional) Remove stopwords
    - (Optional) Lemmatization

    Args:
        text: Raw text string

    Returns:
        Preprocessed text
    """
    if not text or pd.isna(text):
        return ""

    # Basic preprocessing
    text = str(text).lower().strip()

    # TODO: Add more sophisticated preprocessing
    # - Remove HTML tags
    # - Normalize unicode
    # - Handle auction-specific abbreviations

    return text


def extract_text_features(
    texts: list[str],
    method: str = "tfidf",
) -> Any:
    """
    Extract features from text data.

    Methods:
    - 'tfidf': TF-IDF vectorization
    - 'embeddings': Pretrained embeddings (BERT, etc.)
    - 'bow': Bag of words

    Args:
        texts: List of text strings
        method: Feature extraction method

    Returns:
        Feature matrix or embeddings
    """
    logger.info(f"Extracting text features using {method}...")

    # TODO: Implement text feature extraction
    # For 'embeddings', use transformers library
    # For 'tfidf', use sklearn TfidfVectorizer

    raise NotImplementedError(f"Text feature extraction ({method}) not yet implemented")


# =============================================================================
# Image Features
# =============================================================================


def extract_image_features(
    image_paths: list[str],
    model_name: str = "resnet50",
) -> Any:
    """
    Extract features from item images using pretrained CNN.

    Args:
        image_paths: List of paths to images
        model_name: Pretrained model to use for feature extraction

    Returns:
        Feature matrix (N x D) where D is embedding dimension
    """
    logger.info(f"Extracting image features using {model_name}...")

    # TODO: Implement image feature extraction
    # - Load pretrained model (ResNet, EfficientNet, etc.)
    # - Remove classification head
    # - Extract embeddings from penultimate layer

    raise NotImplementedError("Image feature extraction not yet implemented")


# =============================================================================
# Sequential Features
# =============================================================================


def prepare_bid_sequences(
    bids_df: pd.DataFrame,
    max_seq_length: int = 100,
    padding: str = "post",
) -> Any:
    """
    Prepare bid sequences for sequential models (LSTM, Transformer).

    Features per timestep:
    - Bid amount
    - Time since auction start
    - Time since last bid
    - Bid increment
    - Bidder ID (encoded)

    Args:
        bids_df: DataFrame with bid history
        max_seq_length: Maximum sequence length (pad/truncate)
        padding: 'pre' or 'post' padding

    Returns:
        Padded sequences array (N x T x F)
    """
    logger.info("Preparing bid sequences...")

    # TODO: Implement bid sequence preparation
    # - Group bids by item
    # - Create feature vectors per bid
    # - Pad/truncate to max_seq_length

    raise NotImplementedError("Bid sequence preparation not yet implemented")


# =============================================================================
# Feature Pipeline
# =============================================================================


def build_feature_pipeline(
    auctions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    bids_df: pd.DataFrame,
    include_text: bool = True,
    include_images: bool = True,
    include_sequences: bool = True,
) -> dict[str, Any]:
    """
    Build complete feature set for all models.

    Args:
        auctions_df: Auction-level data
        items_df: Item-level data
        bids_df: Bid history data
        include_text: Whether to extract text features
        include_images: Whether to extract image features
        include_sequences: Whether to prepare bid sequences

    Returns:
        Dictionary with feature sets for each model type
    """
    logger.info("Building feature pipeline...")

    features = {}

    # Tabular features (always included)
    auction_features = engineer_auction_features(auctions_df)
    item_features = engineer_item_features(items_df)
    bid_features = engineer_bid_features(bids_df)

    features["tabular"] = {
        "auctions": auction_features,
        "items": item_features,
        "bids": bid_features,
    }

    # Text features
    if include_text:
        # TODO: Extract text features from item descriptions
        features["text"] = None

    # Image features
    if include_images:
        # TODO: Extract image features from item photos
        features["images"] = None

    # Sequential features
    if include_sequences:
        # TODO: Prepare bid sequences
        features["sequences"] = None

    logger.info("Feature pipeline complete")
    return features


# =============================================================================
# Target Variable
# =============================================================================


def prepare_target(
    items_df: pd.DataFrame,
    target_col: str = "winning_price",
    handle_zero_bids: str = "include",
) -> pd.Series:
    """
    Prepare target variable (winning price).

    Args:
        items_df: Item DataFrame with winning prices
        target_col: Name of target column
        handle_zero_bids: How to handle items with no bids
            - 'include': Keep as 0
            - 'exclude': Remove from dataset
            - 'separate': Flag for separate model

    Returns:
        Target series
    """
    target = items_df[target_col].copy()

    if handle_zero_bids == "exclude":
        target = target[target > 0]
        logger.info(f"Excluded {(items_df[target_col] == 0).sum()} zero-bid items")
    elif handle_zero_bids == "include":
        logger.info(f"Including {(target == 0).sum()} zero-bid items")

    return target
