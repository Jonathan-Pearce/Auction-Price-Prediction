# =============================================================================
# Auction Price Prediction - Feature Engineering
# =============================================================================
"""
Feature engineering pipeline for auction price prediction.

Transforms raw auction, item, and bid data into features for ML models:
- Tabular features: Numerical and categorical from auction/item metadata
- Text features: Processed item descriptions and titles
- Image features: Extracted from item photos (via pretrained models)
- Sequential features: Bid history time series
- Enriched features: JSON columns from enriched item data
"""

import json
import re
from collections import Counter
from typing import Any

import pandas as pd
from loguru import logger

# =============================================================================
# Tabular Features
# =============================================================================


def engineer_auction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from auction-level data.

    Features to consider:
    - Auction type (estate sale, moving sale, reseller, etc.)
    - Location (city, state/province, region)
    - Day of week / time of day auction ends
    - Season / month
    - Total number of items in auction
    - Auction duration

    Args:
        df: Raw auction DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering auction features...")
    features = df.copy()

    # TODO: Implement auction-level feature engineering
    # Example features:
    # - features['auction_item_count'] = ...
    # - features['auction_day_of_week'] = ...
    # - features['auction_is_weekend'] = ...

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

    Features to consider:
    - Number of bids
    - Number of unique bidders
    - Bid velocity (bids per hour)
    - Max bid increment
    - Average bid increment
    - Time since last bid
    - Soft-close extensions count
    - Early vs late bidding ratio

    Args:
        df: Raw bid DataFrame (for a single item or aggregated)

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering bid features...")
    features = df.copy()

    # TODO: Implement bid-level feature engineering
    # Example features:
    # - features['num_bids'] = ...
    # - features['num_unique_bidders'] = ...
    # - features['bid_velocity'] = ...

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


# =============================================================================
# Enriched Item JSON Column Features
# =============================================================================


def _safe_parse_json(x: Any) -> list | dict:
    """
    Safely parse JSON data from various input formats.

    Args:
        x: Input data (string, list, dict, or None)

    Returns:
        Parsed JSON as list or dict, or empty list if parsing fails
    """
    if x is None or (isinstance(x, str) and x.strip() in ("", "null", "[]", "{}")):
        return []
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return []


# =============================================================================
# Brands Features
# =============================================================================

# Curated list of luxury/premium brands commonly found in auctions
LUXURY_BRANDS = {
    "royal albert", "waterford", "wedgwood", "limoges", "royal doulton",
    "lenox", "spode", "coalport", "minton", "meissen", "herend", "lalique",
    "baccarat", "steuben", "tiffany", "cartier", "rolex", "omega", "breitling",
    "baker furniture", "henredon", "drexel", "ethan allen", "thomasville",
    "herman miller", "knoll", "eames", "le creuset", "kitchenaid",
    "swarovski", "hummel", "lladro", "royal copenhagen", "bing & grondahl",
}


def engineer_enriched_brands_features(
    df: pd.DataFrame,
    brands_col: str = "enriched_item_brands",
    top_n_brands: int = 50,
) -> pd.DataFrame:
    """
    Engineer features from the enriched_item_brands JSON column.

    Features created:
    - has_brand: Whether item has any brand identified
    - num_brands: Count of brands per item
    - primary_brand: First/main brand (categorical)
    - is_luxury_brand: Whether any brand is in luxury list

    Args:
        df: DataFrame with enriched_item_brands column
        brands_col: Name of the brands column
        top_n_brands: Number of top brands for one-hot encoding

    Returns:
        DataFrame with brand features added
    """
    logger.info("Engineering enriched brands features...")
    features = df.copy()

    if brands_col not in features.columns:
        logger.warning(f"Column {brands_col} not found in DataFrame")
        return features

    # Parse JSON column
    brands_parsed = features[brands_col].apply(_safe_parse_json)

    # Basic features
    features["brand_count"] = brands_parsed.apply(len)
    features["has_brand"] = features["brand_count"] > 0

    # Extract primary brand (first in list)
    def get_primary_brand(brands_list: list) -> str | None:
        if not brands_list:
            return None
        first = brands_list[0]
        if isinstance(first, dict):
            return first.get("name", first.get("brand", str(first)))
        return str(first) if first else None

    features["primary_brand"] = brands_parsed.apply(get_primary_brand)
    
    # Handle empty DataFrame case
    if len(features) == 0:
        features["primary_brand_lower"] = pd.Series(dtype="object")
    else:
        features["primary_brand_lower"] = features["primary_brand"].astype(str).str.lower()
        features.loc[features["primary_brand"].isna(), "primary_brand_lower"] = None

    # Check for luxury brands
    def has_luxury_brand(brands_list: list) -> bool:
        for brand in brands_list:
            name = brand if isinstance(brand, str) else str(brand.get("name", ""))
            if name.lower() in LUXURY_BRANDS:
                return True
        return False

    features["is_luxury_brand"] = brands_parsed.apply(has_luxury_brand)

    # Find top brands for potential one-hot encoding
    brand_counts = Counter()
    for brands_list in brands_parsed:
        for brand in brands_list:
            name = brand if isinstance(brand, str) else str(brand.get("name", ""))
            if name:
                brand_counts[name.lower()] += 1

    top_brands = [brand for brand, _ in brand_counts.most_common(top_n_brands)]

    # Create binary flags for top brands
    for brand in top_brands[:20]:  # Only top 20 as individual features
        safe_name = re.sub(r"[^a-z0-9]", "_", brand)
        features[f"brand_is_{safe_name}"] = features["primary_brand_lower"] == brand

    # Clean up temporary column
    features = features.drop(columns=["primary_brand_lower"])

    logger.info(
        f"Created {features['has_brand'].sum()} items with brand info "
        f"({features['is_luxury_brand'].sum()} luxury brands)"
    )

    return features


# =============================================================================
# Categories Features
# =============================================================================

# Super-category mappings for grouping similar categories
SUPER_CATEGORY_MAPPINGS = {
    "furniture": ["furniture", "chairs", "tables", "seating", "sofa", "couch", "desk",
                  "cabinet", "dresser", "bookcase", "shelf", "bench", "bed"],
    "art": ["art", "painting", "paintings", "prints", "print", "sculpture", "artwork",
            "framed", "canvas", "watercolor", "oil painting", "lithograph"],
    "collectibles": ["collectibles", "collectible", "antiques", "antique", "vintage",
                     "memorabilia", "sports cards", "coins", "stamps"],
    "electronics": ["electronics", "computer", "tv", "audio", "stereo", "camera",
                    "phone", "tablet", "gaming", "appliances"],
    "jewelry": ["jewelry", "jewellery", "watches", "watch", "necklace", "ring",
                "bracelet", "earrings", "pendant", "costume jewelry"],
    "china_glass": ["china", "glassware", "ceramics", "porcelain", "dinnerware",
                    "crystal", "pottery", "figurines", "figurine", "vase"],
    "kitchenware": ["kitchenware", "kitchen", "cookware", "serveware", "tableware",
                    "bakeware", "utensils", "appliances"],
    "books_media": ["books", "book", "records", "vinyl", "cds", "dvds", "media",
                    "magazines", "comics"],
    "lighting": ["lighting", "lamp", "lamps", "chandelier", "light fixture", "sconce"],
    "decor": ["decor", "home decor", "decorative", "mirror", "clock", "rug", "rugs",
              "textile", "curtains", "pillows"],
    "tools": ["tools", "tool", "hardware", "power tools", "hand tools", "garage"],
    "toys": ["toys", "toy", "games", "dolls", "action figures", "model", "hobby"],
    "clothing": ["clothing", "clothes", "fashion", "accessories", "shoes", "handbags",
                 "purses", "vintage clothing"],
}


def engineer_enriched_categories_features(
    df: pd.DataFrame,
    categories_col: str = "enriched_item_categories",
) -> pd.DataFrame:
    """
    Engineer features from the enriched_item_categories JSON column.

    Features created:
    - category_depth: Number of categories (depth of classification)
    - primary_category: First/top-level category
    - secondary_category: Second-level category if present
    - is_furniture, is_art, is_collectibles, etc.: Super-category flags

    Args:
        df: DataFrame with enriched_item_categories column
        categories_col: Name of the categories column

    Returns:
        DataFrame with category features added
    """
    logger.info("Engineering enriched categories features...")
    features = df.copy()

    if categories_col not in features.columns:
        logger.warning(f"Column {categories_col} not found in DataFrame")
        return features

    # Handle empty DataFrame
    if len(features) == 0:
        features["category_depth"] = pd.Series(dtype="int64")
        features["primary_category"] = pd.Series(dtype="object")
        features["secondary_category"] = pd.Series(dtype="object")
        features["category_path"] = pd.Series(dtype="object")
        for super_cat in SUPER_CATEGORY_MAPPINGS.keys():
            features[f"is_{super_cat}"] = pd.Series(dtype="bool")
        return features

    # Parse JSON column
    categories_parsed = features[categories_col].apply(_safe_parse_json)

    # Category depth
    features["category_depth"] = categories_parsed.apply(len)

    # Extract primary and secondary categories
    def get_category_at_depth(cats_list: list, depth: int) -> str | None:
        if not cats_list or len(cats_list) <= depth:
            return None
        cat = cats_list[depth]
        if isinstance(cat, dict):
            return cat.get("name", cat.get("category", str(cat)))
        return str(cat).lower() if cat else None

    features["primary_category"] = categories_parsed.apply(
        lambda x: get_category_at_depth(x, 0)
    )
    features["secondary_category"] = categories_parsed.apply(
        lambda x: get_category_at_depth(x, 1)
    )

    # Normalize primary category
    features["primary_category_lower"] = features["primary_category"].str.lower()

    # Create super-category flags
    def check_super_category(cats_list: list, keywords: list) -> bool:
        for cat in cats_list:
            cat_str = cat if isinstance(cat, str) else str(cat.get("name", cat))
            cat_lower = cat_str.lower()
            for keyword in keywords:
                if keyword in cat_lower:
                    return True
        return False

    for super_cat, keywords in SUPER_CATEGORY_MAPPINGS.items():
        features[f"is_{super_cat}"] = categories_parsed.apply(
            lambda x, kw=keywords: check_super_category(x, kw)
        )

    # Category path as string (for text embeddings)
    features["category_path"] = categories_parsed.apply(
        lambda x: " > ".join(str(c) if isinstance(c, str) else str(c.get("name", c))
                             for c in x) if x else ""
    )

    # Clean up
    features = features.drop(columns=["primary_category_lower"])

    logger.info(f"Created category features with {features['category_depth'].mean():.2f} avg depth")

    return features


# =============================================================================
# Items Features
# =============================================================================


def engineer_enriched_items_features(
    df: pd.DataFrame,
    items_col: str = "enriched_item_items",
) -> pd.DataFrame:
    """
    Engineer features from the enriched_item_items JSON column.

    Features created:
    - num_items_in_lot: Count of individual items in the lot
    - is_single_item: Flag for single-item lots
    - is_multi_item: Flag for lots with 3+ items
    - item_title_length_avg: Average length of item titles
    - item_categories_unique: Number of unique item categories
    - item_titles_combined: Combined text of all item titles

    Args:
        df: DataFrame with enriched_item_items column
        items_col: Name of the items column

    Returns:
        DataFrame with items features added
    """
    logger.info("Engineering enriched items features...")
    features = df.copy()

    if items_col not in features.columns:
        logger.warning(f"Column {items_col} not found in DataFrame")
        return features

    # Parse JSON column
    items_parsed = features[items_col].apply(_safe_parse_json)

    # Count items in lot
    features["num_items_in_lot"] = items_parsed.apply(len)
    features["is_single_item"] = features["num_items_in_lot"] == 1
    features["is_multi_item"] = features["num_items_in_lot"] >= 3

    # Title length statistics
    def get_title_lengths(items_list: list) -> list:
        lengths = []
        for item in items_list:
            if isinstance(item, dict):
                title = item.get("title", "")
                lengths.append(len(title))
        return lengths

    title_lengths = items_parsed.apply(get_title_lengths)
    features["item_title_length_avg"] = title_lengths.apply(
        lambda x: sum(x) / len(x) if x else 0
    )
    features["item_title_length_total"] = title_lengths.apply(sum)

    # Unique categories
    def count_unique_categories(items_list: list) -> int:
        categories = set()
        for item in items_list:
            if isinstance(item, dict):
                cat = item.get("category", "")
                if cat:
                    categories.add(cat.lower())
        return len(categories)

    features["item_categories_unique"] = items_parsed.apply(count_unique_categories)

    # Dominant category
    def get_dominant_category(items_list: list) -> str | None:
        cat_counts: Counter = Counter()
        for item in items_list:
            if isinstance(item, dict):
                cat = item.get("category", "")
                if cat:
                    cat_counts[cat.lower()] += 1
        if cat_counts:
            return cat_counts.most_common(1)[0][0]
        return None

    features["dominant_item_category"] = items_parsed.apply(get_dominant_category)

    # Combined titles for text features
    def combine_titles(items_list: list) -> str:
        titles = []
        for item in items_list:
            if isinstance(item, dict):
                title = item.get("title", "")
                if title:
                    titles.append(title)
        return " | ".join(titles)

    features["item_titles_combined"] = items_parsed.apply(combine_titles)

    logger.info(
        f"Created items features: {features['is_single_item'].sum()} single-item lots, "
        f"{features['is_multi_item'].sum()} multi-item lots"
    )

    return features


# =============================================================================
# Attributes Features
# =============================================================================

# Common materials for classification
MATERIAL_CATEGORIES = {
    "wood": ["wood", "wooden", "oak", "pine", "mahogany", "walnut", "cherry",
             "teak", "maple", "birch", "cedar", "bamboo"],
    "metal": ["metal", "iron", "steel", "brass", "bronze", "copper", "aluminum",
              "chrome", "silver", "gold", "pewter", "tin"],
    "glass": ["glass", "crystal", "mirror"],
    "ceramic": ["ceramic", "porcelain", "china", "pottery", "stoneware",
                "earthenware", "terracotta"],
    "fabric": ["fabric", "cloth", "cotton", "wool", "silk", "linen", "velvet",
               "leather", "suede", "upholstery", "textile"],
    "plastic": ["plastic", "acrylic", "resin", "vinyl", "polyester", "nylon"],
    "paper": ["paper", "cardboard", "canvas"],
    "stone": ["stone", "marble", "granite", "slate", "concrete"],
}

# Year extraction patterns
YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
DECADE_PATTERN = re.compile(r"\b(1[0-9]{2}0|20[0-2]0)s\b", re.IGNORECASE)


def engineer_enriched_attributes_features(
    df: pd.DataFrame,
    attributes_col: str = "enriched_item_attributes",
) -> pd.DataFrame:
    """
    Engineer features from the enriched_item_attributes JSON column.

    Features created:
    - num_attributes: Count of attributes
    - has_dimensions: Whether dimensions are specified
    - has_material: Whether material is specified
    - primary_material: Main material category
    - has_year_info: Whether year/era is specified
    - year_numeric: Extracted year if available
    - decade: Normalized decade (e.g., 1970)
    - dimension_largest: Largest dimension in inches

    Args:
        df: DataFrame with enriched_item_attributes column
        attributes_col: Name of the attributes column

    Returns:
        DataFrame with attributes features added
    """
    logger.info("Engineering enriched attributes features...")
    features = df.copy()

    if attributes_col not in features.columns:
        logger.warning(f"Column {attributes_col} not found in DataFrame")
        return features

    # Parse JSON column
    attrs_parsed = features[attributes_col].apply(_safe_parse_json)

    # Basic counts
    features["num_attributes"] = attrs_parsed.apply(len)
    features["has_attributes"] = features["num_attributes"] > 0

    # Build attribute dictionaries per item
    def attrs_to_dict(attrs_list: list) -> dict:
        result = {}
        for attr in attrs_list:
            if isinstance(attr, dict):
                name = attr.get("name", "").lower()
                value = attr.get("value", "")
                if name:
                    result[name] = value
        return result

    attrs_dicts = attrs_parsed.apply(attrs_to_dict)

    # Dimension features
    def has_dimension_attr(attrs_dict: dict) -> bool:
        dim_keys = ["dimensions", "size", "height", "width", "depth",
                    "height_inches", "width_inches"]
        return any(k in attrs_dict for k in dim_keys)

    features["has_dimensions"] = attrs_dicts.apply(has_dimension_attr)

    # Parse dimensions to extract largest dimension
    def extract_largest_dimension(attrs_dict: dict) -> float | None:
        dim_value = attrs_dict.get("dimensions", attrs_dict.get("size", ""))
        if not dim_value:
            return None
        # Find all numbers (dimensions)
        numbers = re.findall(r"(\d+(?:\.\d+)?)", str(dim_value))
        if numbers:
            return max(float(n) for n in numbers)
        return None

    features["dimension_largest"] = attrs_dicts.apply(extract_largest_dimension)

    # Material features
    def get_material_value(attrs_dict: dict) -> str | None:
        return attrs_dict.get("material", attrs_dict.get("materials", None))

    materials = attrs_dicts.apply(get_material_value)
    features["has_material"] = materials.notna() & (materials != "")

    def classify_material(material_str: str | None) -> str | None:
        if not material_str:
            return None
        material_lower = str(material_str).lower()
        for category, keywords in MATERIAL_CATEGORIES.items():
            for keyword in keywords:
                if keyword in material_lower:
                    return category
        return "other"

    features["primary_material"] = materials.apply(classify_material)

    # Create material binary flags
    for mat_cat in MATERIAL_CATEGORIES.keys():
        features[f"is_material_{mat_cat}"] = features["primary_material"] == mat_cat

    # Year/Era features
    def extract_year_info(attrs_dict: dict) -> tuple:
        """Extract year information from attributes."""
        year_keys = ["year", "era", "age", "period", "year_manufactured",
                     "year_made", "date", "publication_date"]

        for key in year_keys:
            if key in attrs_dict:
                value = str(attrs_dict[key])
                # Try to find specific year
                years = YEAR_PATTERN.findall(value)
                if years:
                    return int(years[0]), (int(years[0]) // 10) * 10
                # Try to find decade
                decades = DECADE_PATTERN.findall(value)
                if decades:
                    decade = int(decades[0])
                    return None, decade
        return None, None

    year_info = attrs_dicts.apply(extract_year_info)
    features["year_numeric"] = year_info.apply(lambda x: x[0])
    features["decade"] = year_info.apply(lambda x: x[1])
    features["has_year_info"] = features["decade"].notna()

    # Era categories
    def get_era_category(decade: float | None) -> str | None:
        if decade is None or pd.isna(decade):
            return None
        decade = int(decade)
        if decade < 1900:
            return "pre_1900"
        elif decade < 1950:
            return "1900_1950"
        elif decade < 1980:
            return "1950_1980"
        elif decade < 2000:
            return "1980_2000"
        else:
            return "2000_plus"

    features["era_category"] = features["decade"].apply(get_era_category)

    # Color features
    def get_color(attrs_dict: dict) -> str | None:
        return attrs_dict.get("color", attrs_dict.get("colour", None))

    features["has_color"] = attrs_dicts.apply(get_color).notna()

    # Style features
    features["has_style"] = attrs_dicts.apply(lambda x: "style" in x)

    # Condition features
    def has_condition_info(attrs_dict: dict) -> bool:
        cond_keys = ["condition", "condition_notes", "condition_details",
                     "condition_detail"]
        return any(k in attrs_dict for k in cond_keys)

    features["has_condition_info"] = attrs_dicts.apply(has_condition_info)

    # Combine all attributes as text for embeddings
    def attrs_to_text(attrs_list: list) -> str:
        parts = []
        for attr in attrs_list:
            if isinstance(attr, dict):
                name = attr.get("name", "")
                value = attr.get("value", "")
                if name and value:
                    parts.append(f"{name}: {value}")
        return " | ".join(parts)

    features["attributes_text"] = attrs_parsed.apply(attrs_to_text)

    logger.info(
        f"Created attributes features: "
        f"{features['has_material'].sum()} with material, "
        f"{features['has_year_info'].sum()} with year info"
    )

    return features


# =============================================================================
# Photos Features
# =============================================================================


def engineer_enriched_photos_features(
    df: pd.DataFrame,
    photos_col: str = "enriched_item_photosTaken",
) -> pd.DataFrame:
    """
    Engineer features from the enriched_item_photosTaken JSON column.

    Features created:
    - num_photos: Count of photos per item
    - avg_description_length: Average photo description length
    - total_description_length: Total description characters
    - has_condition_photo: Photo mentions condition
    - has_detail_photo: Photo mentions detail/closeup
    - has_brand_photo: Photo mentions brand/maker mark
    - photo_descriptions_combined: All descriptions as text
    - primary_photo_path: Path to first/main photo

    Args:
        df: DataFrame with enriched_item_photosTaken column
        photos_col: Name of the photos column

    Returns:
        DataFrame with photos features added
    """
    logger.info("Engineering enriched photos features...")
    features = df.copy()

    if photos_col not in features.columns:
        logger.warning(f"Column {photos_col} not found in DataFrame")
        return features

    # Parse JSON column
    photos_parsed = features[photos_col].apply(_safe_parse_json)

    # Basic counts
    features["num_photos"] = photos_parsed.apply(len)
    features["has_photos"] = features["num_photos"] > 0

    # Description length statistics
    def get_description_lengths(photos_list: list) -> list:
        lengths = []
        for photo in photos_list:
            if isinstance(photo, dict):
                desc = photo.get("description", "")
                lengths.append(len(desc))
        return lengths

    desc_lengths = photos_parsed.apply(get_description_lengths)
    features["photo_desc_length_avg"] = desc_lengths.apply(
        lambda x: sum(x) / len(x) if x else 0
    )
    features["photo_desc_length_total"] = desc_lengths.apply(sum)

    # Photo content flags based on "reason" field
    def check_photo_reason(photos_list: list, keywords: list) -> bool:
        for photo in photos_list:
            if isinstance(photo, dict):
                reason = photo.get("reason", "").lower()
                for keyword in keywords:
                    if keyword in reason:
                        return True
        return False

    features["has_condition_photo"] = photos_parsed.apply(
        lambda x: check_photo_reason(x, ["condition", "wear", "damage", "flaw"])
    )
    features["has_detail_photo"] = photos_parsed.apply(
        lambda x: check_photo_reason(x, ["detail", "close", "closeup", "close-up"])
    )
    features["has_overall_photo"] = photos_parsed.apply(
        lambda x: check_photo_reason(x, ["overall", "full", "complete", "front"])
    )
    features["has_brand_photo"] = photos_parsed.apply(
        lambda x: check_photo_reason(x, ["brand", "maker", "mark", "signature", "label"])
    )

    # Photo coverage score (weighted combination)
    features["photo_coverage_score"] = (
        features["num_photos"] * 0.3 +
        features["has_condition_photo"].astype(int) * 0.2 +
        features["has_detail_photo"].astype(int) * 0.2 +
        features["has_overall_photo"].astype(int) * 0.2 +
        features["has_brand_photo"].astype(int) * 0.1
    )

    # Primary photo path
    def get_primary_photo_path(photos_list: list) -> str | None:
        if not photos_list:
            return None
        first = photos_list[0]
        if isinstance(first, dict):
            return first.get("imagePath", first.get("image_path", None))
        return None

    features["primary_photo_path"] = photos_parsed.apply(get_primary_photo_path)

    # Combined descriptions for text features
    def combine_descriptions(photos_list: list) -> str:
        descriptions = []
        for photo in photos_list:
            if isinstance(photo, dict):
                desc = photo.get("description", "")
                if desc:
                    descriptions.append(desc)
        return " | ".join(descriptions)

    features["photo_descriptions_combined"] = photos_parsed.apply(combine_descriptions)

    logger.info(
        f"Created photos features: {features['num_photos'].mean():.2f} avg photos/item"
    )

    return features


# =============================================================================
# Combined Enriched Features Pipeline
# =============================================================================


def engineer_all_enriched_features(
    df: pd.DataFrame,
    include_brands: bool = True,
    include_categories: bool = True,
    include_items: bool = True,
    include_attributes: bool = True,
    include_photos: bool = True,
) -> pd.DataFrame:
    """
    Apply all enriched item feature engineering functions.

    This is the main entry point for processing the 5 JSON columns
    from the Hugging Face enriched_item_data dataset.

    Args:
        df: DataFrame with enriched item columns
        include_brands: Whether to process enriched_item_brands
        include_categories: Whether to process enriched_item_categories
        include_items: Whether to process enriched_item_items
        include_attributes: Whether to process enriched_item_attributes
        include_photos: Whether to process enriched_item_photosTaken

    Returns:
        DataFrame with all enriched features added

    Example:
        >>> df = load_enriched_data()
        >>> df = engineer_all_enriched_features(df)
    """
    logger.info("Engineering all enriched item features...")
    features = df.copy()

    if include_brands:
        features = engineer_enriched_brands_features(features)

    if include_categories:
        features = engineer_enriched_categories_features(features)

    if include_items:
        features = engineer_enriched_items_features(features)

    if include_attributes:
        features = engineer_enriched_attributes_features(features)

    if include_photos:
        features = engineer_enriched_photos_features(features)

    logger.info("All enriched item features complete")

    return features
