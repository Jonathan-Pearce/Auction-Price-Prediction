# Enriched Item JSON Columns - EDA & Feature Engineering

> **Exploratory Data Analysis and Feature Engineering Recommendations for JSON columns in the Hugging Face enriched_item_data dataset**

## Overview

The enriched item dataset from Hugging Face (`jpearce610/enriched_item_data`) contains approximately 3 million items from MaxSold auctions. This document presents the findings from exploratory data analysis (EDA) on 5 JSON columns and provides feature engineering recommendations for each.

**Dataset**: https://huggingface.co/datasets/jpearce610/enriched_item_data

**JSON Columns Analyzed**:
1. `enriched_item_brands` - Brand identification data
2. `enriched_item_categories` - Hierarchical category classification
3. `enriched_item_items` - Individual items within auction lots
4. `enriched_item_attributes` - Item attributes (dimensions, material, etc.)
5. `enriched_item_photosTaken` - Photo metadata and descriptions

---

## 1. enriched_item_brands

### EDA Summary

| Metric | Value |
|--------|-------|
| Items with brands | 20.8% |
| Items without brands | 79.2% |
| Mean brands per item (when present) | 1.69 |
| Median brands per item | 1.0 |
| Max brands per item | 25 |

**Brand Count Distribution** (when present):
- 1 brand: 66.2% of items with brands
- 2 brands: 17.1%
- 3+ brands: 16.7%

**JSON Structure**:
```json
["Random House"]
["Hammersley", "Myotts"]
["Royal Albert"]
```

**Key Observations**:
- ~80% of items have no brand information
- When present, most items have 1-2 brands
- Brands are stored as a simple array of strings
- Brand names vary in formatting (some uppercase, some mixed case)

### Feature Engineering Recommendations

1. **`has_brand` (Binary)**
   - Simple flag indicating whether any brand is identified
   - High signal for collectibles and antiques

2. **`num_brands` (Numeric)**
   - Count of brands per item
   - May indicate product diversity in a lot

3. **`primary_brand` (Categorical)**
   - First brand in the list (highest priority)
   - Use one-hot encoding for top 50-100 brands
   - Group rare brands as "Other"

4. **`is_luxury_brand` (Binary)**
   - Create curated list of luxury/premium brands
   - Examples: "Royal Albert", "Waterford", "Limoges", "Wedgwood"

5. **`brand_category` (Categorical)**
   - Group brands into categories:
     - Furniture brands
     - China/porcelain brands
     - Electronics brands
     - Fashion brands
     - Art/antique brands

6. **`brand_embedding` (Vector)**
   - For deep learning: embed brand names using pretrained embeddings
   - Concatenate if multiple brands present

---

## 2. enriched_item_categories

### EDA Summary

| Metric | Value |
|--------|-------|
| Items with categories | 100.0% |
| Mean categories per item | 2.55 |
| Median categories | 2.0 |
| Max categories per item | 13 |
| Unique categories (all depths) | 6,689 |
| Unique Level 1 categories | 2,237 |
| Unique Level 2 categories | 3,818 |

**Category Distribution**:
- 1 category: 3.0%
- 2 categories: 60.7%
- 3 categories: 22.7%
- 4+ categories: 13.6%

**Top Level 1 Categories**:
| Category | % of Items |
|----------|------------|
| furniture | 14.2% |
| art | 4.9% |
| kitchenware | 3.0% |
| lighting | 2.5% |
| home decor | 2.3% |
| books | 2.3% |
| glassware | 2.0% |
| electronics | 1.9% |
| jewelry | 1.7% |
| collectibles | 1.6% |

**JSON Structure** (Hierarchical list of strings):
```json
["furniture", "sofa"]
["Mobility Aids", "Medical Equipment", "Stair Lifts"]
["china", "tea set", "dinnerware"]
```

**Key Observations**:
- All items have category information (100% coverage)
- Categories are hierarchical (general → specific)
- Significant inconsistency in casing (e.g., "furniture" vs "Furniture")
- Some paths represent different taxonomies (broad vs specific first)

### Feature Engineering Recommendations

1. **`category_depth` (Numeric)**
   - Number of categories (depth of classification)
   - Deeper categorization may indicate more specialized/valuable items

2. **`primary_category` (Categorical)**
   - First category in list (normalized to lowercase)
   - One-hot encode top 30-50 categories
   - Group rare categories as "Other"

3. **`secondary_category` (Categorical)**
   - Second category if present (often more specific)
   - One-hot encode top 50 categories

4. **`category_path` (String/Embedding)**
   - Full path as "furniture > sofa"
   - Useful for text embeddings

5. **Super-category Flags (Binary)**:
   - `is_furniture`: furniture, chairs, tables, seating
   - `is_art`: art, painting, prints, sculpture
   - `is_collectible`: collectibles, antiques, vintage
   - `is_electronics`: electronics, appliances
   - `is_jewelry`: jewelry, watches, accessories
   - `is_china_glass`: china, glassware, ceramics, dinnerware
   - `is_kitchenware`: kitchenware, serveware, tableware
   - `is_books_media`: books, records, media

6. **Category Hierarchy Features**:
   - Encode categories at each level separately
   - Create embeddings for category paths

---

## 3. enriched_item_items

### EDA Summary

| Metric | Value |
|--------|-------|
| Lots with item details | 100.0% |
| Mean items per lot | 3.45 |
| Median items per lot | 2.0 |
| Max items per lot | 47 |
| Total unique item categories | 5,720 |

**Items per Lot Distribution**:
- 1 item: 36.2%
- 2 items: 17.2%
- 3 items: 11.0%
- 4-5 items: 14.8%
- 6+ items: 20.8%

**Top Item Categories**:
| Category | % of Items |
|----------|------------|
| glassware | 2.5% |
| furniture | 2.1% |
| kitchenware | 2.1% |
| decor | 1.9% |
| books | 1.5% |
| dinnerware | 1.4% |
| figurines | 1.3% |
| lighting | 1.3% |

**JSON Structure**:
```json
[
  {"title": "Wooden piano bench...", "category": "bench"},
  {"title": "175-inch rail...", "category": "StairLiftRail"}
]
```

**Fields Present**:
- `title`: 100% - Item description
- `category`: 100% - Item-specific category

**Key Observations**:
- About 1/3 of lots contain a single item
- Multi-item lots are common (median = 2)
- Each item has a title and category
- Item-level categories are more granular than lot-level

### Feature Engineering Recommendations

1. **`num_items_in_lot` (Numeric)**
   - Count of individual items
   - Strongly related to lot value potential

2. **`is_single_item` (Binary)**
   - Flag for single-item lots
   - May have different pricing dynamics

3. **`is_multi_item` (Binary)**
   - Flag for lots with 3+ items

4. **`item_title_length_avg` (Numeric)**
   - Average character length of item titles
   - Proxy for description detail

5. **`item_title_length_total` (Numeric)**
   - Total characters in all item titles

6. **`item_categories_unique` (Numeric)**
   - Number of unique categories across items in lot
   - Indicates lot homogeneity/diversity

7. **`dominant_item_category` (Categorical)**
   - Most common category among items in lot

8. **`item_titles_combined` (Text)**
   - Concatenated item titles for text modeling
   - Useful for NLP/embedding approaches

9. **Lot Composition Features**:
   - `has_furniture_items`
   - `has_glassware_items`
   - `has_book_items`
   - etc.

---

## 4. enriched_item_attributes

### EDA Summary

| Metric | Value |
|--------|-------|
| Items with attributes | 83.6% |
| Items without attributes | 16.4% |
| Mean attributes per item (all) | 2.85 |
| Mean attributes per item (when present) | 3.41 |
| Median attributes | 3.0 |
| Max attributes | 15 |
| Unique attribute names | 11,649 |

**Top Attributes** (presence rate among items with attributes):
| Attribute | % Present |
|-----------|-----------|
| material | 47.6% |
| color | 23.4% |
| dimensions | 21.2% |
| style | 9.2% |
| origin | 4.8% |
| frame_material | 4.1% |
| condition_notes | 4.1% |
| pattern | 3.7% |
| height_inches | 3.7% |

**JSON Structure**:
```json
[
  {"name": "dimensions", "value": "36x16x20 inches"},
  {"name": "material", "value": "wood and fabric"},
  {"name": "color", "value": "black with multicolor floral embroidery"}
]
```

**Attribute Categories Identified**:
- **Dimension-related**: dimensions, size, height, width, chair_dimensions, etc.
- **Material-related**: material, frame_material, fabric, etc.
- **Color-related**: color, shade_color, base_color, etc.
- **Year/Date-related**: year, era, age, period, year_manufactured, etc.
- **Condition-related**: condition, condition_notes, condition_details, etc.

**Year Pattern Analysis**:
- ~5% of items have year-related attributes
- Formats include: "1970s", "1987", "October 5, 1950"
- Decades ("1970s") and specific years ("1987") both common

### Feature Engineering Recommendations

1. **`num_attributes` (Numeric)**
   - Count of attributes per item
   - More attributes may indicate better-documented items

2. **`has_dimensions` (Binary)**
   - Presence of dimension information

3. **`has_material` (Binary)**
   - Presence of material information

4. **Dimension Extraction**:
   - Parse dimension strings to extract:
     - `height_inches`: numeric height
     - `width_inches`: numeric width
     - `depth_inches`: numeric depth
     - `largest_dimension`: max(H, W, D)
     - `volume_estimate`: H × W × D (when all present)

5. **Material Features**:
   - `primary_material` (Categorical): wood, glass, metal, ceramic, fabric, etc.
   - Material type flags:
     - `is_wood`
     - `is_metal`
     - `is_glass`
     - `is_ceramic`
     - `is_fabric`
     - `is_plastic`

6. **Color Features**:
   - `primary_color` (Categorical): most common colors
   - `color_complexity`: number of colors mentioned
   - Color type flags:
     - `is_neutral_color`
     - `is_multicolor`

7. **Year/Era Features**:
   - `has_year_info` (Binary)
   - `year_numeric`: extracted year (when specific)
   - `decade`: normalized to nearest decade (e.g., 1970)
   - `era_category` (Categorical): 
     - Pre-1900
     - 1900-1950
     - 1950-1980
     - 1980-2000
     - 2000+

8. **Style Features**:
   - `style_category` (Categorical): modern, vintage, antique, art deco, etc.

9. **Origin Features**:
   - `country_of_origin` (Categorical)
   - `is_imported` (Binary)

10. **Condition Features**:
    - `has_condition_info` (Binary)
    - `condition_score` (Numeric): derived from condition text
    - `is_working` (Binary): if explicitly mentioned

11. **Attribute Embedding**:
    - Concatenate all attribute name-value pairs as text
    - Generate embeddings for ML models

---

## 5. enriched_item_photosTaken

### EDA Summary

| Metric | Value |
|--------|-------|
| Items with photos | 100.0% |
| Mean photos per item | 2.47 |
| Median photos | 2.0 |
| Max photos | 7 |

**Photo Count Distribution**:
- 1 photo: 6.6%
- 2 photos: 54.5%
- 3 photos: 31.3%
- 4+ photos: 7.6%

**Fields Present** (all 100%):
- `description`: Photo description (avg 103 chars)
- `reason`: Why photo was taken (avg 52 chars)
- `imageId`: Unique photo identifier
- `imagePath`: Path to image file

**Photo Reason Analysis**:
| Reason Theme | % of Photos |
|--------------|-------------|
| mentions condition | 38.4% |
| mentions detail | 32.0% |
| mentions overall | 13.2% |
| mentions design | 12.1% |
| brand/maker mark | 6.1% |
| closeup shot | 0.5% |

**JSON Structure**:
```json
[
  {
    "description": "Front angled view of wooden piano bench...",
    "reason": "To show overall condition and design",
    "imageId": "lstimg_01j8xb96cpfz49gnvjzvrj4g6d",
    "imagePath": "archive_images/409/31296/31296_9370418.jpg"
  }
]
```

**Key Observations**:
- Nearly all items have photos (99.99%+ coverage)
- Most items have 2-3 photos
- Photos come with rich metadata (descriptions and reasons)
- Descriptions average ~100 characters, providing item details
- Reasons indicate what the photo captures (condition, detail, marks)

### Feature Engineering Recommendations

1. **`num_photos` (Numeric)**
   - Count of photos per item
   - More photos may indicate higher-value items

2. **`avg_description_length` (Numeric)**
   - Average character length of photo descriptions
   - Proxy for documentation quality

3. **`total_description_length` (Numeric)**
   - Total characters across all descriptions

4. **Photo Content Flags** (Binary, from reasons):
   - `has_condition_photo`: mentions condition
   - `has_detail_photo`: mentions detail/closeup
   - `has_overall_photo`: mentions overall view
   - `has_brand_photo`: mentions brand/maker mark

5. **`photo_descriptions_combined` (Text)**
   - Concatenate all photo descriptions
   - Rich text for NLP/embedding models

6. **Photo Coverage Score** (Numeric):
   - Weighted combination of photo types
   - Higher if multiple angles/details shown

7. **`primary_photo_path`** (String):
   - First photo's image path for image model input

8. **Image-based Features** (requires image processing):
   - `image_brightness_avg`
   - `image_quality_score`
   - `background_type` (studio vs home setting)

---

## Implementation Priority

### High Priority (Strong predictive signal)
1. `num_items_in_lot` from items
2. `primary_category` from categories
3. `has_brand` + `primary_brand` from brands
4. `material` extraction from attributes
5. `num_photos` from photosTaken

### Medium Priority (Good signal, moderate complexity)
1. Category super-groupings
2. Dimension extraction and normalization
3. Year/era extraction
4. Photo content flags
5. Item diversity metrics

### Lower Priority (Complex or specialized)
1. Brand embeddings
2. Category path embeddings
3. Full attribute text embeddings
4. Image-based features
5. Color complexity analysis

---

## Usage Example

```python
from src.features import (
    engineer_enriched_brands_features,
    engineer_enriched_categories_features,
    engineer_enriched_items_features,
    engineer_enriched_attributes_features,
    engineer_enriched_photos_features,
)

# Load data
df = load_enriched_data()

# Apply feature engineering
df = engineer_enriched_brands_features(df)
df = engineer_enriched_categories_features(df)
df = engineer_enriched_items_features(df)
df = engineer_enriched_attributes_features(df)
df = engineer_enriched_photos_features(df)
```

---

## Related Documentation

- [ENRICHED_DATA.md](ENRICHED_DATA.md) - General enriched data documentation
- [ENRICHED_ITEM_SCRAPER.md](ENRICHED_ITEM_SCRAPER.md) - Data collection documentation
- [DATA.md](DATA.md) - Overall data documentation

---

*Last updated: January 2026*
