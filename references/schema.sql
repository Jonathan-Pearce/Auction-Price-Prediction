-- ============================================================================
-- Auction Price Prediction - DuckDB Schema
-- ============================================================================
-- This schema defines the data model for storing MaxSold auction data.
-- Designed for analytical queries and ML feature engineering.
-- ============================================================================

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Auctions table: One record per auction event
CREATE TABLE IF NOT EXISTS auctions (
    auction_id INTEGER PRIMARY KEY,
    title VARCHAR NOT NULL,
    description VARCHAR,
    location_city VARCHAR,
    location_province VARCHAR,
    location_country VARCHAR DEFAULT 'Canada',
    
    -- Timing
    start_time TIMESTAMP WITH TIME ZONE,
    scheduled_end_time TIMESTAMP WITH TIME ZONE,
    actual_end_time TIMESTAMP WITH TIME ZONE,  -- May differ due to soft-close
    
    -- Auction metadata
    auction_type VARCHAR,  -- 'estate_sale', 'downsizing', 'reseller', etc.
    total_items INTEGER,
    total_lots INTEGER,
    
    -- Aggregated results (populated after auction ends)
    total_bids INTEGER,
    total_revenue DECIMAL(12, 2),
    items_sold INTEGER,
    items_unsold INTEGER,
    
    -- Data collection metadata
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data_complete BOOLEAN DEFAULT FALSE,
    
    -- Indexes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Items table: One record per auction item/lot
CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY,
    auction_id INTEGER NOT NULL REFERENCES auctions(auction_id),
    lot_number INTEGER,
    
    -- Item details
    title VARCHAR NOT NULL,
    description VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    condition VARCHAR,
    dimensions VARCHAR,
    
    -- Pricing
    starting_bid DECIMAL(10, 2) DEFAULT 1.00,
    reserve_price DECIMAL(10, 2),  -- If applicable
    winning_bid DECIMAL(10, 2),    -- Final winning price
    
    -- Bid statistics
    bid_count INTEGER DEFAULT 0,
    unique_bidders INTEGER DEFAULT 0,
    
    -- Status
    has_bids BOOLEAN GENERATED ALWAYS AS (bid_count > 0) STORED,
    is_sold BOOLEAN GENERATED ALWAYS AS (winning_bid IS NOT NULL AND winning_bid > 0) STORED,
    
    -- Location for pickup
    pickup_location VARCHAR,
    
    -- Timing (for soft-close tracking)
    first_bid_time TIMESTAMP WITH TIME ZONE,
    last_bid_time TIMESTAMP WITH TIME ZONE,
    soft_close_extensions INTEGER DEFAULT 0,
    
    -- Data collection metadata
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    enriched_data_fetched BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Bids table: Full bidding history for each item
CREATE TABLE IF NOT EXISTS bids (
    bid_id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    auction_id INTEGER NOT NULL REFERENCES auctions(auction_id),
    
    -- Bid details
    amount DECIMAL(10, 2) NOT NULL,
    bidder_id VARCHAR NOT NULL,  -- Anonymized bidder identifier
    
    -- Timing
    bid_time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Derived fields for analysis
    bid_sequence INTEGER,  -- 1st bid, 2nd bid, etc. for this item
    time_to_close_seconds INTEGER,  -- Seconds until scheduled end when bid placed
    is_soft_close_trigger BOOLEAN,  -- Was this bid in final 2 minutes?
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Item images: Multiple images per item
CREATE TABLE IF NOT EXISTS item_images (
    image_id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    
    image_url VARCHAR NOT NULL,
    image_order INTEGER DEFAULT 0,  -- 0 = primary image
    
    -- Image metadata (populated during processing)
    width INTEGER,
    height INTEGER,
    file_size_bytes INTEGER,
    
    -- Processing status
    downloaded BOOLEAN DEFAULT FALSE,
    local_path VARCHAR,
    embedding_computed BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Enriched item data: Additional metadata from enriched endpoint
CREATE TABLE IF NOT EXISTS item_enriched (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    
    -- Category hierarchy
    category_level_1 VARCHAR,
    category_level_2 VARCHAR,
    category_level_3 VARCHAR,
    category_level_4 VARCHAR,
    
    -- Estimates
    estimated_value_low DECIMAL(10, 2),
    estimated_value_high DECIMAL(10, 2),
    
    -- Scores
    condition_score DECIMAL(4, 2),
    popularity_score DECIMAL(4, 2),
    
    -- Keywords (stored as JSON array)
    keywords JSON,
    
    -- Similar items data (stored as JSON)
    similar_items JSON,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- FEATURE TABLES (for ML pipelines)
-- ============================================================================

-- Tabular features: Pre-computed features for tabular model
CREATE TABLE IF NOT EXISTS features_tabular (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    
    -- Item features
    title_word_count INTEGER,
    description_word_count INTEGER,
    description_length INTEGER,
    has_dimensions BOOLEAN,
    image_count INTEGER,
    
    -- Category encoding (one-hot or label encoded)
    category_encoded INTEGER,
    subcategory_encoded INTEGER,
    condition_encoded INTEGER,
    
    -- Auction context features
    auction_total_items INTEGER,
    lot_number_normalized DECIMAL(5, 4),  -- lot_number / total_lots
    
    -- Time features
    auction_day_of_week INTEGER,
    auction_hour INTEGER,
    days_until_close DECIMAL(10, 4),
    
    -- Historical features (from similar items)
    category_avg_price DECIMAL(10, 2),
    category_median_price DECIMAL(10, 2),
    category_sell_rate DECIMAL(5, 4),
    
    -- Enriched features
    estimated_value_mid DECIMAL(10, 2),
    condition_score DECIMAL(4, 2),
    popularity_score DECIMAL(4, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Text embeddings: Pre-computed text embeddings
CREATE TABLE IF NOT EXISTS features_text (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    
    -- Combined text for embedding
    combined_text VARCHAR,
    
    -- Embedding vector (stored as array or blob)
    -- Using FLOAT[] for DuckDB
    title_embedding FLOAT[],
    description_embedding FLOAT[],
    combined_embedding FLOAT[],
    
    -- Embedding metadata
    embedding_model VARCHAR,
    embedding_dimension INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Image embeddings: Pre-computed image features
CREATE TABLE IF NOT EXISTS features_image (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    
    -- Primary image embedding
    image_embedding FLOAT[],
    
    -- Multi-image aggregation (if multiple images)
    aggregated_embedding FLOAT[],
    aggregation_method VARCHAR,  -- 'mean', 'max', 'attention'
    
    -- Image metadata
    embedding_model VARCHAR,
    embedding_dimension INTEGER,
    images_processed INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Bid sequence features: Pre-computed sequential features
CREATE TABLE IF NOT EXISTS features_sequential (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    
    -- Sequence statistics
    bid_count INTEGER,
    unique_bidders INTEGER,
    
    -- Timing features
    time_to_first_bid_seconds INTEGER,
    avg_time_between_bids_seconds DECIMAL(10, 2),
    bid_velocity_last_hour DECIMAL(10, 4),
    
    -- Amount features
    bid_amount_sequence FLOAT[],  -- Ordered list of bid amounts
    bid_increment_sequence FLOAT[],  -- Differences between consecutive bids
    
    -- Soft-close features
    soft_close_bid_count INTEGER,
    soft_close_extensions INTEGER,
    
    -- Sequence embedding (from LSTM/GRU)
    sequence_embedding FLOAT[],
    embedding_model VARCHAR,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- MODEL PREDICTIONS TABLE
-- ============================================================================

-- Store predictions for analysis and comparison
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    
    -- Model predictions
    pred_tabular DECIMAL(10, 2),
    pred_image DECIMAL(10, 2),
    pred_text DECIMAL(10, 2),
    pred_sequential DECIMAL(10, 2),
    pred_fusion DECIMAL(10, 2),  -- Final fused prediction
    
    -- Confidence intervals
    pred_fusion_lower DECIMAL(10, 2),
    pred_fusion_upper DECIMAL(10, 2),
    
    -- Model versions
    model_version VARCHAR,
    
    -- Actual outcome (for completed auctions)
    actual_winning_bid DECIMAL(10, 2),
    
    -- Prediction metadata
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_type VARCHAR  -- 'training', 'validation', 'live'
);


-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_items_auction ON items(auction_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
CREATE INDEX IF NOT EXISTS idx_items_winning_bid ON items(winning_bid);
CREATE INDEX IF NOT EXISTS idx_bids_item ON bids(item_id);
CREATE INDEX IF NOT EXISTS idx_bids_time ON bids(bid_time);
CREATE INDEX IF NOT EXISTS idx_bids_auction ON bids(auction_id);
CREATE INDEX IF NOT EXISTS idx_images_item ON item_images(item_id);
CREATE INDEX IF NOT EXISTS idx_auctions_end_time ON auctions(actual_end_time);


-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Items with all features joined
CREATE OR REPLACE VIEW v_items_full AS
SELECT 
    i.*,
    a.title AS auction_title,
    a.location_city,
    a.auction_type,
    a.actual_end_time AS auction_end_time,
    e.estimated_value_low,
    e.estimated_value_high,
    e.condition_score,
    e.popularity_score,
    ft.category_avg_price,
    ft.category_median_price
FROM items i
LEFT JOIN auctions a ON i.auction_id = a.auction_id
LEFT JOIN item_enriched e ON i.item_id = e.item_id
LEFT JOIN features_tabular ft ON i.item_id = ft.item_id;

-- View: Training data with target variable
CREATE OR REPLACE VIEW v_training_data AS
SELECT 
    i.item_id,
    i.winning_bid AS target,
    i.has_bids,
    i.is_sold,
    ft.*,
    fseq.bid_count AS seq_bid_count,
    fseq.unique_bidders AS seq_unique_bidders,
    fseq.soft_close_extensions
FROM items i
INNER JOIN features_tabular ft ON i.item_id = ft.item_id
LEFT JOIN features_sequential fseq ON i.item_id = fseq.item_id
WHERE i.winning_bid IS NOT NULL;  -- Only completed auctions


-- ============================================================================
-- NOTES
-- ============================================================================
-- 
-- Data Flow:
-- 1. Scrape auction → Insert into auctions table
-- 2. Scrape items → Insert into items table
-- 3. Scrape bids → Insert into bids table
-- 4. Fetch enriched → Insert into item_enriched table
-- 5. Download images → Update item_images table
-- 6. Compute features → Insert into features_* tables
-- 7. Run models → Insert into predictions table
--
-- DuckDB-specific:
-- - Using FLOAT[] for embedding storage (native array support)
-- - Using JSON type for flexible nested data
-- - Generated columns for derived boolean fields
-- 
-- ============================================================================
