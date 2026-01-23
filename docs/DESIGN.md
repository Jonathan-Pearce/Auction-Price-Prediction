# System Design Document

> **Auction Price Prediction System**
> Technical architecture and design decisions

## Overview

This document describes the system architecture for predicting winning prices of MaxSold online auctions. The system collects auction data, trains multiple specialized ML models, and deploys a web interface for real-time predictions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MaxSold Public API                                                         │
│   ├── /auctions/items?auctionid={id}          → Auction item list           │
│   ├── /auctions/items?auctionid={id}&itemid={id} → Item + bid history       │
│   └── /listings/am/{id}/enriched              → Enriched metadata           │
│                                                                              │
│              ↓                                                               │
│   ┌──────────────────┐                                                       │
│   │  MaxSold Client  │ ← Rate limiting, retry logic, validation             │
│   └──────────────────┘                                                       │
│              ↓                                                               │
│   ┌──────────────────┐                                                       │
│   │     Scraper      │ ← Orchestrates collection of ~10k auctions           │
│   └──────────────────┘                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA STORAGE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│   │   Hugging Face Datasets     │    │         DuckDB              │        │
│   │   (Raw Data - Parquet)      │    │   (Feature Store & Serving) │        │
│   ├─────────────────────────────┤    ├─────────────────────────────┤        │
│   │ • auctions.parquet          │    │ • auctions table            │        │
│   │ • items.parquet             │    │ • items table               │        │
│   │ • bids.parquet              │    │ • bids table                │        │
│   │ • images/                   │    │ • features_* tables         │        │
│   └─────────────────────────────┘    │ • predictions table         │        │
│                                      └─────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE ENGINEERING LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │   Tabular    │  │    Image     │  │    Text      │  │  Sequential  │    │
│   │   Features   │  │   Features   │  │   Features   │  │   Features   │    │
│   ├──────────────┤  ├──────────────┤  ├──────────────┤  ├──────────────┤    │
│   │ • Category   │  │ • CNN embed  │  │ • Title emb  │  │ • Bid sequence│   │
│   │ • Price hist │  │ • Color hist │  │ • Desc emb   │  │ • Time deltas │   │
│   │ • Bid count  │  │ • Object det │  │ • Keywords   │  │ • Velocity    │   │
│   │ • Time feats │  │ • Quality    │  │ • Sentiment  │  │ • Soft-close  │   │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL TRAINING LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │   Tabular    │  │    Image     │  │    Text      │  │  Sequential  │    │
│   │    Model     │  │    Model     │  │    Model     │  │    Model     │    │
│   ├──────────────┤  ├──────────────┤  ├──────────────┤  ├──────────────┤    │
│   │ XGBoost /    │  │ ResNet /     │  │ DistilBERT / │  │ LSTM /       │    │
│   │ Neural Net   │  │ EfficientNet │  │ Transformer  │  │ GRU          │    │
│   │ (sklearn/    │  │ (PyTorch)    │  │ (PyTorch/HF) │  │ (PyTorch)    │    │
│   │  PyTorch)    │  │              │  │              │  │              │    │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│              ↘              ↓              ↓              ↙                  │
│                    ┌──────────────────────────┐                              │
│                    │      Fusion Model        │                              │
│                    │   (Meta-learner)         │                              │
│                    ├──────────────────────────┤                              │
│                    │ • Weighted average       │                              │
│                    │ • Stacking ensemble      │                              │
│                    │ • Neural fusion          │                              │
│                    └──────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Hugging Face Spaces                                                        │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                                                                  │       │
│   │   ┌──────────────┐         ┌──────────────────────────┐         │       │
│   │   │  Gradio UI   │ ←────→  │      FastAPI Backend     │         │       │
│   │   ├──────────────┤         ├──────────────────────────┤         │       │
│   │   │ • URL input  │         │ • /predict endpoint      │         │       │
│   │   │ • Prediction │         │ • MaxSold API client     │         │       │
│   │   │ • Confidence │         │ • Feature engineering    │         │       │
│   │   │ • History    │         │ • Model inference        │         │       │
│   │   └──────────────┘         │ • DuckDB queries         │         │       │
│   │                            └──────────────────────────┘         │       │
│   │                                                                  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   Hugging Face Hub                                                           │
│   ├── Models: jonathan-pearce/auction-predictor-*                           │
│   └── Datasets: jonathan-pearce/maxsold-auctions                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Data Collection Flow

```
Scraper starts
    │
    ├─→ Fetch auction list (discover auction IDs)
    │
    ├─→ For each auction:
    │       ├─→ Fetch auction items (items endpoint)
    │       ├─→ For each item:
    │       │       ├─→ Fetch bid history (item detail endpoint)
    │       │       ├─→ Fetch enriched data (enriched endpoint)
    │       │       └─→ Download images
    │       └─→ Store to DuckDB
    │
    └─→ Upload to HF Datasets (batch)
```

### 2. Training Flow

```
Load data from HF Datasets
    │
    ├─→ Feature Engineering
    │       ├─→ Tabular features → features_tabular
    │       ├─→ Text embeddings → features_text
    │       ├─→ Image embeddings → features_image
    │       └─→ Sequence features → features_sequential
    │
    ├─→ Train Base Models (parallel)
    │       ├─→ Tabular model
    │       ├─→ Image model
    │       ├─→ Text model
    │       └─→ Sequential model
    │
    ├─→ Generate base model predictions
    │
    ├─→ Train Fusion Model
    │
    └─→ Push models to HF Hub
```

### 3. Inference Flow

```
User enters MaxSold item URL
    │
    ├─→ Parse item_id from URL
    │
    ├─→ Fetch live data from MaxSold API
    │       ├─→ Item details
    │       ├─→ Current bids
    │       └─→ Enriched info
    │
    ├─→ Feature engineering (real-time)
    │
    ├─→ Run through all models
    │       ├─→ Tabular prediction
    │       ├─→ Image prediction
    │       ├─→ Text prediction
    │       └─→ Sequential prediction
    │
    ├─→ Fusion model combines predictions
    │
    └─→ Return prediction with confidence interval
```

---

## Handling Special Cases

### Zero-Bid Items

Items that receive no bids (winning_price = $0) require special handling:

**In Training:**
- Include as valid training examples
- Consider binary classification: "will receive bids?" + regression: "if bids, what price?"
- Alternative: Treat as censored data (Tobit regression)

**In Feature Engineering:**
- Track historical no-bid rates by category
- Include seller/auction-level no-bid history

**In Prediction:**
- Output probability of receiving bids alongside price prediction
- Example output: "70% chance of bids, predicted price if sold: $45"

### Soft-Close Mechanism

Bids in the last 2 minutes extend the auction by 2 minutes:

**Impact on Data:**
- `actual_end_time` may differ from `scheduled_end_time`
- Bid timestamps near end-time need context

**Feature Engineering:**
- Count of soft-close extensions
- Bids per extension round
- Bidder behavior during soft-close

**Model Considerations:**
- Sequential model should capture soft-close dynamics
- Time-to-close features must account for extensions

---

## Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Deep Learning | PyTorch | Flexibility, research-friendly, HF integration |
| Statistical Models | scikit-learn | Robust, well-tested, easy baselines |
| Database | DuckDB | Analytical queries, Parquet support, embedded |
| API Framework | FastAPI | Async, type hints, auto-docs, HF compatible |
| UI Framework | Gradio | Python-only, fast prototyping, HF native |
| Data Storage | HF Datasets | ML-native, versioned, free hosting |
| Model Hosting | HF Hub | Easy loading, versioning, sharing |
| Deployment | HF Spaces | Free tier, Docker support, integrated |

---

## API Design

### External Endpoint (User-facing)

```
POST /predict
{
    "url": "https://maxsold.com/auction/123/item/456"
}

Response:
{
    "item_id": 456,
    "item_title": "Vintage Oak Dresser",
    "current_bid": 45.00,
    "prediction": {
        "expected_price": 85.50,
        "confidence_interval": [65.00, 120.00],
        "probability_of_sale": 0.92
    },
    "model_contributions": {
        "tabular": 80.00,
        "image": 90.00,
        "text": 85.00,
        "sequential": 87.00
    }
}
```

### Internal Endpoints (Development/Admin)

```
GET /health                    # Health check
GET /models/info               # Model versions and metadata
POST /predict/batch            # Batch predictions
GET /metrics                   # Prediction metrics and stats
```

---

## Development Workflow

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed workflow documentation.

### Git Workflow

```
main (protected)
  └── feat/feature-name
  └── fix/bug-description
  └── data/scraping-update
  └── model/experiment-name
```

### CI/CD Pipeline

1. **On Push**: Run tests, linting
2. **On PR to main**: Full test suite, model validation
3. **On Merge to main**: Deploy to HF Spaces

---

## Security Considerations

- **API Keys**: Stored in HF Secrets, never in code
- **Rate Limiting**: Respect MaxSold API limits
- **Data Privacy**: No personal bidder information exposed
- **Input Validation**: Sanitize all URL inputs

---

## Future Enhancements

1. **Real-time updates**: WebSocket for live bid tracking
2. **User accounts**: Save prediction history
3. **Model improvements**: Attention-based fusion, multi-task learning
4. **Additional data sources**: Similar auction platforms
5. **Mobile app**: React Native or Flutter frontend

---

*Last updated: January 2026*
