# Auction Price Prediction 🔨

A full-stack data science project for predicting winning prices of MaxSold online auctions using multi-modal machine learning.

[![CI](https://github.com/Jonathan-Pearce/Auction-Price-Prediction-/actions/workflows/ci.yml/badge.svg)](https://github.com/Jonathan-Pearce/Auction-Price-Prediction-/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/jonathan-pearce/auction-predictor)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project scrapes auction data from [MaxSold.com](https://maxsold.com), trains multiple specialized ML models, and deploys a web interface for real-time price predictions.

**Key Features:**
- 📊 **Multi-modal ML**: Combines tabular, image, text, and sequential models
- 🔄 **Fusion Model**: Meta-learner that combines predictions from all models
- 🌐 **Web Interface**: Gradio UI for easy predictions via URL input
- 📈 **~1M Training Items**: Scraped from ~10,000 completed auctions

## Architecture

```
User enters MaxSold URL
        ↓
┌───────────────────────────────────────┐
│           FastAPI Backend             │
├───────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Tabular │ │  Image  │ │  Text   │ │
│  │  Model  │ │  Model  │ │  Model  │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ │
│       └───────────┼───────────┘      │
│              ┌────┴────┐             │
│              │ Fusion  │             │
│              │  Model  │             │
│              └────┬────┘             │
└───────────────────┼───────────────────┘
                    ↓
         Predicted Price + Confidence
```

## Quick Start

### Option 1: GitHub Codespaces (Recommended)

1. Click **Code** → **Codespaces** → **Create codespace on main**
2. Wait for container to build (~2-3 min)
3. Run: `make setup`

### Option 2: Local Docker

```bash
# Clone repository
git clone https://github.com/Jonathan-Pearce/Auction-Price-Prediction-.git
cd Auction-Price-Prediction-

# Start development container
docker-compose up -d
docker-compose exec app bash

# Install dependencies
make setup
```

### Option 3: Local Python

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e ".[dev]"
```

## Usage

### Data Collection

**Auction-level data** (aggregated metrics per auction):
```bash
# Scrape auction-level data with aggregated item metrics
python -m src.data.auction_scraper --limit 100

# Upload to Hugging Face
python -m src.data.auction_scraper --limit 100 --upload-hf
```

**Item-level data** (detailed item and bid history):
```bash
# Scrape item-level data from MaxSold
make scrape-items

# Or run directly
python -m src.data.item_scraper

# Scrape sample (10 auctions)
make scrape-items-sample
```

**Enriched item data** (detailed item attributes, brands, categories):
```bash
# Scrape enriched item data
make scrape-enriched

# Or run directly
python -m src.data.enriched_item_scraper

# Scrape sample (100 items)
make scrape-enriched-sample
```

See documentation:
- [AUCTION_SCRAPER.md](docs/AUCTION_SCRAPER.md) - Auction-level data
- [ITEM_SCRAPER.md](docs/ITEM_SCRAPER.md) - Item-level data
- [ENRICHED_ITEM_SCRAPER.md](docs/ENRICHED_ITEM_SCRAPER.md) - Enriched item data

### Training Models

```bash
# Train all models
make train

# Train specific model
python -m src.modeling.train --model tabular
```

### Running the API

```bash
# Development server (hot reload)
make api-dev

# Production server
make api-prod
```

Visit `http://localhost:8000` for the API docs or `http://localhost:7860` for the Gradio interface.

## Project Structure

```
├── api/                    # FastAPI + Gradio application
│   ├── main.py            # API endpoints
│   └── gradio_app.py      # Gradio UI
├── src/                    # Main source code
│   ├── data/              # Data collection (scraping)
│   │   ├── maxsold_client.py
│   │   ├── scraper.py     # Item-level scraper
│   │   ├── auction_scraper.py  # Auction-level scraper
│   │   ├── scraper_config.py   # Scraper configuration
│   │   └── schemas.py
│   ├── modeling/          # ML models
│   │   ├── train.py
│   │   └── predict.py
│   ├── config.py          # Configuration
│   ├── dataset.py         # Data loading
│   └── features.py        # Feature engineering
├── data/                   # Local data (gitignored)
├── models/                 # Model checkpoints (gitignored)
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── references/             # Data schemas, API docs
└── tests/                  # Test files
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch |
| Statistical Models | scikit-learn |
| Database | DuckDB |
| API | FastAPI |
| UI | Gradio |
| Data Hosting | Hugging Face Datasets |
| Model Hosting | Hugging Face Hub |
| Deployment | Hugging Face Spaces |

## Models

| Model | Input | Architecture |
|-------|-------|--------------|
| Tabular | Structured features | XGBoost / Neural Network |
| Image | Item photos | EfficientNet / ResNet |
| Text | Title + description | DistilBERT / Transformer |
| Sequential | Bid history | LSTM / GRU |
| Fusion | All predictions | Meta-learner ensemble |

## Documentation

- [System Design](docs/DESIGN.md) - Architecture and data flow
- [Data Documentation](docs/DATA.md) - Data collection and storage
- [Model Documentation](docs/MODELS.md) - ML model details
- [Text Feature Engineering](docs/TEXT_FEATURES.md) - Text feature extraction methods
- [Deployment Guide](docs/DEPLOYMENT.md) - Hugging Face deployment
- [Development Guide](docs/DEVELOPMENT.md) - Development workflow

## Make Commands

```bash
make help          # Show all available commands
make setup         # Install dependencies
make lint          # Run linting (Ruff)
make format        # Format code (Black)
make test          # Run tests
make scrape        # Run data scraper
make train         # Train models
make api-dev       # Start development server
make deploy        # Deploy to Hugging Face
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
HF_TOKEN=your_huggingface_token
DUCKDB_PATH=data/auction_data.duckdb
MAXSOLD_RATE_LIMIT=2.0
```

## Data Source

Data is collected from [MaxSold.com](https://maxsold.com) via their public API:
- ~10,000 completed auctions
- ~1,000,000 items with full bidding history
- Images, descriptions, categories, and metadata

**Auction Mechanisms:**
- **Soft Close**: Bids in the last 2 minutes extend auction by 2 minutes
- **Zero-Bid Items**: Some items receive no bids (winning price = $0)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit changes: `git commit -m "feat: add new feature"`
4. Push to branch: `git push origin feat/my-feature`
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MaxSold](https://maxsold.com) for auction data
- [Hugging Face](https://huggingface.co) for hosting
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for project structure inspiration
