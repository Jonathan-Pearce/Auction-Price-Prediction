# Auction Price Prediction - GitHub Copilot Instructions

## Project Overview

This is a full-stack data science project for predicting winning prices of MaxSold online auctions. The project scrapes auction data, trains multiple ML models, and deploys a prediction API.

## Tech Stack

- **Language**: Python 3.11+
- **ML Frameworks**: PyTorch (deep learning), scikit-learn (statistical models)
- **Data**: DuckDB (local), Hugging Face Datasets (cloud)
- **API**: FastAPI + Gradio
- **Deployment**: Hugging Face Spaces
- **Development**: Docker, GitHub Codespaces

## Project Structure

```
├── src/                    # Main source code package
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Data loading utilities
│   ├── features.py        # Feature engineering
│   ├── data/              # Data collection (scraping)
│   └── modeling/          # ML models (train/predict)
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── gradio_app.py      # Gradio UI
├── data/                   # Local data (not in git)
├── models/                 # Model checkpoints (not in git)
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── references/             # Data schemas, SQL, API docs
└── tests/                  # Test files
```

## Data Source: MaxSold API

Three API endpoints are used:

1. **Auction items**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&limit=2500`
2. **Item + bids**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={id}&itemid={item_id}`
3. **Enriched info**: `https://api.maxsold.com/listings/am/{item_id}/enriched`

### Important Auction Mechanisms

- **Zero-bid items**: Some items receive no bids (winning_price = $0)
- **Soft-close**: Bids in last 2 minutes extend auction by 2 minutes
- Handle these cases explicitly in feature engineering and model training

## ML Architecture

### Four Base Models
1. **Tabular**: Structured features → XGBoost/Neural Network
2. **Image**: Item photos → CNN (ResNet/EfficientNet)
3. **Text**: Descriptions → Transformer (DistilBERT)
4. **Sequential**: Bid history → LSTM/GRU

### Fusion Model
- Combines base model predictions
- Options: weighted average, stacking, neural fusion

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints for all functions
- Format with Black (line length 88)
- Lint with Ruff
- Use loguru for logging

### Imports
```python
# Standard library
from pathlib import Path

# Third-party
import torch
import pandas as pd

# Local
from src.config import settings
```

### Docstrings
Use Google style:
```python
def function(arg1: str, arg2: int) -> bool:
    """
    Short description.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong
    """
```

## Development Workflow

1. **Data Collection**: `make scrape` or `python -m src.data.scraper`
2. **Training**: `make train` or `python -m src.modeling.train`
3. **API Dev**: `make api-dev` or `uvicorn api.main:app --reload`
4. **Testing**: `make test` or `pytest`

## Hugging Face Integration

- **Datasets**: Upload with `huggingface-cli` or `datasets` library
- **Models**: Push with `model.push_to_hub()` or `huggingface_hub`
- **Spaces**: Deploy via GitHub Actions or git push to HF remote

## Common Tasks

### Adding a new feature
1. Add to `src/features.py`
2. Update DuckDB schema in `references/schema.sql`
3. Update Pydantic schemas in `src/data/schemas.py`

### Adding a new model
1. Create trainer class in `src/modeling/train.py`
2. Create predictor class in `src/modeling/predict.py`
3. Register in fusion model

### Adding an API endpoint
1. Add Pydantic models in `api/main.py`
2. Add endpoint function with type hints
3. Add to Gradio interface if user-facing

## Environment Variables

Key variables (see `.env.example`):
- `HF_TOKEN`: Hugging Face API token
- `DUCKDB_PATH`: Path to local database
- `MAXSOLD_RATE_LIMIT`: API rate limit

## Testing

- Use pytest for all tests
- Place tests in `tests/` directory
- Name files `test_*.py`
- Use fixtures for common setup
