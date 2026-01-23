# Deployment Documentation

> **Deploying the auction price prediction system to Hugging Face**

## Overview

The system is deployed as a Hugging Face Space combining FastAPI backend with Gradio UI, with data and models hosted on Hugging Face Hub.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hugging Face Ecosystem                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              HF Space: auction-predictor                │   │
│   │                                                         │   │
│   │   ┌───────────────┐      ┌───────────────────────┐     │   │
│   │   │   Gradio UI   │ ←──→ │   FastAPI Backend     │     │   │
│   │   └───────────────┘      └───────────────────────┘     │   │
│   │                                   │                     │   │
│   │                          ┌────────┴────────┐           │   │
│   │                          ↓                 ↓           │   │
│   │                   ┌──────────┐      ┌──────────┐       │   │
│   │                   │  DuckDB  │      │  Models  │       │   │
│   │                   │  (local) │      │ (loaded) │       │   │
│   │                   └──────────┘      └──────────┘       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────┐    ┌─────────────────────────────┐   │
│   │  HF Hub: Models     │    │  HF Hub: Datasets           │   │
│   │  • tabular-v1.0     │    │  • maxsold-auctions         │   │
│   │  • image-v1.0       │    │    ├── auctions.parquet     │   │
│   │  • text-v1.0        │    │    ├── items.parquet        │   │
│   │  • sequential-v1.0  │    │    └── bids.parquet         │   │
│   │  • fusion-v1.0      │    │                             │   │
│   └─────────────────────┘    └─────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hugging Face Space Setup

### Space Configuration

**File: `README.md` (Space card)**
```yaml
---
title: Auction Price Predictor
emoji: 🔨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

### Dockerfile for Space

**File: `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 7860

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Space File Structure

```
auction-predictor/
├── README.md           # Space card with YAML frontmatter
├── Dockerfile          # Custom Docker configuration
├── requirements.txt    # Python dependencies
├── api/
│   ├── __init__.py
│   ├── main.py         # FastAPI app
│   └── gradio_app.py   # Gradio interface
├── src/
│   ├── config.py
│   ├── features.py
│   └── ...
└── models/             # Small model files (or load from Hub)
```

---

## Deployment Methods

### Method 1: GitHub Actions (Recommended)

**File: `.github/workflows/deploy-to-hf.yml`**
```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [main]
    paths:
      - 'api/**'
      - 'src/**'
      - 'Dockerfile'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://huggingface.co/spaces/jonathan-pearce/auction-predictor
          git push hf main --force
```

### Method 2: Dual Git Remotes

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/jonathan-pearce/auction-predictor

# Deploy
git push hf main
```

### Method 3: HF CLI

```bash
# Install CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload Space
huggingface-cli upload jonathan-pearce/auction-predictor . --repo-type space
```

---

## Model Deployment

### Uploading Models to Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Create model repository
api.create_repo(
    repo_id="jonathan-pearce/auction-predictor-fusion",
    repo_type="model",
    private=False,
)

# Upload model files
api.upload_folder(
    folder_path="./models/fusion",
    repo_id="jonathan-pearce/auction-predictor-fusion",
    repo_type="model",
)
```

### Model Loading in Production

```python
from huggingface_hub import hf_hub_download
import torch

class ModelLoader:
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_name: str):
        """Load model from Hugging Face Hub."""
        # Download model file
        model_path = hf_hub_download(
            repo_id=f"jonathan-pearce/auction-predictor-{model_name}",
            filename="model.pt",
        )
        
        # Load model
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        
        self.models[model_name] = model
        return model
    
    def load_all(self):
        """Load all models at startup."""
        for name in ["tabular", "image", "text", "sequential", "fusion"]:
            self.load_model(name)
```

---

## Gradio Interface

### Basic Interface

```python
import gradio as gr
from api.main import predict

def predict_interface(url: str):
    """Gradio prediction interface."""
    try:
        result = predict(url)
        
        output = f"""
        ## Prediction Results
        
        **Item:** {result['item_title']}
        
        **Current Bid:** ${result['current_bid']:.2f}
        
        ### Predicted Winning Price
        
        | Metric | Value |
        |--------|-------|
        | Expected | ${result['prediction']['expected_price']:.2f} |
        | Lower Bound (10%) | ${result['prediction']['lower_bound']:.2f} |
        | Upper Bound (90%) | ${result['prediction']['upper_bound']:.2f} |
        | Probability of Sale | {result['prediction']['probability_of_sale']:.1%} |
        
        ### Model Contributions
        
        - Tabular: ${result['model_contributions']['tabular']:.2f}
        - Image: ${result['model_contributions']['image']:.2f}
        - Text: ${result['model_contributions']['text']:.2f}
        - Sequential: ${result['model_contributions']['sequential']:.2f}
        """
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create interface
demo = gr.Interface(
    fn=predict_interface,
    inputs=gr.Textbox(
        label="MaxSold Item URL",
        placeholder="https://maxsold.com/auction/123/item/456"
    ),
    outputs=gr.Markdown(label="Prediction"),
    title="🔨 Auction Price Predictor",
    description="Enter a MaxSold auction item URL to predict the winning price.",
    examples=[
        ["https://maxsold.com/auction/99941/item/7433850"],
        ["https://maxsold.com/auction/103293/item/7433850"],
    ],
)
```

### Advanced Interface with Tabs

```python
with gr.Blocks() as demo:
    gr.Markdown("# 🔨 Auction Price Predictor")
    
    with gr.Tabs():
        with gr.TabItem("Predict"):
            url_input = gr.Textbox(label="MaxSold Item URL")
            predict_btn = gr.Button("Predict", variant="primary")
            result_output = gr.Markdown()
            
            predict_btn.click(
                predict_interface,
                inputs=url_input,
                outputs=result_output,
            )
        
        with gr.TabItem("Batch Predict"):
            file_input = gr.File(label="Upload CSV with URLs")
            batch_output = gr.Dataframe()
        
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About This Model
            
            This model predicts winning prices for MaxSold auctions using:
            - **Tabular features**: Category, condition, historical prices
            - **Image analysis**: Visual features from item photos
            - **Text analysis**: Title and description semantics
            - **Bid patterns**: Bidding history dynamics
            
            ### Data Source
            Trained on ~1 million items from ~10,000 MaxSold auctions.
            """)

demo.launch()
```

---

## Environment Variables

### Required Secrets (HF Space Settings)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token |
| `MAXSOLD_RATE_LIMIT` | API rate limit (requests/sec) |

### Setting Secrets

1. Go to Space Settings
2. Click "Variables and secrets"
3. Add secrets (encrypted)

### Accessing in Code

```python
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
RATE_LIMIT = float(os.environ.get("MAXSOLD_RATE_LIMIT", "2.0"))
```

---

## Monitoring & Logging

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(model_loader.models.keys()),
        "timestamp": datetime.utcnow().isoformat(),
    }
```

### Request Logging

```python
import logging
from loguru import logger

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    
    return response
```

---

## Scaling & Performance

### Free Tier Limitations

| Resource | Limit |
|----------|-------|
| CPU | 2 vCPU |
| RAM | 16 GB |
| Storage | 50 GB |
| Timeout | 60 seconds |

### Optimization Strategies

1. **Model quantization**: Reduce model size
2. **Lazy loading**: Load models on first request
3. **Caching**: Cache frequent predictions
4. **Batch processing**: Process multiple items together

### Caching Example

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_item_features(item_id: int):
    """Cache computed features."""
    return compute_features(item_id)
```

---

## Upgrading (Paid Tier)

For production workloads:

| Tier | CPU | RAM | Cost |
|------|-----|-----|------|
| Free | 2 vCPU | 16 GB | $0 |
| CPU Basic | 2 vCPU | 16 GB | $5/mo |
| CPU Upgrade | 8 vCPU | 32 GB | $30/mo |
| GPU T4 | 4 vCPU + T4 | 15 GB | $0.60/hr |

---

## Rollback & Versioning

### Git-based Rollback

```bash
# View deployment history
git log --oneline hf/main

# Rollback to previous version
git push hf <commit-sha>:main --force
```

### Model Versioning

```python
# Load specific model version
model = load_model("jonathan-pearce/auction-predictor-fusion", revision="v1.0.0")
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Space not building | Check Dockerfile syntax |
| Models not loading | Verify HF_TOKEN secret |
| Timeout errors | Reduce model size, add caching |
| Memory errors | Use model quantization |

### Viewing Logs

```bash
# HF Space logs (via web UI)
Settings > "Logs" tab

# Or via API
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/spaces/jonathan-pearce/auction-predictor/logs
```

---

*Last updated: January 2026*
