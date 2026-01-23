---
applyTo: "api/**/*.py,**/gradio*.py,**/deploy*,**/*hf*"
---

# Deployment Agent Instructions

You are a deployment and API specialist for the Auction Price Prediction project. Your focus is on serving models, building APIs, and deploying to Hugging Face.

## Your Domain

- FastAPI application (`api/main.py`)
- Gradio interface (`api/gradio_app.py`)
- Hugging Face Spaces deployment
- GitHub Actions workflows
- Docker configuration

## API Architecture

### FastAPI Endpoints

```
GET  /           → Root/docs redirect
GET  /health     → Health check
GET  /models     → List available models
POST /predict    → Predict from item_id + auction_id
POST /predict/url → Predict from MaxSold URL
```

### Request/Response Models

```python
class PredictRequest(BaseModel):
    item_id: int
    auction_id: int
    item_data: dict | None = None  # Optional pre-fetched data

class PredictResponse(BaseModel):
    item_id: str
    predicted_price: float
    confidence_interval: dict[str, float]
    model_predictions: dict[str, float | None]
```

## Gradio Interface

### Structure
```python
with gr.Blocks() as demo:
    # Header with instructions
    gr.Markdown("# Auction Price Predictor")
    
    # Input
    url_input = gr.Textbox(label="MaxSold Item URL")
    predict_btn = gr.Button("Predict")
    
    # Output
    price_output = gr.Markdown()
    details_output = gr.Markdown()
    
    # Event handler
    predict_btn.click(predict_fn, inputs=[url_input], outputs=[...])
```

### For HF Spaces
```python
# Module-level demo for HF Spaces auto-detection
demo = create_interface()

# Or in app.py for Spaces
if __name__ == "__main__":
    demo.launch()
```

## Hugging Face Spaces Deployment

### Option 1: Docker Space

Create `Dockerfile` in space root:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api/gradio_app.py"]
```

### Option 2: Gradio Space

Create `app.py` in space root:
```python
import gradio as gr
# ... interface code ...
demo.launch()
```

### Space Configuration (`README.md` frontmatter)
```yaml
---
title: Auction Price Predictor
emoji: 🔨
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: api/gradio_app.py
pinned: false
---
```

## GitHub Actions for HF Sync

```yaml
name: Sync to Hugging Face
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Push to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push https://user:$HF_TOKEN@huggingface.co/spaces/USER/REPO main
```

## Model Loading

### Lazy Loading Pattern
```python
class ModelManager:
    _instance = None
    _models = {}
    
    @classmethod
    def get_predictor(cls) -> EnsemblePredictor:
        if "predictor" not in cls._models:
            cls._models["predictor"] = EnsemblePredictor(load_models=True)
        return cls._models["predictor"]
```

### From Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="username/auction-predictor",
    filename="tabular_model.pt"
)
model.load_state_dict(torch.load(model_path))
```

## Error Handling

```python
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        result = await predictor.predict(...)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
```

## CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_item_features(item_id: int) -> dict:
    # Cache frequently accessed items
    ...
```

### Async Operations
```python
async def predict(self, item_id: int) -> PredictResult:
    # Fetch data and run models concurrently
    item_data, enriched = await asyncio.gather(
        self.fetch_item(item_id),
        self.fetch_enriched(item_id),
    )
```

## Monitoring

- Log all predictions with item_id and response time
- Track prediction latency
- Monitor error rates
- Set up alerts for model drift

## Local Development

```bash
# FastAPI with hot reload
uvicorn api.main:app --reload --port 8000

# Gradio
python -m api.gradio_app --share  # Creates public URL

# Docker
docker-compose up app
```

## Common Issues

1. **Cold Starts**: HF Spaces free tier sleeps - first request is slow
2. **Model Size**: Keep models small for free tier (2GB memory limit)
3. **Timeouts**: Long predictions may timeout - consider async responses
4. **CORS**: Always configure for frontend access
