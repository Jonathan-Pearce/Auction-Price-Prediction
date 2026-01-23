---
applyTo: "src/modeling/**/*.py,src/features.py,src/dataset.py"
---

# ML Training Agent Instructions

You are a machine learning specialist for the Auction Price Prediction project. Your focus is on model development, training, and evaluation.

## Your Domain

- Feature engineering (`src/features.py`)
- Dataset utilities (`src/dataset.py`)
- Model training (`src/modeling/train.py`)
- Model inference (`src/modeling/predict.py`)
- Model evaluation and metrics

## Model Architecture

### Base Models (4 total)

1. **Tabular Model**
   - Input: Structured auction/item features
   - Options: XGBoost, Random Forest, or Fully Connected NN
   - Features: category, starting_bid, num_images, auction_type, location, etc.

2. **Image Model**
   - Input: Item photos (primary image)
   - Architecture: Pretrained CNN (ResNet50, EfficientNet) with regression head
   - Transfer learning from ImageNet

3. **Text Model**
   - Input: Item title + description
   - Architecture: Transformer (DistilBERT, RoBERTa) with regression head
   - Fine-tune on auction descriptions

4. **Sequential Model**
   - Input: Bid history time series
   - Architecture: LSTM or GRU
   - Features per timestep: bid_amount, time_delta, bid_increment

### Fusion Model
- Combines predictions from all base models
- Options:
  - Simple: Weighted average
  - Linear: Ridge regression on base predictions
  - Neural: Small MLP on concatenated predictions
  - Stacking: XGBoost on base predictions + features

## Key Considerations

### Target Variable
- Regression task: predict `winning_price` (continuous, ≥ 0)
- Handle zero-bid items:
  - Option 1: Include as 0 (may need special treatment)
  - Option 2: Two-stage model (classify has_bids, then regress price)

### Evaluation Metrics
- Primary: MAE (Mean Absolute Error) - interpretable in dollars
- Secondary: RMSE, MAPE, R²
- Business: % of predictions within $X of actual

### Data Splits
- Time-based split recommended (train on older auctions, test on recent)
- Stratify by auction_type or price_range if needed
- Typical: 70% train, 15% validation, 15% test

### Loss Functions
- MSE or MAE loss for regression
- Consider Huber loss for robustness to outliers
- For zero-inflated data, consider custom losses

## Code Patterns

### Trainer Base Class
```python
class BaseTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.output_dir = settings.models_dir / model_name
        
    def load_data(self) -> None: ...
    def build_model(self) -> None: ...
    def train(self) -> dict[str, float]: ...
    def evaluate(self) -> dict[str, float]: ...
    def save(self) -> None: ...
```

### PyTorch Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
```

### Feature Engineering
```python
def engineer_item_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    
    # Numerical features
    features['description_length'] = features['description'].str.len()
    features['num_images'] = features['images'].apply(len)
    
    # Categorical encoding
    features = pd.get_dummies(features, columns=['category'])
    
    return features
```

## Device Management

```python
device = settings.training.device  # auto-detected: cuda, mps, or cpu

model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}
```

## Model Saving

### PyTorch Models
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
    'metrics': metrics,
}, path)
```

### Hugging Face Hub
```python
model.push_to_hub(settings.huggingface.model_id)
```

## Experiment Tracking

Consider integrating:
- Weights & Biases (`wandb`)
- MLflow
- TensorBoard

Log: hyperparameters, metrics, model artifacts, data versions

## Common Issues

1. **Overfitting**: Use dropout, early stopping, data augmentation
2. **Class Imbalance**: Many items have low prices - consider weighted sampling
3. **Missing Values**: Handle NaN in features explicitly
4. **GPU Memory**: Reduce batch size or use gradient accumulation
