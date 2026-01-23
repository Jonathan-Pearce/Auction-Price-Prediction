# Model Architecture Documentation

> **ML models for auction price prediction**

## Overview

The prediction system uses four specialized models that capture different aspects of auction items, combined through a fusion model for final predictions.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Input: Auction Item                          │
└──────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────┬───────────┼───────────┬──────────────┐
         ↓              ↓           ↓           ↓              ↓
    ┌─────────┐   ┌─────────┐ ┌─────────┐ ┌─────────┐   ┌───────────┐
    │ Tabular │   │  Image  │ │  Text   │ │Sequential│   │ (Future)  │
    │  Model  │   │  Model  │ │  Model  │ │  Model  │   │  Models   │
    └────┬────┘   └────┬────┘ └────┬────┘ └────┬────┘   └───────────┘
         │              │           │           │
         └──────────────┴─────┬─────┴───────────┘
                              ↓
                    ┌─────────────────┐
                    │  Fusion Model   │
                    │  (Meta-learner) │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Prediction    │
                    │ + Uncertainty   │
                    └─────────────────┘
```

---

## Model 1: Tabular Model

### Purpose
Predict price from structured features: category, auction metadata, historical patterns.

### Architecture Options

**Option A: Gradient Boosting (Default)**
```python
# XGBoost configuration
model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

**Option B: Neural Network**
```python
# Fully connected network
class TabularNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### Input Features

| Feature Group | Features | Dimension |
|--------------|----------|-----------|
| Item | word_count, desc_length, image_count | 3 |
| Category | category_encoded (one-hot or embedding) | ~50 |
| Condition | condition_encoded | 5 |
| Auction | total_items, lot_position | 2 |
| Time | day_of_week, hour (cyclic encoded) | 4 |
| Historical | category_avg, category_median, sell_rate | 3 |
| Enriched | estimated_value, condition_score | 3 |
| **Total** | | **~70** |

### Training

```python
# Training configuration
config = {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
    "loss": "mse",  # or "huber" for robustness
}
```

---

## Model 2: Image Model

### Purpose
Extract visual features from item photos to predict value based on appearance, quality, and item type.

### Architecture

**Base: EfficientNet-B0** (or ResNet-50)

```python
class ImageModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(1280, 256),  # EfficientNet-B0 output dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def get_embedding(self, x):
        """Extract features for fusion model."""
        return self.backbone(x)
```

### Input Processing

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Multi-Image Handling

```python
def aggregate_image_embeddings(embeddings: List[Tensor], method: str = "mean"):
    """Combine multiple image embeddings for one item."""
    stacked = torch.stack(embeddings)
    
    if method == "mean":
        return stacked.mean(dim=0)
    elif method == "max":
        return stacked.max(dim=0)[0]
    elif method == "attention":
        # Learn attention weights
        weights = self.attention(stacked)
        return (stacked * weights).sum(dim=0)
```

### Training

```python
config = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "augmentation": ["horizontal_flip", "color_jitter", "random_crop"],
}
```

---

## Model 3: Text Model

### Purpose
Understand item descriptions and titles to predict value based on semantic content, keywords, and sentiment.

### Architecture

**Base: DistilBERT** (or sentence-transformers)

```python
from transformers import AutoModel, AutoTokenizer

class TextModel(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze lower layers (optional)
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Regression head
        hidden_size = self.encoder.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return self.head(cls_embedding)
    
    def get_embedding(self, input_ids, attention_mask):
        """Extract features for fusion model."""
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
```

### Input Processing

```python
def prepare_text(title: str, description: str, max_length: int = 256):
    """Combine and tokenize item text."""
    combined = f"{title} [SEP] {description}"
    
    encoding = tokenizer(
        combined,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    return encoding["input_ids"], encoding["attention_mask"]
```

### Training

```python
config = {
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
}
```

---

## Model 4: Sequential Model

### Purpose
Capture bidding dynamics: velocity, timing patterns, competition intensity.

### Architecture

**LSTM-based sequence model**

```python
class SequentialModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,  # amount, time_delta, bidder_id_encoded, is_soft_close
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        
        # Attention over sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, lengths):
        # Pack sequence for efficiency
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # Attention-weighted sum
        attention_weights = F.softmax(self.attention(output), dim=1)
        context = (output * attention_weights).sum(dim=1)
        
        return self.head(context)
```

### Input Features (per bid)

| Feature | Description | Type |
|---------|-------------|------|
| `amount` | Bid amount (normalized) | float |
| `time_delta` | Seconds since previous bid | float |
| `bidder_encoded` | Bidder ID embedding | int |
| `is_soft_close` | Bid in final 2 minutes | bool |
| `position_in_auction` | Normalized time position | float |

### Handling Variable Length

```python
def collate_bid_sequences(batch):
    """Pad sequences to same length in batch."""
    sequences = [item["bids"] for item in batch]
    targets = torch.tensor([item["winning_price"] for item in batch])
    
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    
    # Pad sequences
    padded = torch.zeros(len(sequences), max_len, seq_dim)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    return padded, torch.tensor(lengths), targets
```

### Training

```python
config = {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "clip_grad_norm": 5.0,
    "teacher_forcing_ratio": 0.5,  # If using decoder
}
```

---

## Fusion Model

### Purpose
Combine predictions from all base models to produce final price estimate.

### Architecture Options

**Option A: Learned Weighted Average**
```python
class WeightedFusion(nn.Module):
    def __init__(self, num_models: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, predictions: List[Tensor]):
        weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(predictions, dim=1)
        return (stacked * weights).sum(dim=1)
```

**Option B: Stacking with Meta-learner**
```python
class StackingFusion(nn.Module):
    def __init__(self, num_models: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.meta_learner = nn.Sequential(
            nn.Linear(num_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, predictions: List[Tensor]):
        stacked = torch.stack(predictions, dim=1)
        return self.meta_learner(stacked)
```

**Option C: Neural Fusion with Embeddings**
```python
class NeuralFusion(nn.Module):
    def __init__(self, embedding_dims: Dict[str, int]):
        super().__init__()
        
        total_dim = sum(embedding_dims.values())
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, embeddings: Dict[str, Tensor]):
        concatenated = torch.cat(list(embeddings.values()), dim=1)
        return self.fusion(concatenated)
```

### Training Strategy

**Two-stage training:**
1. Train base models independently
2. Freeze base models, train fusion on held-out data

```python
# Stage 2: Fusion training
for batch in fusion_dataloader:
    with torch.no_grad():
        pred_tabular = tabular_model(batch["tabular"])
        pred_image = image_model(batch["images"])
        pred_text = text_model(batch["text"])
        pred_seq = sequential_model(batch["bids"])
    
    final_pred = fusion_model([pred_tabular, pred_image, pred_text, pred_seq])
    loss = criterion(final_pred, batch["target"])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Uncertainty Estimation

### Quantile Regression

```python
class QuantileHead(nn.Module):
    def __init__(self, input_dim: int, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in quantiles
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)

def quantile_loss(preds, targets, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())
    return sum(losses)
```

### Output Format

```python
{
    "expected_price": 85.50,        # Median prediction (q=0.5)
    "lower_bound": 65.00,           # 10th percentile (q=0.1)
    "upper_bound": 120.00,          # 90th percentile (q=0.9)
    "confidence": 0.80,             # 80% confidence interval
}
```

---

## Model Evaluation

### Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| MAE | mean(\|y - ŷ\|) | Average error magnitude |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| MAPE | mean(\|y - ŷ\| / y) | Percentage error |
| R² | 1 - SS_res/SS_tot | Variance explained |

### Evaluation Split

```
Total Data (1M items)
├── Train (70%): 700k items
├── Validation (15%): 150k items
└── Test (15%): 150k items

Split by auction (not item) to prevent leakage
```

### Zero-Bid Handling

```python
# Separate metrics for items with/without bids
metrics = {
    "all_items": compute_metrics(y_true, y_pred),
    "items_with_bids": compute_metrics(y_true[has_bids], y_pred[has_bids]),
    "items_no_bids": compute_metrics(y_true[~has_bids], y_pred[~has_bids]),
}
```

---

## Model Versioning

### Hugging Face Hub

```python
# Push model to Hub
model.push_to_hub(
    "jonathan-pearce/auction-predictor-tabular",
    commit_message="v1.0.0: Initial tabular model",
)

# Load model from Hub
model = TabularModel.from_pretrained("jonathan-pearce/auction-predictor-tabular")
```

### Version Naming

```
auction-predictor-tabular-v1.0.0
auction-predictor-image-v1.0.0
auction-predictor-text-v1.0.0
auction-predictor-sequential-v1.0.0
auction-predictor-fusion-v1.0.0
```

---

*Last updated: January 2026*
