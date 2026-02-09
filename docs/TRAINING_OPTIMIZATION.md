# Training Optimization Guide

## Problem: Slow Training & High Memory Usage

If you're experiencing:
- ❌ Very long epochs (>10 minutes per epoch)
- ❌ Process getting killed (OOM - Out of Memory)
- ❌ Training taking too long

## Solutions Applied

### 1. **Reduced Steps Per Epoch** (BIGGEST IMPACT)

**Before**: 33,176 steps/epoch (~40 min/epoch) ❌  
**After**: 1,000 steps/epoch (~3-5 min/epoch) ✅

```yaml
# In src/ml_config.yaml
image_embeddings_model:
  training:
    steps_per_epoch: 1000  # Limit steps to avoid extremely long epochs
```

**Why it helps**: You don't need to see ALL training data each epoch. Random sampling gives good results.

### 2. **Increased Batch Size** (Speed + Memory Trade-off)

**Before**: 64  
**After**: 256 (4x larger)

```yaml
image_embeddings_model:
  training:
    batch_size: 256  # Increased from 64 for faster training
    learning_rate: 0.002  # Increased proportionally (was 0.001)
```

**Why it helps**:
- Fewer forward/backward passes per epoch
- Better GPU utilization (if using GPU)
- More stable gradients
- **Note**: Uses more memory per step, but overall faster

### 3. **Simplified Model Architecture**

**Before**: 576 → 256 → 128 → 1 (166,657 params)  
**After**: 576 → 128 → 64 → 1 (82,817 params)

```yaml
image_embeddings_model:
  hidden_layers: [128, 64]  # Reduced from [256, 128]
  dropout_rate: 0.2  # Reduced from 0.3
```

**Why it helps**:
- 50% fewer parameters = faster training
- Less memory per batch
- Still has enough capacity for this task

### 4. **Reduced Shuffle Buffer Size**

**Before**: 10,000 samples in buffer  
**After**: 2,000 samples in buffer

```yaml
image_embeddings_model:
  training:
    shuffle_buffer_size: 2000  # Reduced from 10000 for lower memory
```

**Why it helps**:
- Less RAM consumed by buffer
- Still provides good shuffling

### 5. **Reduced Validation Batches**

**Before**: 100 batches per validation  
**After**: 50 batches per validation

```yaml
image_embeddings_model:
  training:
    val_batches: 50  # Reduced from 100 for faster validation
```

**Why it helps**:
- Faster validation phase
- Still gives representative metrics

## Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Steps/Epoch | 33,176 | 1,000 | **97% fewer** |
| Time/Epoch | ~40 min | ~3-5 min | **8-13x faster** |
| Samples/Epoch | 2.1M | 256K | More efficient sampling |
| Model Params | 166K | 82K | **50% smaller** |
| Memory Usage | High (OOM) | Moderate | **Fits in memory** |

## Expected Training Time

With optimizations:
- **1 Epoch**: ~3-5 minutes
- **10 Epochs**: ~30-50 minutes
- **100 Epochs** (with early stopping): ~1-3 hours

## Running Optimized Training

```bash
# Use the updated config (automatic)
python -m src.modeling.image_embeddings --streaming

# Or override specific values
python -m src.modeling.image_embeddings --streaming \
  --batch-size 256 \
  --steps-per-epoch 1000 \
  --epochs 50
```

## Further Optimizations (If Still Too Slow)

### Option 1: Even Smaller Model
```yaml
hidden_layers: [64]  # Single hidden layer
```

### Option 2: Fewer Steps
```yaml
steps_per_epoch: 500  # Half the steps
```

### Option 3: Larger Batch Size (If Memory Allows)
```yaml
batch_size: 512  # Double the batch size
learning_rate: 0.004  # Increase proportionally
```

### Option 4: Reduce Training Data
Create a smaller training split:
```yaml
data_split:
  train_ratio: 0.50  # Use only 50% for training
  validation_ratio: 0.25
  test_ratio: 0.25
```

### Option 5: Use Mixed Precision (If GPU Available)
```python
# In train_streaming method, use torch.cuda.amp
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

## Monitoring Training

Watch for these healthy signs:
- ✅ Loss decreasing each epoch
- ✅ Validation MAE improving
- ✅ Training completes epochs without OOM
- ✅ Each epoch takes 3-10 minutes

## Troubleshooting

### Still Getting OOM?
1. Reduce batch_size to 128 or 64
2. Reduce shuffle_buffer_size to 1000
3. Reduce val_batches to 25
4. Close other applications

### Training Too Slow?
1. Increase batch_size to 512 (if memory allows)
2. Reduce steps_per_epoch to 500
3. Use GPU if available (see APPLE_SILICON_GPU_SETUP.md)

### Model Not Learning?
1. Increase steps_per_epoch to 2000
2. Decrease learning_rate to 0.001
3. Add model complexity back: `hidden_layers: [256, 128]`

## Theory: Why Fewer Steps Works

**Key Insight**: With 2.1M training samples, you don't need to see them all each epoch!

- Random sampling with 256K samples/epoch (1000 steps × 256 batch) gives ~12% coverage
- Each epoch sees different samples due to shuffling
- Over 10 epochs, you'll see most unique samples
- Much faster than processing all samples sequentially

This is similar to **Stochastic Gradient Descent** - we approximate the full gradient using random samples.

## Recommended Settings by Use Case

### Quick Experimentation
```yaml
steps_per_epoch: 500
batch_size: 256
hidden_layers: [64]
max_epochs: 20
```

### Production Training
```yaml
steps_per_epoch: 2000
batch_size: 128
hidden_layers: [128, 64]
max_epochs: 100
```

### Final Model
```yaml
steps_per_epoch: 5000
batch_size: 64
hidden_layers: [256, 128]
max_epochs: 200
```
