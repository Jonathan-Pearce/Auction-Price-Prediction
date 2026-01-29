# =============================================================================
# Tests for Image Embeddings Model
# =============================================================================
"""
Tests for the image embeddings price prediction model.
"""

import numpy as np
import pytest
import torch

from src.modeling.image_embeddings import (
    ImageEmbeddingsDataset,
    ImageEmbeddingsModel,
    ImageEmbeddingsTrainer,
    compute_regression_metrics,
)
from src.modeling.ml_config import (
    get_inverse_transform,
    get_target_transform,
    load_ml_config,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_embeddings():
    """Generate sample image embeddings."""
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 576
    return np.random.randn(n_samples, embedding_dim).astype(np.float32)


@pytest.fixture
def sample_prices():
    """Generate sample prices with realistic distribution."""
    np.random.seed(42)
    n_samples = 100
    # Log-normal distribution to simulate realistic price distribution
    prices = np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples).astype(np.float32)
    # Add some zero-bid items
    prices[:5] = 0.0
    return prices


@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        "input_dim": 576,
        "hidden_layers": [256, 128],
        "output_dim": 1,
        "dropout_rate": 0.3,
        "use_batch_norm": True,
        "activation": "relu",
    }


# =============================================================================
# ML Config Tests
# =============================================================================


def test_load_ml_config():
    """Test loading ML configuration from YAML."""
    config = load_ml_config()

    assert config is not None
    assert "target" in config
    assert "data_split" in config
    assert "metrics" in config
    assert "training" in config
    assert "image_embeddings_model" in config


def test_target_transform_log1p():
    """Test log1p transformation."""
    transform_fn = get_target_transform("log1p")
    inverse_fn = get_inverse_transform("log1p")

    values = np.array([0.0, 1.0, 10.0, 100.0])
    transformed = transform_fn(values)
    recovered = inverse_fn(transformed)

    np.testing.assert_array_almost_equal(values, recovered, decimal=5)


def test_target_transform_none():
    """Test no transformation."""
    transform_fn = get_target_transform("none")
    inverse_fn = get_inverse_transform("none")

    values = np.array([0.0, 1.0, 10.0, 100.0])
    transformed = transform_fn(values)
    recovered = inverse_fn(transformed)

    np.testing.assert_array_equal(values, transformed)
    np.testing.assert_array_equal(values, recovered)


def test_data_split_config():
    """Test data split configuration values."""
    config = load_ml_config()
    split_config = config.get("data_split", {})

    train_ratio = split_config.get("train_ratio", 0.7)
    val_ratio = split_config.get("validation_ratio", 0.15)
    test_ratio = split_config.get("test_ratio", 0.15)

    # Ratios should sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6


# =============================================================================
# Dataset Tests
# =============================================================================


def test_image_embeddings_dataset_creation(sample_embeddings, sample_prices):
    """Test creating the dataset."""
    dataset = ImageEmbeddingsDataset(sample_embeddings, sample_prices)

    assert len(dataset) == len(sample_embeddings)
    assert len(dataset) == len(sample_prices)


def test_image_embeddings_dataset_getitem(sample_embeddings, sample_prices):
    """Test getting items from the dataset."""
    dataset = ImageEmbeddingsDataset(sample_embeddings, sample_prices)

    embedding, price = dataset[0]

    assert isinstance(embedding, torch.Tensor)
    assert isinstance(price, torch.Tensor)
    assert embedding.shape == (576,)
    assert price.shape == (1,)


def test_image_embeddings_dataset_with_transform(sample_embeddings, sample_prices):
    """Test dataset with price transformation."""
    transform_fn = get_target_transform("log1p")
    dataset = ImageEmbeddingsDataset(
        sample_embeddings, sample_prices, transform_fn=transform_fn
    )

    _, price = dataset[10]
    expected_price = np.log1p(sample_prices[10])

    np.testing.assert_almost_equal(price.item(), expected_price, decimal=5)


# =============================================================================
# Model Architecture Tests
# =============================================================================


def test_image_embeddings_model_creation(model_config):
    """Test creating the model."""
    model = ImageEmbeddingsModel(**model_config)

    assert model is not None
    assert model.input_dim == 576
    assert model.hidden_layers == [256, 128]


def test_image_embeddings_model_forward_pass(model_config):
    """Test forward pass through the model."""
    model = ImageEmbeddingsModel(**model_config)

    batch_size = 16
    input_tensor = torch.randn(batch_size, 576)

    output = model(input_tensor)

    assert output.shape == (batch_size, 1)


def test_image_embeddings_model_parameter_count(model_config):
    """Test counting model parameters."""
    model = ImageEmbeddingsModel(**model_config)

    num_params = model.count_parameters()

    # Expected approximate count:
    # Layer 1: 576 * 256 + 256 (bias) + 256 * 2 (batch norm) = ~148,224
    # Layer 2: 256 * 128 + 128 (bias) + 128 * 2 (batch norm) = ~33,152
    # Output: 128 * 1 + 1 = 129
    # Total: ~181,505
    assert num_params > 100000  # Should have significant parameters
    assert num_params < 500000  # But not too many


def test_image_embeddings_model_different_architectures():
    """Test different model architectures."""
    # 2-layer model
    model_2layer = ImageEmbeddingsModel(hidden_layers=[128, 64])
    linear_layers_2 = [
        layer for layer in model_2layer.network if isinstance(layer, torch.nn.Linear)
    ]
    assert len(linear_layers_2) == 3

    # 3-layer model
    model_3layer = ImageEmbeddingsModel(hidden_layers=[256, 128, 64])
    linear_layers_3 = [
        layer for layer in model_3layer.network if isinstance(layer, torch.nn.Linear)
    ]
    assert len(linear_layers_3) == 4


def test_image_embeddings_model_no_batch_norm():
    """Test model without batch normalization."""
    model = ImageEmbeddingsModel(use_batch_norm=False)

    # Check no BatchNorm layers
    batch_norm_layers = [
        layer for layer in model.network if isinstance(layer, torch.nn.BatchNorm1d)
    ]
    assert len(batch_norm_layers) == 0


def test_image_embeddings_model_different_activations():
    """Test different activation functions."""
    for activation in ["relu", "leaky_relu", "gelu"]:
        model = ImageEmbeddingsModel(activation=activation)

        # Model should be created without error
        input_tensor = torch.randn(8, 576)
        output = model(input_tensor)

        assert output.shape == (8, 1)


# =============================================================================
# Metrics Tests
# =============================================================================


def test_compute_regression_metrics():
    """Test computing regression metrics."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0, 52.0])

    metrics = compute_regression_metrics(y_true, y_pred, prefix="test_")

    assert "test_mae" in metrics
    assert "test_mse" in metrics
    assert "test_rmse" in metrics
    assert "test_r2" in metrics

    # Check that metrics are reasonable
    assert metrics["test_mae"] > 0
    assert metrics["test_rmse"] >= metrics["test_mae"]
    assert metrics["test_r2"] > 0  # Good predictions should have positive R²


def test_compute_regression_metrics_perfect_predictions():
    """Test metrics with perfect predictions."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = y_true.copy()

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_compute_regression_metrics_with_zeros():
    """Test metrics computation with zero values."""
    y_true = np.array([0.0, 10.0, 20.0, 30.0])
    y_pred = np.array([5.0, 12.0, 18.0, 32.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    # MAPE should be computed only for non-zero values
    assert "mape" in metrics
    assert metrics["mape"] > 0


# =============================================================================
# Trainer Tests
# =============================================================================


def test_trainer_initialization():
    """Test initializing the trainer."""
    trainer = ImageEmbeddingsTrainer()

    assert trainer.config is not None
    assert trainer.output_dir is not None
    assert trainer.transform_fn is not None
    assert trainer.inverse_transform_fn is not None


def test_trainer_build_model():
    """Test building the model through trainer."""
    trainer = ImageEmbeddingsTrainer()
    model = trainer.build_model()

    assert model is not None
    assert isinstance(model, ImageEmbeddingsModel)


def test_trainer_prepare_data_shapes(sample_embeddings, sample_prices):
    """Test data preparation with mock data."""
    # This test uses mock data instead of HuggingFace
    # to avoid network dependency
    from sklearn.model_selection import train_test_split

    # Split the mock data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sample_embeddings, sample_prices, train_size=0.7, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Check shapes are correct
    assert X_train.shape[0] == 70
    assert X_val.shape[0] == 15
    assert X_test.shape[0] == 15

    assert X_train.shape[1] == 576
    assert X_val.shape[1] == 576
    assert X_test.shape[1] == 576


def test_trainer_short_training(sample_embeddings, sample_prices):
    """Test a short training run."""
    # Create a minimal config for quick testing
    config = load_ml_config()
    config["training"]["max_epochs"] = 2
    config["training"]["early_stopping"]["patience"] = 5
    config["image_embeddings_model"]["training"]["batch_size"] = 32

    trainer = ImageEmbeddingsTrainer(config=config)
    trainer.model = trainer.build_model()

    # Split data
    split_idx = int(0.7 * len(sample_embeddings))
    X_train = sample_embeddings[:split_idx]
    y_train = sample_prices[:split_idx]
    X_val = sample_embeddings[split_idx:]
    y_val = sample_prices[split_idx:]

    # Train for a few epochs
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Check history is recorded
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 2


def test_trainer_prediction(sample_embeddings, sample_prices):
    """Test making predictions with trained model."""
    config = load_ml_config()
    config["training"]["max_epochs"] = 1

    trainer = ImageEmbeddingsTrainer(config=config)
    trainer.model = trainer.build_model()

    # Quick training
    split_idx = int(0.8 * len(sample_embeddings))
    trainer.train(
        sample_embeddings[:split_idx],
        sample_prices[:split_idx],
        sample_embeddings[split_idx:],
        sample_prices[split_idx:],
    )

    # Make predictions
    predictions = trainer.predict(sample_embeddings[:10], return_original_scale=True)

    assert len(predictions) == 10
    # Predictions are finite numbers (not NaN or Inf)
    assert all(np.isfinite(p) for p in predictions)


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_training_pipeline_mock(sample_embeddings, sample_prices):
    """Test the full training pipeline with mock data."""
    # Use minimal config
    config = load_ml_config()
    config["training"]["max_epochs"] = 3
    config["training"]["early_stopping"]["patience"] = 10

    trainer = ImageEmbeddingsTrainer(config=config)
    trainer.model = trainer.build_model()

    # Split data
    n = len(sample_embeddings)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train = sample_embeddings[:train_end]
    y_train = sample_prices[:train_end]
    X_val = sample_embeddings[train_end:val_end]
    y_val = sample_prices[train_end:val_end]
    X_test = sample_embeddings[val_end:]
    y_test = sample_prices[val_end:]

    # Train
    history = trainer.train(X_train, y_train, X_val, y_val)

    assert len(history["train_loss"]) == 3

    # Evaluate
    test_metrics = trainer.evaluate(X_test, y_test)

    assert "test_mae" in test_metrics
    assert "test_rmse" in test_metrics
    assert "test_r2" in test_metrics


def test_model_save_and_load(tmp_path, sample_embeddings, sample_prices):
    """Test saving and loading model."""
    config = load_ml_config()
    config["training"]["max_epochs"] = 1

    # Train and save
    trainer = ImageEmbeddingsTrainer(config=config, output_dir=tmp_path)
    trainer.model = trainer.build_model()

    split_idx = int(0.8 * len(sample_embeddings))
    trainer.train(
        sample_embeddings[:split_idx],
        sample_prices[:split_idx],
        sample_embeddings[split_idx:],
        sample_prices[split_idx:],
    )

    save_path = tmp_path / "test_model.pt"
    trainer.save(save_path)

    # Load in new trainer
    new_trainer = ImageEmbeddingsTrainer(config=config, output_dir=tmp_path)
    new_trainer.model = new_trainer.build_model()
    new_trainer.load(save_path)

    # Predictions should match
    test_input = sample_embeddings[:5]
    pred1 = trainer.predict(test_input)
    pred2 = new_trainer.predict(test_input)

    np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
