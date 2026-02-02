# =============================================================================
# Auction Price Prediction - Image Embeddings Model
# =============================================================================
"""
Image embeddings-based price prediction model.

This module implements a neural network model that takes pre-computed image
embeddings (576-dimensional from MobileNetV3) and predicts auction item prices.

Architecture:
- Input: 576-dim image embeddings
- Hidden layers: 2-3 fully connected layers with BatchNorm, ReLU, Dropout
- Output: Single value (predicted log1p(price))
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.config import settings
from src.modeling.ml_config import (
    get_inverse_transform,
    get_target_transform,
    load_ml_config,
)

# =============================================================================
# Dataset
# =============================================================================


class ImageEmbeddingsDataset(Dataset):
    """
    PyTorch Dataset for image embeddings and prices.

    Loads embeddings from a numpy array or pandas DataFrame and pairs them
    with corresponding price targets.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        transform_fn: Callable | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            embeddings: Array of image embeddings (N x 576)
            prices: Array of prices (N,)
            transform_fn: Optional function to transform prices (e.g., log1p)
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Apply transformation to prices if specified
        if transform_fn is not None:
            prices = transform_fn(prices)

        self.prices = torch.tensor(prices, dtype=torch.float32).unsqueeze(1)

        assert len(self.embeddings) == len(
            self.prices
        ), f"Embeddings ({len(self.embeddings)}) and prices ({len(self.prices)}) must have same length"

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.prices[idx]


class StreamingEmbeddingsDataset(torch.utils.data.IterableDataset):
    """
    Streaming PyTorch Dataset that loads embeddings in batches from HuggingFace.

    This avoids loading the entire embeddings dataset into memory by streaming
    batches directly from HuggingFace and joining with pre-loaded item prices.
    """

    def __init__(
        self,
        embeddings_repo: str,
        item_prices: dict[int, float],
        embedding_col: str = "image_embedding",
        item_id_col: str = "item_id",
        transform_fn: Callable | None = None,
        shuffle_buffer_size: int = 10000,
    ):
        """
        Initialize the streaming dataset.

        Args:
            embeddings_repo: HuggingFace repository ID for embeddings dataset
            item_prices: Dictionary mapping item_id to price
            embedding_col: Column name for embeddings in the dataset
            item_id_col: Column name for item IDs
            transform_fn: Optional function to transform prices (e.g., log1p)
            shuffle_buffer_size: Size of shuffle buffer for streaming
        """
        self.embeddings_repo = embeddings_repo
        self.item_prices = item_prices
        self.embedding_col = embedding_col
        self.item_id_col = item_id_col
        self.transform_fn = transform_fn
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        from datasets import load_dataset

        # Load dataset in streaming mode
        dataset = load_dataset(self.embeddings_repo, split="train", streaming=True)

        # Shuffle the stream
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        for sample in dataset:
            item_id = sample[self.item_id_col]

            # Skip if item_id not in prices (e.g., filtered out)
            if item_id not in self.item_prices:
                continue

            embedding = np.array(sample[self.embedding_col], dtype=np.float32)
            price = self.item_prices[item_id]

            # Apply transformation
            if self.transform_fn is not None:
                price = self.transform_fn(np.array([price]))[0]

            yield (
                torch.tensor(embedding, dtype=torch.float32),
                torch.tensor([price], dtype=torch.float32),
            )


# =============================================================================
# Model Architecture
# =============================================================================


class ImageEmbeddingsModel(nn.Module):
    """
    Neural network for predicting prices from image embeddings.

    Architecture:
    - Input layer: 576 dimensions (MobileNetV3 embedding size)
    - Hidden layers: Configurable (default: [256, 128])
      Each hidden layer consists of:
      - Linear transformation
      - Batch Normalization (optional)
      - ReLU activation
      - Dropout
    - Output layer: 1 dimension (predicted price)

    Design rationale:
    - BatchNorm helps with training stability and faster convergence
    - Dropout (0.3) provides regularization to prevent overfitting
    - Decreasing layer sizes (576 -> 256 -> 128 -> 1) gradually compress
      the representation while extracting price-relevant features
    """

    def __init__(
        self,
        input_dim: int = 576,
        hidden_layers: list[int] | None = None,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize the model.

        Args:
            input_dim: Input embedding dimension (576 for MobileNetV3)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (1 for regression)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ("relu", "leaky_relu", "gelu")
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 128]
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Select activation function
        activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
        }
        self.activation = activation_map.get(activation, nn.ReLU())

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings (batch_size, input_dim)

        Returns:
            Predicted prices (batch_size, 1)
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Metrics
# =============================================================================


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "val_", "test_")

    Returns:
        Dictionary of metric name -> value
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {}

    # MAE - Mean Absolute Error
    metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)

    # MSE - Mean Squared Error
    metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred)

    # RMSE - Root Mean Squared Error
    metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])

    # R² - Coefficient of determination
    metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)

    # MAPE - Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
            * 100
        )
        metrics[f"{prefix}mape"] = mape

    return metrics


# =============================================================================
# Trainer
# =============================================================================


class ImageEmbeddingsTrainer:
    """
    Trainer for the image embeddings price prediction model.

    Handles:
    - Data loading from Hugging Face datasets
    - Train/validation/test splitting
    - Training loop with early stopping
    - Metric computation on both transformed and original scales
    - Model checkpointing
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: ML configuration dictionary (loaded from YAML if None)
            output_dir: Directory for saving models and logs
        """
        self.config = config or load_ml_config()
        self.output_dir = output_dir or settings.models_dir / "image_embeddings"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract relevant config sections
        self.model_config = self.config.get("image_embeddings_model", {})
        self.training_config = self.config.get("training", {})
        self.target_config = self.config.get("target", {})
        self.split_config = self.config.get("data_split", {})
        self.metrics_config = self.config.get("metrics", {})

        # Get target transformation functions
        self.transform_fn = get_target_transform(
            self.target_config.get("transformation", "log1p")
        )
        self.inverse_transform_fn = get_inverse_transform(
            self.target_config.get("transformation", "log1p")
        )

        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(self.training_config.get("device", "cpu"))

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_mae": []}

    def load_data_from_huggingface(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load embeddings and prices from Hugging Face datasets.

        Note: This loads both datasets fully into memory. For large datasets,
        use load_items_and_create_streaming_pipeline() instead.

        Returns:
            Tuple of (embeddings_df, items_df)
        """
        from datasets import load_dataset

        hf_config = self.config.get("huggingface", {})

        # Load image embeddings dataset
        embeddings_config = hf_config.get("image_embeddings_dataset", {})
        embeddings_repo = embeddings_config.get(
            "repo_id", "jpearce610/image_embeddings"
        )

        logger.info(f"Loading embeddings from {embeddings_repo}...")
        embeddings_ds = load_dataset(embeddings_repo, split="train")
        embeddings_df = embeddings_ds.to_pandas()

        # Load item data dataset (contains prices)
        items_config = hf_config.get("item_data_dataset", {})
        items_repo = items_config.get("repo_id", "jpearce610/item_data")

        logger.info(f"Loading item data from {items_repo}...")
        items_ds = load_dataset(items_repo, split="train")
        items_df = items_ds.to_pandas()

        logger.info(f"Loaded {len(embeddings_df)} embeddings and {len(items_df)} items")

        return embeddings_df, items_df

    def load_items_dataset(self) -> pd.DataFrame:
        """
        Load only the items dataset (contains prices).

        This is memory efficient as the items dataset is much smaller
        than the embeddings dataset.

        Returns:
            DataFrame with item data including prices
        """
        from datasets import load_dataset

        hf_config = self.config.get("huggingface", {})
        items_config = hf_config.get("item_data_dataset", {})
        items_repo = items_config.get("repo_id", "jpearce610/item_data")

        logger.info(f"Loading item data from {items_repo}...")
        items_ds = load_dataset(items_repo, split="train")
        items_df = items_ds.to_pandas()

        logger.info(f"Loaded {len(items_df)} items")
        return items_df

    def create_item_price_lookup(
        self,
        items_df: pd.DataFrame,
        valid_item_ids: set[int] | None = None,
    ) -> dict[int, float]:
        """
        Create a lookup dictionary mapping item_id to price.

        Args:
            items_df: DataFrame with item data
            valid_item_ids: Optional set of item IDs to include (for filtering)

        Returns:
            Dictionary mapping item_id to price
        """
        hf_config = self.config.get("huggingface", {})
        items_config = hf_config.get("item_data_dataset", {})

        item_id_col = items_config.get("item_id_column", "item_id")
        price_col = items_config.get("price_column", "item_current_bid")

        # Handle zero-bid items
        handle_zeros = self.target_config.get("handle_zero_bids", "include")

        price_lookup = {}
        for _, row in items_df.iterrows():
            item_id = row[item_id_col]
            price = row[price_col]

            # Skip if not in valid_item_ids (when filtering)
            if valid_item_ids is not None and item_id not in valid_item_ids:
                continue

            # Handle zero-bid items
            if handle_zeros == "exclude" and price == 0:
                continue

            price_lookup[item_id] = float(price)

        logger.info(f"Created price lookup with {len(price_lookup)} items")
        return price_lookup

    def get_embedding_item_ids(self) -> set[int]:
        """
        Get the set of item IDs that have embeddings by streaming through the dataset.

        This avoids loading the full embeddings dataset into memory.

        Returns:
            Set of item IDs that have embeddings
        """
        from datasets import load_dataset

        hf_config = self.config.get("huggingface", {})
        embeddings_config = hf_config.get("image_embeddings_dataset", {})
        embeddings_repo = embeddings_config.get(
            "repo_id", "jpearce610/image_embeddings"
        )
        item_id_col = embeddings_config.get("item_id_column", "item_id")

        logger.info(f"Scanning embedding item IDs from {embeddings_repo}...")

        # Stream through to get item IDs without loading embeddings
        dataset = load_dataset(embeddings_repo, split="train", streaming=True)

        item_ids = set()
        for sample in dataset:
            item_ids.add(sample[item_id_col])

        logger.info(f"Found {len(item_ids)} unique item IDs with embeddings")
        return item_ids

    def create_streaming_dataloader(
        self,
        item_prices: dict[int, float],
        batch_size: int = 64,
        shuffle_buffer_size: int = 10000,
    ) -> DataLoader:
        """
        Create a streaming DataLoader that loads embeddings in batches from HuggingFace.

        Args:
            item_prices: Dictionary mapping item_id to price
            batch_size: Batch size for training
            shuffle_buffer_size: Size of shuffle buffer for streaming

        Returns:
            DataLoader that streams embeddings from HuggingFace
        """
        hf_config = self.config.get("huggingface", {})
        embeddings_config = hf_config.get("image_embeddings_dataset", {})
        embeddings_repo = embeddings_config.get(
            "repo_id", "jpearce610/image_embeddings"
        )
        embedding_col = embeddings_config.get("embedding_column", "image_embedding")
        item_id_col = embeddings_config.get("item_id_column", "item_id")

        dataset = StreamingEmbeddingsDataset(
            embeddings_repo=embeddings_repo,
            item_prices=item_prices,
            embedding_col=embedding_col,
            item_id_col=item_id_col,
            transform_fn=self.transform_fn,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # Streaming datasets don't support multi-processing
        )

    def train_streaming(
        self,
        train_prices: dict[int, float],
        val_prices: dict[int, float],
        steps_per_epoch: int | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the model using streaming data loading.

        This method loads embeddings in batches from HuggingFace instead of
        loading the entire dataset into memory.

        Args:
            train_prices: Dictionary mapping item_id to price for training
            val_prices: Dictionary mapping item_id to price for validation
            steps_per_epoch: Number of steps per epoch (if None, streams until exhausted)

        Returns:
            Training history dictionary
        """
        model_training_config = self.model_config.get("training", {})
        batch_size = model_training_config.get(
            "batch_size", self.training_config.get("batch_size", 64)
        )

        # Create streaming dataloaders
        train_loader = self.create_streaming_dataloader(
            train_prices, batch_size=batch_size
        )
        val_loader = self.create_streaming_dataloader(
            val_prices, batch_size=batch_size, shuffle_buffer_size=1000
        )

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Setup optimizer
        learning_rate = model_training_config.get(
            "learning_rate", self.training_config.get("learning_rate", 0.001)
        )
        weight_decay = model_training_config.get("weight_decay", 0.0001)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup learning rate scheduler
        lr_config = self.training_config.get("lr_scheduler", {})
        if lr_config.get("enabled", True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 5),
                min_lr=lr_config.get("min_lr", 1e-6),
            )

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        max_epochs = self.training_config.get("max_epochs", 100)
        early_stopping_config = self.training_config.get("early_stopping", {})
        patience = early_stopping_config.get("patience", 10)
        min_delta = early_stopping_config.get("min_delta", 0.0001)

        # Gradient clipping config
        grad_clip_config = self.training_config.get("gradient_clip", {})
        use_grad_clip = grad_clip_config.get("enabled", True)
        max_grad_norm = grad_clip_config.get("max_norm", 1.0)

        # Steps per epoch (for streaming, we need to limit iterations)
        if steps_per_epoch is None:
            steps_per_epoch = len(train_prices) // batch_size

        logger.info(f"Starting streaming training for {max_epochs} epochs...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping patience: {patience}")

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            # Create fresh dataloader for each epoch
            train_loader = self.create_streaming_dataloader(
                train_prices, batch_size=batch_size
            )

            for batch_embeddings, batch_prices in train_loader:
                if train_steps >= steps_per_epoch:
                    break

                batch_embeddings = batch_embeddings.to(self.device)
                batch_prices = batch_prices.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_embeddings)
                loss = criterion(outputs, batch_prices)
                loss.backward()

                # Gradient clipping
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                self.optimizer.step()
                train_loss += loss.item()
                train_steps += 1

            if train_steps > 0:
                train_loss /= train_steps

            # Validation phase
            val_loss, val_metrics = self._evaluate_streaming(val_loader, criterion)

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(
                val_metrics.get("val_mae_original", val_metrics.get("val_mae", 0))
            )

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val MAE (original): ${val_metrics.get('val_mae_original', 0):.2f} - "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping check
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save(self.output_dir / "best_model.pt")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Streaming training complete!")
        return self.history

    def _evaluate_streaming(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        max_batches: int = 100,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate the model on streaming data.

        Args:
            data_loader: Streaming DataLoader for evaluation
            criterion: Loss function
            max_batches: Maximum number of batches to evaluate

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch_embeddings, batch_prices in data_loader:
                if num_batches >= max_batches:
                    break

                batch_embeddings = batch_embeddings.to(self.device)
                batch_prices = batch_prices.to(self.device)

                outputs = self.model(batch_embeddings)
                loss = criterion(outputs, batch_prices)
                total_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_prices.cpu().numpy())
                num_batches += 1

        if num_batches == 0:
            return 0.0, {}

        total_loss /= num_batches
        predictions_transformed = np.concatenate(all_predictions).flatten()
        y_true_transformed = np.concatenate(all_targets).flatten()

        # Compute metrics on transformed scale
        metrics = compute_regression_metrics(
            y_true_transformed, predictions_transformed, prefix="val_"
        )

        # Compute metrics on original scale
        if self.metrics_config.get("compute_original_scale_metrics", True):
            predictions_original = self.inverse_transform_fn(predictions_transformed)
            y_original = self.inverse_transform_fn(y_true_transformed)
            original_metrics = compute_regression_metrics(
                y_original, predictions_original, prefix="val_"
            )

            # Add original scale metrics with "_original" suffix
            for key, value in original_metrics.items():
                metrics[f"{key}_original"] = value

        return total_loss, metrics

    def prepare_data(
        self,
        embeddings_df: pd.DataFrame,
        items_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data for training.

        Args:
            embeddings_df: DataFrame with embeddings
            items_df: DataFrame with prices

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        hf_config = self.config.get("huggingface", {})
        embeddings_config = hf_config.get("image_embeddings_dataset", {})
        items_config = hf_config.get("item_data_dataset", {})

        # Get column names - defaults match actual HuggingFace dataset structure
        embedding_col = embeddings_config.get("embedding_column", "image_embedding")
        item_id_col = embeddings_config.get("item_id_column", "item_id")
        price_col = items_config.get("price_column", "item_current_bid")
        items_id_col = items_config.get("item_id_column", "item_id")

        # Merge on item_id
        logger.info("Merging embeddings with prices...")

        # Ensure item_id columns match
        embeddings_df = embeddings_df.rename(columns={item_id_col: "item_id"})
        items_df = items_df.rename(columns={items_id_col: "item_id"})

        merged_df = embeddings_df.merge(
            items_df[["item_id", price_col]], on="item_id", how="inner"
        )

        logger.info(f"Merged data has {len(merged_df)} samples")

        # Extract embeddings and prices
        embeddings = np.stack(merged_df[embedding_col].values)
        prices = merged_df[price_col].values.astype(np.float32)

        # Handle zero-bid items
        handle_zeros = self.target_config.get("handle_zero_bids", "include")
        if handle_zeros == "exclude":
            mask = prices > 0
            embeddings = embeddings[mask]
            prices = prices[mask]
            logger.info(f"Excluded zero-bid items, remaining: {len(prices)}")

        # Get split ratios
        train_ratio = self.split_config.get("train_ratio", 0.70)
        val_ratio = self.split_config.get("validation_ratio", 0.15)
        test_ratio = self.split_config.get("test_ratio", 0.15)
        random_state = self.split_config.get("random_state", 42)

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            embeddings,
            prices,
            train_size=train_ratio,
            random_state=random_state,
        )

        # Second split: val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_ratio,
            random_state=random_state,
        )

        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self) -> ImageEmbeddingsModel:
        """Build the model from configuration."""
        model = ImageEmbeddingsModel(
            input_dim=self.model_config.get("input_dim", 576),
            hidden_layers=self.model_config.get("hidden_layers", [256, 128]),
            output_dim=self.model_config.get("output_dim", 1),
            dropout_rate=self.model_config.get("dropout_rate", 0.3),
            use_batch_norm=self.model_config.get("use_batch_norm", True),
            activation=self.model_config.get("activation", "relu"),
        )

        logger.info("Model architecture:")
        logger.info(f"  Input dim: {model.input_dim}")
        logger.info(f"  Hidden layers: {model.hidden_layers}")
        logger.info(f"  Dropout rate: {model.dropout_rate}")
        logger.info(f"  Use batch norm: {model.use_batch_norm}")
        logger.info(f"  Total parameters: {model.count_parameters():,}")

        return model.to(self.device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Args:
            X_train: Training embeddings
            y_train: Training prices (original scale)
            X_val: Validation embeddings
            y_val: Validation prices (original scale)

        Returns:
            Training history dictionary
        """
        # Create datasets and dataloaders
        model_training_config = self.model_config.get("training", {})
        batch_size = model_training_config.get(
            "batch_size", self.training_config.get("batch_size", 64)
        )

        train_dataset = ImageEmbeddingsDataset(
            X_train, y_train, transform_fn=self.transform_fn
        )
        val_dataset = ImageEmbeddingsDataset(
            X_val, y_val, transform_fn=self.transform_fn
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.training_config.get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
        )

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Setup optimizer
        learning_rate = model_training_config.get(
            "learning_rate", self.training_config.get("learning_rate", 0.001)
        )
        weight_decay = model_training_config.get("weight_decay", 0.0001)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup learning rate scheduler
        lr_config = self.training_config.get("lr_scheduler", {})
        if lr_config.get("enabled", True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 5),
                min_lr=lr_config.get("min_lr", 1e-6),
            )

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        max_epochs = self.training_config.get("max_epochs", 100)
        early_stopping_config = self.training_config.get("early_stopping", {})
        patience = early_stopping_config.get("patience", 10)
        min_delta = early_stopping_config.get("min_delta", 0.0001)

        # Gradient clipping config
        grad_clip_config = self.training_config.get("gradient_clip", {})
        use_grad_clip = grad_clip_config.get("enabled", True)
        max_grad_norm = grad_clip_config.get("max_norm", 1.0)

        logger.info(f"Starting training for {max_epochs} epochs...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping patience: {patience}")

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_embeddings, batch_prices in train_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_prices = batch_prices.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_embeddings)
                loss = criterion(outputs, batch_prices)
                loss.backward()

                # Gradient clipping
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            val_loss, val_metrics = self._evaluate(val_loader, criterion, y_val)

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(
                val_metrics.get("val_mae_original", val_metrics.get("val_mae", 0))
            )

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val MAE (original): ${val_metrics.get('val_mae_original', 0):.2f} - "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping check
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save(self.output_dir / "best_model.pt")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info("Training complete!")
        return self.history

    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        y_original: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            criterion: Loss function
            y_original: Original (untransformed) prices for metric computation

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []

        with torch.no_grad():
            for batch_embeddings, batch_prices in data_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_prices = batch_prices.to(self.device)

                outputs = self.model(batch_embeddings)
                loss = criterion(outputs, batch_prices)
                total_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())

        total_loss /= len(data_loader)
        predictions_transformed = np.concatenate(all_predictions).flatten()

        # Compute metrics on transformed scale
        y_true_transformed = self.transform_fn(y_original)
        metrics = compute_regression_metrics(
            y_true_transformed, predictions_transformed, prefix="val_"
        )

        # Compute metrics on original scale
        if self.metrics_config.get("compute_original_scale_metrics", True):
            predictions_original = self.inverse_transform_fn(predictions_transformed)
            original_metrics = compute_regression_metrics(
                y_original, predictions_original, prefix="val_"
            )

            # Add original scale metrics with "_original" suffix
            for key, value in original_metrics.items():
                metrics[f"{key}_original"] = value

        return total_loss, metrics

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate the model on test set.

        Args:
            X_test: Test embeddings
            y_test: Test prices (original scale)

        Returns:
            Dictionary of evaluation metrics
        """
        test_dataset = ImageEmbeddingsDataset(
            X_test, y_test, transform_fn=self.transform_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config.get("training", {}).get("batch_size", 64),
            shuffle=False,
        )

        criterion = nn.MSELoss()
        test_loss, test_metrics = self._evaluate(test_loader, criterion, y_test)

        # Rename metrics to have "test_" prefix
        test_metrics = {k.replace("val_", "test_"): v for k, v in test_metrics.items()}
        test_metrics["test_loss"] = test_loss

        logger.info("Test set evaluation:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test MAE (transformed): ${test_metrics.get('test_mae', 0):.4f}")
        logger.info(
            f"  Test MAE (original): ${test_metrics.get('test_mae_original', 0):.2f}"
        )
        logger.info(
            f"  Test RMSE (original): ${test_metrics.get('test_rmse_original', 0):.2f}"
        )
        logger.info(
            f"  Test R² (original): {test_metrics.get('test_r2_original', 0):.4f}"
        )

        return test_metrics

    def predict(
        self, embeddings: np.ndarray, return_original_scale: bool = True
    ) -> np.ndarray:
        """
        Make predictions on new embeddings.

        Args:
            embeddings: Input embeddings (N x 576)
            return_original_scale: Whether to inverse transform predictions

        Returns:
            Predicted prices

        Raises:
            RuntimeError: If the model has not been built or loaded yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built or loaded. "
                "Call build_model() or load() first."
            )

        self.model.eval()

        with torch.no_grad():
            inputs = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            predictions = self.model(inputs).cpu().numpy().flatten()

        if return_original_scale:
            predictions = self.inverse_transform_fn(predictions)

        return predictions

    def save(self, path: Path | None = None) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save the model (uses default if None)
        """
        save_path = path or self.output_dir / "model.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "config": self.config,
            "model_config": self.model_config,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, path: Path | None = None) -> None:
        """
        Load model from checkpoint.

        Args:
            path: Path to load the model from
        """
        load_path = path or self.output_dir / "best_model.pt"

        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # Rebuild model if needed
        if self.model is None:
            self.model = self.build_model()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        logger.info(f"Model loaded from {load_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for training the image embeddings model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train image embeddings price prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ML config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate a trained model (no training)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming data loading (memory efficient for large datasets)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Number of training steps per epoch (for streaming mode)",
    )

    args = parser.parse_args()

    # Load config
    config = load_ml_config(args.config)

    # Override config with CLI args
    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["image_embeddings_model"]["training"]["batch_size"] = args.batch_size

    # Initialize trainer
    output_dir = Path(args.output_dir) if args.output_dir else None
    trainer = ImageEmbeddingsTrainer(config=config, output_dir=output_dir)

    if args.evaluate_only:
        # Load data and evaluate
        embeddings_df, items_df = trainer.load_data_from_huggingface()
        _, _, X_test, _, _, y_test = trainer.prepare_data(embeddings_df, items_df)
        trainer.load()
        metrics = trainer.evaluate(X_test, y_test)
        print(f"Test metrics: {metrics}")
    elif args.streaming:
        # Streaming training pipeline (memory efficient)
        print("Using streaming data loading (memory efficient mode)...")

        # Step 1: Load items dataset (smaller, contains prices)
        items_df = trainer.load_items_dataset()

        # Step 2: Get item IDs that have embeddings (streaming scan)
        embedding_item_ids = trainer.get_embedding_item_ids()

        # Step 3: Create price lookup for items with embeddings
        all_prices = trainer.create_item_price_lookup(
            items_df, valid_item_ids=embedding_item_ids
        )

        # Step 4: Split item IDs into train/val/test
        from sklearn.model_selection import train_test_split

        item_ids = list(all_prices.keys())
        train_ratio = config.get("data_split", {}).get("train_ratio", 0.70)
        val_ratio = config.get("data_split", {}).get("validation_ratio", 0.15)
        test_ratio = config.get("data_split", {}).get("test_ratio", 0.15)
        random_state = config.get("data_split", {}).get("random_state", 42)

        # First split: train vs (val + test)
        train_ids, temp_ids = train_test_split(
            item_ids, train_size=train_ratio, random_state=random_state
        )

        # Second split: val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=relative_test_ratio, random_state=random_state
        )

        train_prices = {item_id: all_prices[item_id] for item_id in train_ids}
        val_prices = {item_id: all_prices[item_id] for item_id in val_ids}
        test_prices = {item_id: all_prices[item_id] for item_id in test_ids}

        logger.info(
            f"Data split: train={len(train_prices)}, "
            f"val={len(val_prices)}, test={len(test_prices)}"
        )

        # Step 5: Train using streaming
        trainer.train_streaming(
            train_prices, val_prices, steps_per_epoch=args.steps_per_epoch
        )

        # Load best model
        trainer.load()

        print("\n" + "=" * 50)
        print("STREAMING TRAINING COMPLETE")
        print("=" * 50)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Training samples: {len(train_prices)}")
        print(f"Validation samples: {len(val_prices)}")
        print(f"Test samples: {len(test_prices)}")
    else:
        # Full training pipeline (loads all data into memory)
        embeddings_df, items_df = trainer.load_data_from_huggingface()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
            embeddings_df, items_df
        )

        # Train
        trainer.train(X_train, y_train, X_val, y_val)

        # Load best model and evaluate on test set
        trainer.load()
        test_metrics = trainer.evaluate(X_test, y_test)

        print("\n" + "=" * 50)
        print("TRAINING COMPLETE")
        print("=" * 50)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(
            f"Test MAE (original scale): ${test_metrics.get('test_mae_original', 0):.2f}"
        )
        print(
            f"Test RMSE (original scale): ${test_metrics.get('test_rmse_original', 0):.2f}"
        )
        print(
            f"Test R² (original scale): {test_metrics.get('test_r2_original', 0):.4f}"
        )


if __name__ == "__main__":
    main()
