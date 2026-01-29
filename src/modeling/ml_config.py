# =============================================================================
# Auction Price Prediction - ML Config Loader
# =============================================================================
"""
ML configuration management utilities.

Provides functions for loading and accessing the shared ML configuration
YAML file used across all model pillars.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "ml_config.yaml"


def load_ml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load ML configuration from YAML file.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        Configuration dictionary
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"ML config file not found at {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded ML config from {path}")
    return config


def get_target_transform(transformation: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the target transformation function.

    Args:
        transformation: Name of transformation ("log1p", "log", "sqrt", "none")

    Returns:
        Transformation function
    """
    transforms = {
        "log1p": lambda x: np.log1p(x),
        "log": lambda x: np.log(x + 1e-8),  # Add small constant to avoid log(0)
        "sqrt": lambda x: np.sqrt(x),
        "none": lambda x: x,
    }

    if transformation not in transforms:
        logger.warning(f"Unknown transformation '{transformation}', using 'none'")
        return transforms["none"]

    return transforms[transformation]


def get_inverse_transform(transformation: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the inverse transformation function.

    Args:
        transformation: Name of transformation ("log1p", "log", "sqrt", "none")

    Returns:
        Inverse transformation function
    """
    inverse_transforms = {
        "log1p": lambda x: np.expm1(x),  # expm1(x) = exp(x) - 1
        "log": lambda x: np.exp(x) - 1e-8,
        "sqrt": lambda x: np.square(x),
        "none": lambda x: x,
    }

    if transformation not in inverse_transforms:
        logger.warning(f"Unknown transformation '{transformation}', using 'none'")
        return inverse_transforms["none"]

    return inverse_transforms[transformation]


def get_data_split_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get data split configuration.

    Args:
        config: Full config dict (loads from file if None)

    Returns:
        Data split configuration section
    """
    if config is None:
        config = load_ml_config()

    return config.get(
        "data_split",
        {
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
        },
    )


def get_training_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get general training configuration.

    Args:
        config: Full config dict (loads from file if None)

    Returns:
        Training configuration section
    """
    if config is None:
        config = load_ml_config()

    return config.get(
        "training",
        {
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_epochs": 100,
        },
    )


def get_metrics_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get metrics configuration.

    Args:
        config: Full config dict (loads from file if None)

    Returns:
        Metrics configuration section
    """
    if config is None:
        config = load_ml_config()

    return config.get(
        "metrics",
        {
            "primary_metric": "mae",
            "compute_transformed_metrics": True,
            "compute_original_scale_metrics": True,
        },
    )


def get_model_config(
    model_name: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model (e.g., "image_embeddings_model")
        config: Full config dict (loads from file if None)

    Returns:
        Model-specific configuration
    """
    if config is None:
        config = load_ml_config()

    model_config = config.get(model_name, {})

    if not model_config:
        logger.warning(f"No configuration found for model '{model_name}'")

    return model_config
