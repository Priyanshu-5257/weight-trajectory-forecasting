"""
Weight Trajectory Forecasting: Core library modules.
"""

from .models import (
    RegressionMLP,
    CompactMNISTMLP,
    SmallCNN,
    GrokkingTransformer,
    get_flat_params,
    set_flat_params,
    eval_regression_mse,
    eval_mnist_tensors,
    eval_grokking_tensors
)
from .data import (
    load_california_housing_splits,
    load_mnist_tensors,
    load_modular_addition_dataset
)
from .low_rank import LowRankTrajectoryEngine

__all__ = [
    "RegressionMLP",
    "CompactMNISTMLP",
    "SmallCNN",
    "GrokkingTransformer",
    "get_flat_params",
    "set_flat_params",
    "eval_regression_mse",
    "eval_mnist_tensors",
    "eval_grokking_tensors",
    "load_california_housing_splits",
    "load_mnist_tensors",
    "load_modular_addition_dataset",
    "LowRankTrajectoryEngine"
]
