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
from .models_transformer import (
    SmolGPTConfig,
    SmolGPT,
    evaluate_transformer_loss_and_ppl,
    get_mlp_params,
    set_mlp_params
)
from .data import (
    load_california_housing_splits,
    load_mnist_tensors,
    load_modular_addition_dataset
)
from .data_transformer import (
    load_tinystories_dataset,
    get_random_batch,
    get_deterministic_val_batches,
    get_deterministic_train_batches
)
from .low_rank import LowRankTrajectoryEngine

__all__ = [
    "RegressionMLP",
    "CompactMNISTMLP",
    "SmallCNN",
    "GrokkingTransformer",
    "SmolGPTConfig",
    "SmolGPT",
    "evaluate_transformer_loss_and_ppl",
    "get_flat_params",
    "set_flat_params",
    "eval_regression_mse",
    "eval_mnist_tensors",
    "eval_grokking_tensors",
    "load_california_housing_splits",
    "load_mnist_tensors",
    "load_modular_addition_dataset",
    "load_tinystories_dataset",
    "get_random_batch",
    "get_deterministic_val_batches",
    "LowRankTrajectoryEngine"
]
