"""
Data loading and batch generation for SmolGPT language modeling on TinyStories.
"""

import os
from typing import List, Tuple
import numpy as np
import torch


def load_tinystories_dataset(
    npz_path: str = "data/tinystories/tinystories_sub.npz"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads pre-tokenized train and validation token arrays.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset token file not found at {npz_path}.")
    data = np.load(npz_path)
    return data["train"], data["val"]


def get_random_batch(
    tokens: np.ndarray,
    batch_size: int = 16,
    block_size: int = 128,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random training batch (x, y) where y is the next-token target.
    """
    max_idx = len(tokens) - block_size - 1
    indices = np.random.randint(0, max_idx, size=batch_size)
    x_np = np.stack([tokens[i : i + block_size].astype(np.int64) for i in indices])
    y_np = np.stack([tokens[i + 1 : i + block_size + 1].astype(np.int64) for i in indices])
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)


def get_deterministic_val_batches(
    val_tokens: np.ndarray,
    num_batches: int = 20,
    batch_size: int = 16,
    block_size: int = 128,
    device: str = "cuda"
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts fixed, non-overlapping evaluation chunks for deterministic validation metrics.
    Total evaluated tokens = num_batches * batch_size * block_size (default: 20 * 16 * 128 = 40,960 tokens).
    """
    batches = []
    chunk_len = batch_size * block_size
    assert len(val_tokens) >= (num_batches + 1) * chunk_len, "Validation split too small for requested batches."
    
    for b in range(num_batches):
        start = b * chunk_len
        indices = [start + i * block_size for i in range(batch_size)]
        x_np = np.stack([val_tokens[i : i + block_size].astype(np.int64) for i in indices])
        y_np = np.stack([val_tokens[i + 1 : i + block_size + 1].astype(np.int64) for i in indices])
        batches.append((torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)))
    return batches
