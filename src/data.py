import os
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

def load_california_housing_splits(test_size=0.15, val_size=0.15, seed=42):
    """
    Returns strict, leakage-free California Housing dataset splits:
    Train: 70%, Val: 15%, Test: 15%.
    StandardScaler is fitted strictly on Train.
    """
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    indices = np.arange(len(X))

    temp_size = val_size + test_size
    idx_train, idx_temp = train_test_split(indices, test_size=temp_size, random_state=seed)
    val_fraction_of_temp = val_size / temp_size
    idx_val, idx_test = train_test_split(idx_temp, test_size=1.0 - val_fraction_of_temp, random_state=seed)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X[idx_train])
    y_train_scaled = scaler_y.fit_transform(y[idx_train].reshape(-1, 1)).flatten()

    X_val_scaled = scaler_x.transform(X[idx_val])
    y_val_scaled = scaler_y.transform(y[idx_val].reshape(-1, 1)).flatten()

    X_test_scaled = scaler_x.transform(X[idx_test])
    y_test_scaled = scaler_y.transform(y[idx_test].reshape(-1, 1)).flatten()

    return {
        "X_train": X_train_scaled, "y_train": y_train_scaled,
        "X_val": X_val_scaled, "y_val": y_val_scaled,
        "X_test": X_test_scaled, "y_test": y_test_scaled,
        "scaler_x": scaler_x, "scaler_y": scaler_y
    }

def load_mnist_tensors(data_root="./data", seed=42, device="cpu"):
    """
    Returns pre-loaded MNIST tensors on the specified device:
    50,000 train, 10,000 val, 10,000 test.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(seed)
    )
    test_dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)

    val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
    for x_v, y_v in val_loader:
        x_val_t = x_v.to(device)
        y_val_t = y_v.to(device)
        break

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    for x_t, y_t in test_loader:
        x_test_t = x_t.to(device)
        y_test_t = y_t.to(device)
        break

    return {
        "train_subset": train_subset,
        "x_val_t": x_val_t, "y_val_t": y_val_t,
        "x_test_t": x_test_t, "y_test_t": y_test_t
    }

def load_modular_addition_dataset(p=113, frac_train=0.3, seed=42, device="cuda"):
    """
    Creates the exact modular addition dataset from Nanda et al. (2023) / Power et al. (2022).
    - p: prime modulus (default 113)
    - frac_train: fraction of p^2 equation pairs used for training (default 0.3 = 30%)
    - Input sequence: [a, b, =] of length 3, with '=' mapped to token index p.
    - Target label: (a + b) % p
    Returns strictly disjoint train and test tensors directly on the specified device.
    """
    # Generate all pairs (a, b) in [0, p-1] x [0, p-1]
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full((p * p,), p, dtype=torch.long) # '=' token is index p
    dataset = torch.stack([a, b, eq], dim=1) # (p*p, 3)
    labels = (a + b) % p # (p*p,)

    # Random permutation with deterministic seed
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    cutoff = int(p * p * frac_train)

    train_indices = perm[:cutoff]
    test_indices = perm[cutoff:]

    train_data = dataset[train_indices].to(device)
    train_labels = labels[train_indices].to(device)
    test_data = dataset[test_indices].to(device)
    test_labels = labels[test_indices].to(device)

    return {
        "p": p,
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "num_train": len(train_indices),
        "num_test": len(test_indices),
    }
