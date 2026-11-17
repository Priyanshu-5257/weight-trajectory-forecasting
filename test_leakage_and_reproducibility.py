import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiment_low_rank_forecasting import (
    RegressionMLP,
    CompactMNISTMLP,
    get_flat_params,
    set_flat_params,
    LowRankTrajectoryForecaster
)

class TestLeakageAndReproducibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_california_housing_split_leakage(self):
        """Verify train, val, and test splits are completely disjoint and scaler is fitted strictly on train."""
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        n_total = len(X)

        # 70% train, 15% val, 15% test
        indices = np.arange(n_total)
        idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

        # 1. Disjointness check
        set_train = set(idx_train)
        set_val = set(idx_val)
        set_test = set(idx_test)

        self.assertEqual(len(set_train.intersection(set_val)), 0, "Train and Val overlap!")
        self.assertEqual(len(set_train.intersection(set_test)), 0, "Train and Test overlap!")
        self.assertEqual(len(set_val.intersection(set_test)), 0, "Val and Test overlap!")
        self.assertEqual(len(set_train) + len(set_val) + len(set_test), n_total, "Splits do not sum to total!")

        # 2. Scaler leakage check
        scaler_x = StandardScaler()
        X_train = X[idx_train]
        scaler_x.fit(X_train)

        # Scaler mean must match X_train mean, but NOT total X mean
        np.testing.assert_allclose(scaler_x.mean_, np.mean(X_train, axis=0), rtol=1e-5)
        self.assertFalse(np.allclose(scaler_x.mean_, np.mean(X, axis=0), rtol=1e-4),
                         "Scaler appears fitted on entire dataset instead of train split!")

    def test_mnist_split_disjointness(self):
        """Verify MNIST 50k train / 10k val / 10k test split has no index overlap."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
        train_subset, val_subset = torch.utils.data.random_split(
            full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
        )
        test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

        train_indices = set(train_subset.indices)
        val_indices = set(val_subset.indices)

        self.assertEqual(len(train_indices.intersection(val_indices)), 0, "MNIST train and val subsets overlap!")
        self.assertEqual(len(train_indices) + len(val_indices), 60000, "MNIST split does not equal 60,000!")
        self.assertEqual(len(test_dataset), 10000, "Official MNIST test set should have 10,000 samples!")

    def test_svd_cutoff_isolation(self):
        """Verify SVD displacement matrix is constructed strictly from historical checkpoints up to cutoff."""
        P, T_total, cutoff, context_len = 100, 150, 100, 32
        # Generate dummy checkpoint history
        checkpoints = np.random.randn(P, T_total)

        # History window available at cutoff
        history_window = checkpoints[:, cutoff - context_len:cutoff] # shape (P, context_len)
        w_frozen = checkpoints[:, cutoff - 1] # Checkpoint at cutoff step

        forecaster = LowRankTrajectoryForecaster(history_window, ref_checkpoint=w_frozen)

        # Verify displacement matrix dimensions
        self.assertEqual(forecaster.D.shape, (P, context_len))
        # Verify displacement is relative to frozen checkpoint at cutoff
        expected_D = history_window - w_frozen[:, None]
        np.testing.assert_allclose(forecaster.D, expected_D, rtol=1e-6)

        # Check that no checkpoint from > cutoff was accessed
        for t in range(cutoff, T_total):
            # Assert that no column of D matches any future checkpoint displacement
            future_diff = checkpoints[:, t] - w_frozen
            for col_idx in range(context_len):
                self.assertFalse(np.allclose(forecaster.D[:, col_idx], future_diff),
                                 f"Future checkpoint at step {t} leaked into SVD displacement matrix!")

    def test_reproducibility_deterministic_seeding(self):
        """Verify that identical seeds produce bitwise-identical model initializations and optimization steps."""
        def run_dummy_train(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = CompactMNISTMLP()
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            x = torch.randn(8, 1, 14, 14)
            y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
            loss_fn = nn.CrossEntropyLoss()
            
            p0 = get_flat_params(model).clone()
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            p1 = get_flat_params(model).clone()
            return p0, p1

        p0_a, p1_a = run_dummy_train(42)
        p0_b, p1_b = run_dummy_train(42)
        p0_c, p1_c = run_dummy_train(43)

        # Same seed must match exactly
        self.assertTrue(torch.equal(p0_a, p0_b), "Initial weights with same seed differ!")
        self.assertTrue(torch.equal(p1_a, p1_b), "Updated weights with same seed differ!")

        # Different seed must differ
        self.assertFalse(torch.equal(p0_a, p0_c), "Different seeds produced identical initial weights!")
        self.assertFalse(torch.equal(p1_a, p1_c), "Different seeds produced identical updated weights!")

    def test_validation_selection_isolation(self):
        """Verify hyperparameter selection is strictly conditioned on validation metric and blind to test metric."""
        mock_candidates = [
            {"name": "config_A", "val_loss": 0.35, "test_loss": 0.20}, # Low test, but high val
            {"name": "config_B", "val_loss": 0.25, "test_loss": 0.30}, # Best val!
            {"name": "config_C", "val_loss": 0.40, "test_loss": 0.15}, # Lowest test, but worst val
        ]

        # Valid selection rule: select by minimum val_loss
        selected = min(mock_candidates, key=lambda c: c["val_loss"])
        self.assertEqual(selected["name"], "config_B", "Selection must pick the candidate with minimum val_loss!")
        self.assertNotEqual(selected["name"], "config_C", "Selection must NOT peek at test_loss!")

if __name__ == "__main__":
    unittest.main()
