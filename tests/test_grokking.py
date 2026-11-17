import os
import sys
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import GrokkingTransformer, get_flat_params, eval_grokking_tensors
from src.data import load_modular_addition_dataset
from src.low_rank import LowRankTrajectoryEngine

class TestGrokkingComponents:
    def test_modular_addition_dataset_disjointness(self):
        p = 97
        frac_train = 0.5
        seed = 42
        data = load_modular_addition_dataset(p=p, frac_train=frac_train, seed=seed, device="cpu")
        
        train_x = data["train_data"]
        test_x = data["test_data"]
        train_y = data["train_labels"]
        test_y = data["test_labels"]

        # Shape verification
        assert train_x.shape[1] == 3
        assert test_x.shape[1] == 3
        assert len(train_x) + len(test_x) == p * p

        # Check label correctness: (a + b) % p
        expected_train_y = (train_x[:, 0] + train_x[:, 1]) % p
        expected_test_y = (test_x[:, 0] + test_x[:, 1]) % p
        assert torch.equal(train_y, expected_train_y)
        assert torch.equal(test_y, expected_test_y)

        # Check strict disjointness: no pair (a, b) in train should exist in test
        train_pairs = set(zip(train_x[:, 0].tolist(), train_x[:, 1].tolist()))
        test_pairs = set(zip(test_x[:, 0].tolist(), test_x[:, 1].tolist()))
        overlap = train_pairs.intersection(test_pairs)
        assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} overlapping equation pairs."

    def test_grokking_transformer_forward_shape(self):
        p = 97
        model = GrokkingTransformer(p=p, d_model=128, n_heads=4, d_mlp=512)
        x = torch.randint(0, p, (16, 3))
        x[:, 2] = p # '=' token
        logits = model(x)
        assert logits.shape == (16, p)

        # Check parameter count is non-trivial and matches expected range
        params = get_flat_params(model)
        assert 200000 <= params.numel() <= 300000

    def test_grokking_low_rank_engine(self):
        # Synthetic checkpoint trajectory
        P = 227200
        T = 32
        rng = torch.Generator().manual_seed(42)
        w_ref = torch.randn(P, generator=rng)
        
        # 1-D dominant trajectory + small noise
        direction = torch.randn(P, generator=rng)
        direction = direction / torch.norm(direction)
        
        weights = []
        for t in range(T):
            step_w = w_ref + direction * (t * 0.1) + torch.randn(P, generator=rng) * 0.001
            weights.append(step_w)

        W_window = torch.stack(weights, dim=1).numpy()
        engine = LowRankTrajectoryEngine(W_window, ref_checkpoint=weights[-1].numpy())

        diag_1 = engine.get_rank_diagnostics(1)
        diag_2 = engine.get_rank_diagnostics(2)
        
        # Rank 1 should explain >90% of variance for 1-D trajectory
        assert diag_1["explained_variance"] > 0.90
        assert diag_2["explained_variance"] >= diag_1["explained_variance"]
