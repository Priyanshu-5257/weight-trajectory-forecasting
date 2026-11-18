"""
Unit tests for SmolGPT Transformer architecture, data loaders, and low-rank forecasting integration.
"""

import math
import numpy as np
import pytest
import torch

from src.models_transformer import (
    SmolGPTConfig,
    SmolGPT,
    get_flat_params,
    set_flat_params,
    evaluate_transformer_loss_and_ppl
)
from src.data_transformer import (
    get_random_batch,
    get_deterministic_val_batches
)
from src.low_rank import LowRankTrajectoryEngine


class TestSmolGPTComponents:
    @pytest.fixture
    def config(self):
        return SmolGPTConfig(
            block_size=32,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embed=64,
            dropout=0.0
        )

    def test_smolgpt_forward_and_loss(self, config):
        model = SmolGPT(config)
        model.eval()

        # Batch without targets
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        logits, loss = model(idx)
        assert loss is None
        assert logits.shape == (2, 1, config.vocab_size)

        # Batch with targets
        targets = torch.randint(0, config.vocab_size, (2, config.block_size))
        logits, loss = model(idx, targets)
        assert loss is not None
        assert logits.shape == (2, config.block_size, config.vocab_size)
        assert loss.item() > 0.0
        # Loss should be close to random guess: ln(vocab_size) = ln(256) = 5.54
        assert abs(loss.item() - math.log(config.vocab_size)) < 2.0

    def test_smolgpt_flat_parameters_roundtrip(self, config):
        model = SmolGPT(config)
        flat_orig = get_flat_params(model)
        assert flat_orig.dim() == 1
        assert flat_orig.numel() == sum(p.numel() for p in model.parameters())

        # Modify parameters and verify
        perturbed = flat_orig + 0.42
        set_flat_params(model, perturbed)
        flat_check = get_flat_params(model)
        assert torch.allclose(flat_check, perturbed, atol=1e-6)

        # Verify weight tying is preserved
        assert torch.allclose(model.wte.weight, model.lm_head.weight)

    def test_smolgpt_mlp_parameters_isolation(self, config):
        from src.models_transformer import get_mlp_params, set_mlp_params
        model = SmolGPT(config)
        mlp_orig = get_mlp_params(model)
        assert mlp_orig.dim() == 1

        wte_before = model.wte.weight.clone()
        attn_before = model.h[0].attn.c_q.weight.clone()

        perturbed_mlp = mlp_orig + 0.99
        set_mlp_params(model, perturbed_mlp)

        mlp_check = get_mlp_params(model)
        assert torch.allclose(mlp_check, perturbed_mlp, atol=1e-6)
        # Verify non-MLP parameters were untouched
        assert torch.allclose(model.wte.weight, wte_before)
        assert torch.allclose(model.h[0].attn.c_q.weight, attn_before)

    def test_tinystories_batch_loader(self):
        # Create synthetic token sequence
        seq_len = 1000
        synthetic_tokens = np.arange(seq_len, dtype=np.uint16)

        # Random batch
        x, y = get_random_batch(synthetic_tokens, batch_size=4, block_size=16, device="cpu")
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        # Next-token targets should be x shifted by 1
        assert torch.all(y == x + 1)

        # Deterministic val batches
        val_batches = get_deterministic_val_batches(
            synthetic_tokens, num_batches=3, batch_size=2, block_size=16, device="cpu"
        )
        assert len(val_batches) == 3
        x0, y0 = val_batches[0]
        assert x0.shape == (2, 16)
        assert y0.shape == (2, 16)
        assert torch.all(y0 == x0 + 1)

    def test_smolgpt_low_rank_engine(self, config):
        model = SmolGPT(config)
        P = sum(p.numel() for p in model.parameters())
        T = 10

        # Construct synthetic weight trajectory
        w0 = get_flat_params(model).cpu().numpy()
        window = np.zeros((P, T), dtype=np.float32)
        direction = np.random.randn(P).astype(np.float32)
        for t in range(T):
            window[:, t] = w0 + direction * (t * 0.05)

        engine = LowRankTrajectoryEngine(window, ref_checkpoint=window[:, -1])
        diag = engine.get_rank_diagnostics(rank=2)
        assert diag["rank"] == 2
        # Because displacement is purely 1-dimensional, rank 1 should explain ~100%
        assert diag["explained_variance"] > 0.999

        # Forecast linear
        w_pred, c_pred, elapsed = engine.forecast_linear(rank=2, horizon=4)
        assert w_pred.shape == (P,)
        assert elapsed >= 0.0

        # Forecast damped
        w_damped, c_damped, elapsed_d = engine.forecast_damped(rank=2, horizon=4, gamma=0.95)
        assert w_damped.shape == (P,)

        # Inject into model and evaluate
        set_flat_params(model, w_pred)
        x = torch.randint(0, config.vocab_size, (2, 16))
        y = torch.randint(0, config.vocab_size, (2, 16))
        val_loss, ppl = evaluate_transformer_loss_and_ppl(model, [(x, y)])
        assert not math.isnan(val_loss)
        assert not math.isnan(ppl)
