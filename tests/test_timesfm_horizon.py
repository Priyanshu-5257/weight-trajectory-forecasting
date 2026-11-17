import os
import sys
import pytest
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.low_rank import LowRankTrajectoryEngine

class DummyTimesFM:
    """
    Mock TimesFM model returning distinct outputs across 128 prediction steps.
    """
    def __init__(self):
        pass

    def __call__(self, inputs_t):
        batch_size = inputs_t.shape[0]
        # Generate monotonic increasing steps so each step has a distinct value
        steps = torch.arange(1, 129, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        # Prediction: step value times input mean
        preds = steps.to(inputs_t.device) * 0.1
        class DummyOutput:
            def __init__(self, p):
                self.mean_predictions = p
        return DummyOutput(preds)

def test_timesfm_horizon_sensitivity():
    P = 100
    T = 32
    weights = np.random.randn(P, T)
    engine = LowRankTrajectoryEngine(weights)

    mock_tfm = DummyTimesFM()
    device = "cpu"

    w_8, c_8, _, _ = engine.forecast_timesfm(rank=2, horizon=8, tfm_model=mock_tfm, device=device)
    w_16, c_16, _, _ = engine.forecast_timesfm(rank=2, horizon=16, tfm_model=mock_tfm, device=device)
    w_32, c_32, _, _ = engine.forecast_timesfm(rank=2, horizon=32, tfm_model=mock_tfm, device=device)

    # Verify predictions are NOT constant across horizons
    assert not torch.allclose(w_8, w_16), "TimesFM weights at h=8 and h=16 must not be identical!"
    assert not torch.allclose(w_16, w_32), "TimesFM weights at h=16 and h=32 must not be identical!"
    assert not np.allclose(c_8, c_16), "TimesFM latent coefficients at h=8 and h=16 must not be identical!"
    assert not np.allclose(c_16, c_32), "TimesFM latent coefficients at h=16 and h=32 must not be identical!"

def test_weight_decay_baseline():
    P = 100
    T = 32
    weights = np.ones((P, T), dtype=np.float32)
    engine = LowRankTrajectoryEngine(weights)

    w_decay, c_decay, _ = engine.forecast_weight_decay(horizon=8, eta=1e-3, weight_decay=1.0, interval=25)
    expected_factor = (1.0 - 1e-3 * 1.0) ** (8 * 25)
    expected_w = weights[:, -1] * expected_factor

    assert np.allclose(w_decay.numpy(), expected_w, atol=1e-5), "Weight decay formula calculation mismatch!"
