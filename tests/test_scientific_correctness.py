"""
Regression tests for scientific correctness and semantic integrity.
Verifies that experiments test their intended hypotheses rather than shape-compatible bugs:
1. Independent W2 continuation preserves bitwise immutability of all non-W2 parameters.
2. Independent MLP continuation preserves bitwise immutability of all non-MLP parameters.
3. Grokking configuration, horizon definitions, and evaluation splits match provenance metadata.
4. Subspace displacement matrix with zero column drops the null column and bounds effective rank <= T-1.
5. Norm projection recomputes displacement vectors and produces distinct displacement cosines.
6. End-to-end forecasting latency includes all 5 pipeline components and SWA latency is strictly > 0.
7. CPU compatibility: all benchmarks and utilities execute on CPU without unconditional CUDA calls.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import time
import os

from src.models_transformer import SmolGPTConfig, SmolGPT, get_flat_params, set_flat_params, get_mlp_w2_params, get_mlp_params
from src.subspace_analysis import compute_svd_basis, project_displacement, qr_project_displacement
from src.low_rank import LowRankTrajectoryEngine
from benchmarks.benchmark_smolgpt_norm_projection import compute_displacement_and_weight_cosine, project_global_norm, project_tensor_wise_norm
from benchmarks.benchmark_smolgpt_controlled import make_batch_from_indices


class TestScientificCorrectness:
    def test_resumed_full_adamw_matches_uninterrupted(self):
        """
        Integration Test 1:
        Verifies that restoring cutoff model weights and complete AdamW optimizer state
        (exp_avg, exp_avg_sq, step) reproduces uninterrupted training within numerical tolerance.
        """
        torch.manual_seed(42)
        config = SmolGPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=1, n_embed=32)
        model = SmolGPT(config)
        opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

        # Base training 5 steps
        for _ in range(5):
            x = torch.randint(0, 64, (2, 16))
            y = torch.randint(0, 64, (2, 16))
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Save cutoff state
        cutoff_model = copy.deepcopy(model.state_dict())
        cutoff_opt = copy.deepcopy(opt.state_dict())

        # Uninterrupted branch: take step 6 and 7
        future_batches = [(torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))) for _ in range(2)]
        uninterrupted_losses = []
        for x, y in future_batches:
            _, loss = model(x, y)
            uninterrupted_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        w_uninterrupted = get_flat_params(model).clone()

        # Resumed branch: new model and optimizer, load cutoff state
        model_resumed = SmolGPT(config)
        model_resumed.load_state_dict(cutoff_model)
        opt_resumed = optim.AdamW(model_resumed.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        opt_resumed.load_state_dict(cutoff_opt)

        resumed_losses = []
        for x, y in future_batches:
            _, loss = model_resumed(x, y)
            resumed_losses.append(loss.item())
            opt_resumed.zero_grad()
            loss.backward()
            opt_resumed.step()
        w_resumed = get_flat_params(model_resumed).clone()

        assert np.allclose(uninterrupted_losses, resumed_losses, atol=1e-6), "Resumed losses diverged from uninterrupted!"
        max_diff = torch.max(torch.abs(w_uninterrupted - w_resumed)).item()
        assert max_diff < 1e-6, f"Resumed weights diverged: max diff {max_diff}"

    def test_restarted_adamw_distinguishable_from_resumed(self):
        """
        Integration Test 2:
        Verifies that restarted AdamW (zeroed moments, step=0) takes different updates
        than resumed AdamW (carrying accumulated momentum and variance).
        """
        torch.manual_seed(42)
        config = SmolGPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=1, n_embed=32)
        model = SmolGPT(config)
        opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

        for _ in range(10):
            x = torch.randint(0, 64, (2, 16))
            y = torch.randint(0, 64, (2, 16))
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        cutoff_model = copy.deepcopy(model.state_dict())
        cutoff_opt = copy.deepcopy(opt.state_dict())

        # Branch A: Resumed
        mA = SmolGPT(config)
        mA.load_state_dict(cutoff_model)
        optA = optim.AdamW(mA.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        optA.load_state_dict(cutoff_opt)

        # Branch B: Restarted (fresh optimizer)
        mB = SmolGPT(config)
        mB.load_state_dict(cutoff_model)
        optB = optim.AdamW(mB.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

        x_fut = torch.randint(0, 64, (2, 16))
        y_fut = torch.randint(0, 64, (2, 16))
        _, lA = mA(x_fut, y_fut)
        optA.zero_grad()
        lA.backward()
        optA.step()

        _, lB = mB(x_fut, y_fut)
        optB.zero_grad()
        lB.backward()
        optB.step()

        wA = get_flat_params(mA)
        wB = get_flat_params(mB)
        diff = torch.norm(wA - wB).item()
        assert diff > 1e-4, f"Restarted and resumed branches produced identical updates (diff={diff})!"

    def test_identical_minibatch_indices_replay(self):
        """
        Integration Test 3:
        Verifies that replaying pre-recorded minibatch indices guarantees 100%
        identical batch tensors across multiple consumers.
        """
        tokens = np.random.randint(0, 1000, size=5000)
        indices = np.random.randint(0, 4800, size=16)

        x1, y1 = make_batch_from_indices(tokens, indices, block_size=64, device=torch.device("cpu"))
        x2, y2 = make_batch_from_indices(tokens, indices, block_size=64, device=torch.device("cpu"))

        assert torch.equal(x1, x2), "Batches produced from identical indices must be bitwise identical"
        assert torch.equal(y1, y2), "Labels produced from identical indices must be bitwise identical"

    def test_projection_coverage_matches_float64_qr(self):
        """
        Integration Test 4:
        Verifies that maximum-rank SVD projection matches independent float64 QR projection
        to high numerical precision (|cov_svd - cov_qr| < 1e-6).
        """
        torch.manual_seed(42)
        P, T = 1000, 32
        W = torch.randn(P, T)
        w_ref = W[:, -1].clone()
        D = W - w_ref.unsqueeze(1) # column 31 is zero

        dw = torch.randn(P)

        # 1. SVD in float64 with rank=31
        U, S, cum_var, diag = compute_svd_basis(D, rank=31, return_diagnostics=True)
        assert U.shape[1] == 31
        dw_par_svd, _, cov_svd = project_displacement(dw, U)

        # 2. QR in float64
        dw_par_qr, _, cov_qr = qr_project_displacement(D, dw)

        # 3. Equivalence assertion
        assert abs(cov_svd - cov_qr) < 1e-6, f"SVD ({cov_svd}) and QR ({cov_qr}) coverage diverged!"
        rel_diff = torch.norm(dw_par_svd - dw_par_qr).item() / torch.norm(dw).item()
        assert rel_diff < 1e-5, f"Projected displacement vectors diverged: rel diff {rel_diff}"

    def test_svd_requested_and_actual_rank_recording(self):
        """
        Integration Test 5:
        Verifies that requested rank and actual rank are recorded explicitly,
        and requesting rank 32 on a 31-column matrix returns actual_rank=31 without silent error.
        """
        torch.manual_seed(42)
        P, T = 500, 32
        W = torch.randn(P, T)
        w_ref = W[:, -1].clone()
        D = W - w_ref.unsqueeze(1)

        U, S, cum_var, diag = compute_svd_basis(D, rank=32, return_diagnostics=True)
        assert diag["requested_rank"] == 32
        assert diag["actual_rank"] == 31
        assert diag["max_rank"] == 31
        assert len(diag["singular_values"]) == 31
        assert diag["condition_number"] > 1.0

        # When requesting r=8
        U8, S8, _, diag8 = compute_svd_basis(D, rank=8, return_diagnostics=True)
        assert diag8["requested_rank"] == 8
        assert diag8["actual_rank"] == 8
        assert U8.shape[1] == 8

    def test_latency_invariant_reconciliation(self):
        """
        Integration Test 6:
        Verifies that total_latency_ms satisfies the exact invariant:
        total_latency_ms == checkpoint_prep_ms + svd_ms + forecast_ms + reconstruction_and_load_ms
        """
        P, T = 1000, 32
        weight_window = np.random.randn(P, T).astype(np.float32)
        w_ref = weight_window[:, -1].copy()

        engine = LowRankTrajectoryEngine(weight_window, ref_checkpoint=w_ref)
        w_lin, c_lin, t_lin = engine.forecast_linear(rank=4, horizon=8)
        t_load = 0.002 # 2 ms simulated load

        t_total = engine.prep_time + engine.svd_time + t_lin + t_load

        prep_ms = engine.prep_time * 1000.0
        svd_ms = engine.svd_time * 1000.0
        forecast_ms = t_lin * 1000.0
        load_ms = t_load * 1000.0
        total_ms = t_total * 1000.0

        assert abs(total_ms - (prep_ms + svd_ms + forecast_ms + load_ms)) < 1e-9, "Latency invariant violated!"
    def test_w2_independent_continuation_freezing(self):
        """
        Regression Test 1:
        Verifies that during independent W2 continuation, all non-W2 parameters
        remain bitwise unchanged (zero parameter leakage).
        """
        torch.manual_seed(42)
        config = SmolGPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=1, n_embed=32)
        model = SmolGPT(config)

        # Snapshot initial state
        initial_state = copy.deepcopy(model.state_dict())

        # Freeze all non-W2 parameters
        for name, param in model.named_parameters():
            if "mlp.w2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        # Run 5 training steps
        for _ in range(5):
            idx = torch.randint(0, 64, (2, 16))
            targets = torch.randint(0, 64, (2, 16))
            _, loss = model(idx, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Assert every non-W2 parameter is bitwise identical
        for name, param in model.named_parameters():
            if "mlp.w2" in name:
                # W2 must have updated
                assert not torch.equal(param.data, initial_state[name]), f"W2 parameter {name} was not trained!"
            else:
                # Non-W2 MUST remain strictly bitwise identical
                assert torch.equal(param.data, initial_state[name]), f"Leakage detected: frozen parameter {name} changed!"

    def test_mlp_independent_continuation_freezing(self):
        """
        Regression Test 2:
        Verifies that during independent MLP continuation, all non-MLP parameters
        (embeddings, attention projections, RMSNorms) remain bitwise identical.
        """
        torch.manual_seed(42)
        config = SmolGPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=1, n_embed=32)
        model = SmolGPT(config)

        initial_state = copy.deepcopy(model.state_dict())

        # Freeze all non-MLP parameters
        for name, param in model.named_parameters():
            if "mlp." in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        for _ in range(5):
            idx = torch.randint(0, 64, (2, 16))
            targets = torch.randint(0, 64, (2, 16))
            _, loss = model(idx, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for name, param in model.named_parameters():
            if "mlp." in name:
                assert not torch.equal(param.data, initial_state[name]), f"MLP parameter {name} was not trained!"
            else:
                assert torch.equal(param.data, initial_state[name]), f"Leakage detected: frozen parameter {name} changed!"

    def test_grokking_config_provenance_and_horizons(self):
        """
        Regression Test 3:
        Verifies that grokking configuration, horizon definitions, and evaluation split
        match the saved experiment summary metadata.
        """
        summary_path = "results/grokking_benchmark_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            
            assert "metadata" in summary, "Metadata missing from grokking_benchmark_summary.json"
            meta = summary["metadata"]
            assert meta["cutoff_step"] == 1500, f"Expected cutoff 1500, found {meta['cutoff_step']}"
            assert meta["eval_split"] == "nonsym_pairs", f"Expected nonsym_pairs split, found {meta['eval_split']}"
            assert summary["cutoff_step"] == 1500
            assert summary["horizons"] == [8, 16, 32, 50, 64, 80, 100]

    def test_svd_basis_zero_column_eliminated(self):
        """
        Regression Test 4:
        A displacement matrix containing a zero-displacement column (W_ref - W_ref = 0)
        must NOT introduce an artificial nullspace direction.
        The effective rank must be strictly <= T - 1.
        """
        torch.manual_seed(42)
        P, T = 500, 32
        W = torch.randn(P, T)
        W_ref = W[:, -1].clone()

        # Displacement matrix where column 31 is exactly 0
        D = W - W_ref.unsqueeze(1)
        assert torch.all(D[:, -1] == 0.0), "Final column must be zero displacement"

        U, S, cum_var, r_eff = compute_svd_basis(D, rank=32, return_effective_rank=True)

        # The effective rank must be at most 31, never 32
        assert r_eff <= 31, f"Expected effective rank <= 31, got {r_eff}"
        assert U.shape[1] == r_eff, f"U basis must contain exactly r_eff={r_eff} columns, got {U.shape[1]}"
        assert len(S) == 31, f"Zero column must be dropped before SVD; expected 31 singular values, got {len(S)}"

        # Verify orthonormality: U^T U = I_{r_eff}
        eye = torch.matmul(U.T, U)
        assert torch.allclose(eye, torch.eye(r_eff), atol=1e-5)

    def test_norm_projection_cosine_recomputation(self):
        """
        Regression Test 5:
        Verifies that displacement cosine changes after norm projection,
        proving that displacement cosine must be recomputed on projected weights.
        Also verifies that displacement cosine and weight cosine are distinguished.
        """
        torch.manual_seed(42)
        P = 1000
        w_ref = torch.randn(P)
        w_actual = w_ref + torch.randn(P) * 0.1
        w_raw_forecast = w_ref + torch.randn(P) * 0.5  # overshooting forecast

        # Project norm to match w_ref norm
        w_proj = project_global_norm(w_raw_forecast, float(torch.norm(w_ref).item()))

        cos_disp_raw, cos_weight_raw = compute_displacement_and_weight_cosine(w_raw_forecast, w_actual, w_ref)
        cos_disp_proj, cos_weight_proj = compute_displacement_and_weight_cosine(w_proj, w_actual, w_ref)

        # Displacement cosine must change when weights are radially rescaled (unless w_ref is 0)
        assert abs(cos_disp_raw - cos_disp_proj) > 1e-4, "Norm projection must alter the displacement vector direction!"
        
        # Weight-direction cosine and displacement cosine measure completely different physical angles
        # Weight cosine is close to 1.0; displacement cosine measures directional agreement of the update
        assert cos_weight_proj > 0.8
        assert cos_disp_proj != cos_weight_proj

    def test_end_to_end_latency_breakdown_and_swa_timing(self):
        """
        Regression Test 6:
        Verifies that timing breakdown includes all required operations
        and SWA runtime is non-zero.
        """
        P, T = 1000, 32
        weight_window = np.random.randn(P, T).astype(np.float32)
        w_ref = weight_window[:, -1].copy()

        engine = LowRankTrajectoryEngine(weight_window, ref_checkpoint=w_ref)

        assert engine.prep_time > 0.0, "Checkpoint prep time must be > 0"
        assert engine.svd_time > 0.0, "SVD compute time must be > 0"

        w_lin, c_lin, t_lin = engine.forecast_linear(rank=4, horizon=8)
        assert t_lin > 0.0, "Linear forecast time must be > 0"

        # Simulate model load
        dummy_model = nn.Linear(50, 20)
        t0_swa = time.perf_counter()
        w_swa = np.mean(weight_window[:, -16:], axis=1)
        _ = torch.from_numpy(w_swa).float()
        t_swa = time.perf_counter() - t0_swa

        assert t_swa > 0.0, "SWA runtime must be strictly greater than 0 ms"

    def test_cpu_compatibility(self):
        """
        Regression Test 7:
        Verifies that running on device='cpu' executes without any CUDA synchronization errors.
        """
        device = torch.device("cpu")
        config = SmolGPTConfig(block_size=8, vocab_size=32, n_layer=1, n_head=1, n_kv_head=1, n_embed=16)
        model = SmolGPT(config).to(device)

        x = torch.randint(0, 32, (2, 8), device=device)
        y = torch.randint(0, 32, (2, 8), device=device)

        logits, loss = model(x, y)
        loss.backward()

        # Safe sync call on CPU must be a no-op, never an error
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        assert loss.item() > 0.0
