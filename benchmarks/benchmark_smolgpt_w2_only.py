"""
Rigorous Weight Trajectory Forecasting Benchmark: W2 Down-Projection Isolated Architecture (Choice A Option 2).
Forecasts ONLY the Feed-Forward Down-Projection linear layers (w2 across all 4 layers = 174,592 params)
while keeping Token Embeddings, Causal Attention, RMSNorms, and MLP w1/w3 strictly FROZEN at Step 1000.
"""

import os
import sys
import json
import time
import math
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TimesFmModelForPrediction
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models_transformer import (
    SmolGPTConfig,
    SmolGPT,
    get_flat_params,
    set_flat_params,
    get_mlp_w2_params,
    set_mlp_w2_params,
    evaluate_transformer_loss_and_ppl
)
from src.data_transformer import (
    load_tinystories_dataset,
    get_random_batch,
    get_deterministic_val_batches,
    get_deterministic_train_batches
)
from src.low_rank import LowRankTrajectoryEngine


def calc_stats(vals):
    vals = np.array(vals, dtype=np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2.0, len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}


def calc_paired_diff_stats(method_vals, baseline_vals):
    diffs = np.array(method_vals, dtype=np.float64) - np.array(baseline_vals, dtype=np.float64)
    res = calc_stats(diffs)
    if len(diffs) > 1 and np.std(diffs) > 1e-9:
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        res["t_stat"] = float(t_stat)
        res["p_val"] = float(p_val)
    else:
        res["t_stat"] = 0.0
        res["p_val"] = 1.0
    return res


def run_smolgpt_w2_benchmark(
    seeds=[42, 43, 44],
    cutoff_step=1000,
    context_checkpoints=32,
    checkpoint_interval=20,
    ranks=[1, 2, 4, 8, 16],
    horizons=[8, 16, 32, 64],  # +160, +320, +640, +1280 steps
    batch_size=16,
    block_size=128,
    lr=1e-3,
    weight_decay=0.01,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" RIGOROUS SMOLGPT W2-ONLY WEIGHT TRAJECTORY BENCHMARK (Choice A Option 2)")
    print(f" Target: MLP Down-Projection (w2 across all 4 layers = 174,592 params)")
    print(f" Frozen: Token Embeddings, Causal Attention, RMSNorms, and MLP w1/w3 at Step {cutoff_step}")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizons: {horizons}")
    print(f" Device: {device}")
    print("=" * 80, flush=True)

    # 1. Load Dataset
    print("Loading TinyStories token dataset...", flush=True)
    train_tokens, val_tokens = load_tinystories_dataset("data/tinystories/tinystories_sub.npz")

    val_batches = get_deterministic_val_batches(
        val_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device
    )
    train_batches = get_deterministic_train_batches(
        train_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device
    )
    print(f"Materialized 20 val batches and 20 train batches (40,960 tokens each)", flush=True)

    def eval_both(m):
        t_loss, t_ppl = evaluate_transformer_loss_and_ppl(m, train_batches)
        v_loss, v_ppl = evaluate_transformer_loss_and_ppl(m, val_batches)
        return {
            "train_loss": t_loss,
            "train_ppl": t_ppl,
            "val_loss": v_loss,
            "val_ppl": v_ppl,
            "gen_gap": v_loss - t_loss
        }

    # 2. Load TimesFM Model
    t_load_start = time.perf_counter()
    print("Loading TimesFM-2.0-500M...", flush=True)
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()
    print(f"TimesFM loaded in {time.perf_counter() - t_load_start:.2f} s", flush=True)

    max_horizon = max(horizons)
    total_steps = cutoff_step + (max_horizon * checkpoint_interval)
    start_context_step = cutoff_step - (context_checkpoints * checkpoint_interval)

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        t_seed_start = time.perf_counter()
        print(f"\n>>> Running Seed {seed} ({seed_idx + 1}/{len(seeds)})", flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        config = SmolGPTConfig(
            block_size=block_size,
            vocab_size=4096,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embed=128,
            dropout=0.0
        )
        model = SmolGPT(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

        # Track trajectory checkpoints
        w2_checkpoints_history = []  # shape: (P_w2, context_checkpoints)
        future_checkpoints = {}      # h -> dict

        print(f"Phase 1: Training to Cutoff Step {cutoff_step}...", flush=True)
        t_train_start = time.perf_counter()
        step_times = []

        for step in range(1, total_steps + 1):
            t_s0 = time.perf_counter()
            model.train()
            x, y = get_random_batch(train_tokens, batch_size=batch_size, block_size=block_size, device=device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_step = time.perf_counter() - t_s0
            step_times.append(dt_step)

            # Record history checkpoints
            if start_context_step < step <= cutoff_step:
                if (step - start_context_step) % checkpoint_interval == 0:
                    w2_t = get_mlp_w2_params(model).cpu().numpy().copy()
                    w2_checkpoints_history.append(w2_t)
                    if step == cutoff_step:
                        w_ref_full = get_flat_params(model).cpu().numpy().copy()
                        cutoff_model_state = copy.deepcopy(model.state_dict())
                        torch_rng_cutoff = torch.get_rng_state()
                        cuda_rng_cutoff = torch.cuda.get_rng_state(device) if device.type == "cuda" and torch.cuda.is_available() else None
                        np_rng_cutoff = np.random.get_state()

            # Record future continuation checkpoints
            if step > cutoff_step:
                steps_after = step - cutoff_step
                if steps_after % checkpoint_interval == 0:
                    h_curr = steps_after // checkpoint_interval
                    if h_curr in horizons:
                        w_future_full = get_flat_params(model).cpu().numpy().copy()
                        w2_future = get_mlp_w2_params(model).cpu().numpy().copy()
                        future_checkpoints[h_curr] = {
                            "w_full": w_future_full,
                            "w_w2": w2_future,
                            "cum_step_time": float(np.sum(step_times[cutoff_step:step]))
                        }

            if step % 200 == 0:
                v_loss, v_ppl = evaluate_transformer_loss_and_ppl(model, val_batches)
                print(f"  Step {step:4d}/{total_steps} | Batch Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f} | PPL: {v_ppl:.2f}", flush=True)

        train_elapsed = time.perf_counter() - t_train_start
        print(f"Training completed in {train_elapsed:.2f} s ({total_steps / train_elapsed:.1f} steps/s)", flush=True)

        # ----------------------------------------------------------------------
        # Phase 1b: Genuine Independent W2-Only Continuation from Cutoff Step
        # ----------------------------------------------------------------------
        print(f"Phase 1b: Independent W2-Only Continuation from Cutoff Step {cutoff_step}...", flush=True)
        model_w2 = copy.deepcopy(model)
        model_w2.load_state_dict(cutoff_model_state)
        for name, param in model_w2.named_parameters():
            if "mlp.w2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer_w2 = optim.AdamW(
            [p for p in model_w2.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )

        # Restore RNG to cutoff step so identical future data batches are produced
        torch.set_rng_state(torch_rng_cutoff)
        if cuda_rng_cutoff is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_cutoff, device)
        np.random.set_state(np_rng_cutoff)

        indep_w2_checkpoints = {}
        w2_step_times = []
        t_w2_start = time.perf_counter()

        for step in range(cutoff_step + 1, total_steps + 1):
            t_w2_s0 = time.perf_counter()
            model_w2.train()
            x, y = get_random_batch(train_tokens, batch_size=batch_size, block_size=block_size, device=device)
            _, loss = model_w2(x, y)
            optimizer_w2.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model_w2.parameters() if p.requires_grad], 1.0)
            optimizer_w2.step()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_step = time.perf_counter() - t_w2_s0
            w2_step_times.append(dt_step)

            steps_after = step - cutoff_step
            if steps_after % checkpoint_interval == 0:
                h_curr = steps_after // checkpoint_interval
                if h_curr in horizons:
                    # Semantic regression check: bitwise immutability of all frozen parameters
                    for name, param in model_w2.named_parameters():
                        if not param.requires_grad:
                            ref_p = cutoff_model_state[name].to(device)
                            assert torch.equal(param.data, ref_p.data), f"Frozen parameter {name} modified during W2 continuation!"

                    w_future_full_indep = get_flat_params(model_w2).cpu().numpy().copy()
                    w2_future_indep = get_mlp_w2_params(model_w2).cpu().numpy().copy()
                    indep_w2_checkpoints[h_curr] = {
                        "w_full": w_future_full_indep,
                        "w_w2": w2_future_indep,
                        "cum_step_time": float(np.sum(w2_step_times[:steps_after]))
                    }

        w2_elapsed = time.perf_counter() - t_w2_start
        print(f"Independent W2 training completed in {w2_elapsed:.2f} s", flush=True)

        # Assemble W2 historical weight window
        w2_weight_window = np.column_stack(w2_checkpoints_history)  # (174592, 32)
        w_w2_ref = w2_weight_window[:, -1].copy()

        # Build trajectory engine specifically on W2 weights
        engine = LowRankTrajectoryEngine(w2_weight_window, ref_checkpoint=w_w2_ref)

        def reset_to_frozen_ref():
            set_flat_params(model, torch.from_numpy(w_ref_full).float())

        # Evaluate Frozen Checkpoint
        reset_to_frozen_ref()
        frozen_metrics = eval_both(model)

        # Evaluate W2-Only Matched SWA (over last 16 W2 checkpoints)
        reset_to_frozen_ref()
        t0_swa = time.perf_counter()
        w_w2_swa = np.mean(w2_weight_window[:, -16:], axis=1)
        set_mlp_w2_params(model, torch.from_numpy(w_w2_swa).float())
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_swa_load = time.perf_counter() - t0_swa
        swa_w2_metrics = eval_both(model)

        # Evaluate W2-Only Matched EMA (beta=0.9)
        reset_to_frozen_ref()
        t0_ema = time.perf_counter()
        w_w2_ema = w2_weight_window[:, 0].copy()
        beta = 0.9
        for col in range(1, w2_weight_window.shape[1]):
            w_w2_ema = beta * w_w2_ema + (1.0 - beta) * w2_weight_window[:, col]
        set_mlp_w2_params(model, torch.from_numpy(w_w2_ema).float())
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ema_load = time.perf_counter() - t0_ema
        ema_w2_metrics = eval_both(model)

        # SVD Rank Diagnostics for W2
        rank_diagnostics = {}
        for r in ranks:
            diag = engine.get_rank_diagnostics(r)
            rank_diagnostics[f"rank_{r}"] = diag

        seed_data = {
            "seed": seed,
            "cutoff_step": cutoff_step,
            "frozen": frozen_metrics,
            "swa_w2": swa_w2_metrics,
            "ema_w2": ema_w2_metrics,
            "rank_diagnostics": rank_diagnostics,
            "singular_values": [float(s) for s in engine.S],
            "explained_variance_ratio": [float(v) for v in engine.explained_variance_ratio],
            "horizons": {}
        }

        # Evaluate Horizons
        for h in horizons:
            h_data = {
                "horizon_checkpoints": h,
                "equivalent_steps": h * checkpoint_interval,
                "methods": {}
            }

            w_actual_full = future_checkpoints[h]["w_full"]
            w_actual_w2 = future_checkpoints[h]["w_w2"]
            adamw_time = future_checkpoints[h]["cum_step_time"]

            # 1. Full Continued AdamW (Ground Truth Joint Optimizer)
            set_flat_params(model, torch.from_numpy(w_actual_full).float())
            full_adamw_metrics = eval_both(model)
            h_data["methods"]["full_continued_adamw"] = {
                **full_adamw_metrics,
                "cos_sim_w2": 1.0,
                "rel_l2_w2": 0.0,
                "time_ms": adamw_time * 1000.0,
                "speedup": 1.0,
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": adamw_time * 1000.0
                }
            }

            # 2a. Genuine Independent W2-Only Continuation (Upper bound for updating ONLY W2)
            w_indep_full = indep_w2_checkpoints[h]["w_full"]
            w_indep_w2 = indep_w2_checkpoints[h]["w_w2"]
            indep_w2_time = indep_w2_checkpoints[h]["cum_step_time"]
            set_flat_params(model, torch.from_numpy(w_indep_full).float())
            w2_indep_metrics = eval_both(model)
            norm_w2_act = np.linalg.norm(w_actual_w2 - w_w2_ref)
            norm_w2_indep = np.linalg.norm(w_indep_w2 - w_w2_ref)
            d_indep = w_indep_w2 - w_w2_ref
            d_act = w_actual_w2 - w_w2_ref
            cos_indep = float(np.dot(d_indep, d_act) / (norm_w2_indep * norm_w2_act + 1e-8)) if norm_w2_indep > 1e-8 else 0.0

            h_data["methods"]["w2_independent_continuation"] = {
                **w2_indep_metrics,
                "cos_sim_w2": cos_indep,
                "rel_l2_w2": float(np.linalg.norm(w_indep_w2 - w_actual_w2) / (norm_w2_act + 1e-8)),
                "time_ms": indep_w2_time * 1000.0,
                "speedup": float(adamw_time / max(indep_w2_time, 1e-6)),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": indep_w2_time * 1000.0
                }
            }

            # 2b. Oracle W2 Transplant (Preserved Historical Condition: W2 from full training inserted into frozen checkpoint)
            reset_to_frozen_ref()
            set_mlp_w2_params(model, torch.from_numpy(w_actual_w2).float())
            w2_actual_metrics = eval_both(model)
            h_data["methods"]["oracle_w2_transplant"] = {
                **w2_actual_metrics,
                "cos_sim_w2": 1.0,
                "rel_l2_w2": 0.0,
                "time_ms": adamw_time * 1000.0,
                "speedup": 1.0,
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": adamw_time * 1000.0
                }
            }
            # Maintain backward-compatible alias for existing downstream tools
            h_data["methods"]["w2_only_actual_continuation"] = h_data["methods"]["oracle_w2_transplant"]

            # 3. Frozen Checkpoint Baseline
            reset_to_frozen_ref()
            h_data["methods"]["frozen_checkpoint"] = {
                **frozen_metrics,
                "cos_sim_w2": 0.0,
                "rel_l2_w2": 1.0,
                "time_ms": 0.0,
                "speedup": float("inf"),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": 0.0
                }
            }

            # 4. W2-Only SWA Baseline
            reset_to_frozen_ref()
            set_mlp_w2_params(model, torch.from_numpy(w_w2_swa).float())
            d_swa = w_w2_swa - w_w2_ref
            cos_swa = float(np.dot(d_swa, d_act) / (np.linalg.norm(d_swa) * norm_w2_act + 1e-8)) if np.linalg.norm(d_swa) > 1e-8 else 0.0
            h_data["methods"]["w2_matched_swa"] = {
                **swa_w2_metrics,
                "cos_sim_w2": cos_swa,
                "rel_l2_w2": float(np.linalg.norm(w_w2_swa - w_actual_w2) / (norm_w2_act + 1e-8)),
                "time_ms": t_swa_load * 1000.0,
                "speedup": float(adamw_time / max(t_swa_load, 1e-6)),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": t_swa_load * 1000.0,
                    "total_latency_ms": t_swa_load * 1000.0
                }
            }

            # 5. W2-Only EMA Baseline
            reset_to_frozen_ref()
            set_mlp_w2_params(model, torch.from_numpy(w_w2_ema).float())
            d_ema = w_w2_ema - w_w2_ref
            cos_ema = float(np.dot(d_ema, d_act) / (np.linalg.norm(d_ema) * norm_w2_act + 1e-8)) if np.linalg.norm(d_ema) > 1e-8 else 0.0
            h_data["methods"]["w2_matched_ema"] = {
                **ema_w2_metrics,
                "cos_sim_w2": cos_ema,
                "rel_l2_w2": float(np.linalg.norm(w_w2_ema - w_actual_w2) / (norm_w2_act + 1e-8)),
                "time_ms": t_ema_load * 1000.0,
                "speedup": float(adamw_time / max(t_ema_load, 1e-6)),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": t_ema_load * 1000.0,
                    "total_latency_ms": t_ema_load * 1000.0
                }
            }

            # Extrapolations across ranks strictly in W2 subspace
            for r in ranks:
                # Linear
                w_lin_w2, c_lin, t_lin = engine.forecast_linear(r, h)
                w_lin_np = w_lin_w2.numpy()
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(model, w_lin_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lin_load = time.perf_counter() - t0_load
                lin_metrics = eval_both(model)
                d_lin = w_lin_np - w_w2_ref
                cos_lin = float(np.dot(d_lin, d_act) / (np.linalg.norm(d_lin) * norm_w2_act + 1e-8)) if np.linalg.norm(d_lin) > 1e-8 else 0.0
                t_lin_total = engine.svd_time + t_lin + t_lin_load

                h_data["methods"][f"w2_linear_r{r}"] = {
                    **lin_metrics,
                    "cos_sim_w2": cos_lin,
                    "rel_l2_w2": float(np.linalg.norm(w_lin_np - w_actual_w2) / (norm_w2_act + 1e-8)),
                    "time_ms": t_lin_total * 1000.0,
                    "speedup": float(adamw_time / max(t_lin_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_lin * 1000.0,
                        "reconstruction_and_load_ms": t_lin_load * 1000.0,
                        "total_latency_ms": t_lin_total * 1000.0
                    }
                }

                # Damped (gamma=0.95)
                w_damp_w2, c_damp, t_damp = engine.forecast_damped(r, h, gamma=0.95)
                w_damp_np = w_damp_w2.numpy()
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(model, w_damp_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_damp_load = time.perf_counter() - t0_load
                damp_metrics = eval_both(model)
                d_damp = w_damp_np - w_w2_ref
                cos_damp = float(np.dot(d_damp, d_act) / (np.linalg.norm(d_damp) * norm_w2_act + 1e-8)) if np.linalg.norm(d_damp) > 1e-8 else 0.0
                t_damp_total = engine.svd_time + t_damp + t_damp_load

                h_data["methods"][f"w2_damped_r{r}"] = {
                    **damp_metrics,
                    "cos_sim_w2": cos_damp,
                    "rel_l2_w2": float(np.linalg.norm(w_damp_np - w_actual_w2) / (norm_w2_act + 1e-8)),
                    "time_ms": t_damp_total * 1000.0,
                    "speedup": float(adamw_time / max(t_damp_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_damp * 1000.0,
                        "reconstruction_and_load_ms": t_damp_load * 1000.0,
                        "total_latency_ms": t_damp_total * 1000.0
                    }
                }

                # TimesFM-2.0
                w_tfm_w2, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device)
                w_tfm_np = w_tfm_w2.numpy()
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(model, w_tfm_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_tfm_load = time.perf_counter() - t0_load
                tfm_metrics = eval_both(model)
                d_tfm = w_tfm_np - w_w2_ref
                cos_tfm = float(np.dot(d_tfm, d_act) / (np.linalg.norm(d_tfm) * norm_w2_act + 1e-8)) if np.linalg.norm(d_tfm) > 1e-8 else 0.0
                t_tfm_total = engine.svd_time + t_tfm_f + t_tfm_r + t_tfm_load

                h_data["methods"][f"w2_timesfm_r{r}"] = {
                    **tfm_metrics,
                    "cos_sim_w2": cos_tfm,
                    "rel_l2_w2": float(np.linalg.norm(w_tfm_np - w_actual_w2) / (norm_w2_act + 1e-8)),
                    "time_ms": t_tfm_total * 1000.0,
                    "speedup": float(adamw_time / max(t_tfm_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r + t_tfm_load) * 1000.0,
                        "total_latency_ms": t_tfm_total * 1000.0
                    }
                }

                # Shuffled TimesFM Control
                w_shuf_w2, c_shuf, t_shuf_f, t_shuf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                w_shuf_np = w_shuf_w2.numpy()
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(model, w_shuf_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_shuf_load = time.perf_counter() - t0_load
                shuf_metrics = eval_both(model)
                d_shuf = w_shuf_np - w_w2_ref
                cos_shuf = float(np.dot(d_shuf, d_act) / (np.linalg.norm(d_shuf) * norm_w2_act + 1e-8)) if np.linalg.norm(d_shuf) > 1e-8 else 0.0
                t_shuf_total = engine.svd_time + t_shuf_f + t_shuf_r + t_shuf_load

                h_data["methods"][f"w2_shuffled_timesfm_r{r}"] = {
                    **shuf_metrics,
                    "cos_sim_w2": cos_shuf,
                    "rel_l2_w2": float(np.linalg.norm(w_shuf_np - w_actual_w2) / (norm_w2_act + 1e-8)),
                    "time_ms": t_shuf_total * 1000.0,
                    "speedup": float(adamw_time / max(t_shuf_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_shuf_f * 1000.0,
                        "reconstruction_and_load_ms": (t_shuf_r + t_shuf_load) * 1000.0,
                        "total_latency_ms": t_shuf_total * 1000.0
                    }
                }

            seed_data["horizons"][f"h_{h}"] = h_data

        all_seed_results.append(seed_data)
        print(f"Seed {seed} finished in {time.perf_counter() - t_seed_start:.2f} s", flush=True)

    # 3. Statistical Aggregation
    print("\nComputing multi-seed statistical summary with 95% CIs and paired tests...", flush=True)
    summary = {
        "seeds": seeds,
        "cutoff_step": cutoff_step,
        "ranks": ranks,
        "horizons": horizons,
        "frozen": {
            "train_loss": calc_stats([s["frozen"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["frozen"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["frozen"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["frozen"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["frozen"]["gen_gap"] for s in all_seed_results]),
        },
        "swa_w2": {
            "train_loss": calc_stats([s["swa_w2"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["swa_w2"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["swa_w2"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["swa_w2"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["swa_w2"]["gen_gap"] for s in all_seed_results]),
        },
        "ema_w2": {
            "train_loss": calc_stats([s["ema_w2"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["ema_w2"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["ema_w2"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["ema_w2"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["ema_w2"]["gen_gap"] for s in all_seed_results]),
        },
        "explained_variance_cum": {},
        "horizons": {}
    }

    for r in ranks:
        ev_vals = [s["rank_diagnostics"][f"rank_{r}"]["explained_variance"] for s in all_seed_results]
        summary["explained_variance_cum"][f"rank_{r}"] = calc_stats(ev_vals)

    first_h_key = f"h_{horizons[0]}"
    all_method_keys = list(all_seed_results[0]["horizons"][first_h_key]["methods"].keys())

    for h in horizons:
        h_key = f"h_{h}"
        summary["horizons"][h_key] = {"methods": {}}
        adamw_ppl_vals = [s["horizons"][h_key]["methods"]["full_continued_adamw"]["val_ppl"] for s in all_seed_results]
        frozen_ppl_vals = [s["horizons"][h_key]["methods"]["frozen_checkpoint"]["val_ppl"] for s in all_seed_results]

        for method in all_method_keys:
            t_loss_vals = [s["horizons"][h_key]["methods"][method]["train_loss"] for s in all_seed_results]
            t_ppl_vals = [s["horizons"][h_key]["methods"][method]["train_ppl"] for s in all_seed_results]
            v_loss_vals = [s["horizons"][h_key]["methods"][method]["val_loss"] for s in all_seed_results]
            v_ppl_vals = [s["horizons"][h_key]["methods"][method]["val_ppl"] for s in all_seed_results]
            gap_vals = [s["horizons"][h_key]["methods"][method]["gen_gap"] for s in all_seed_results]
            cos_vals = [s["horizons"][h_key]["methods"][method]["cos_sim_w2"] for s in all_seed_results]
            l2_vals = [s["horizons"][h_key]["methods"][method]["rel_l2_w2"] for s in all_seed_results]
            time_vals = [s["horizons"][h_key]["methods"][method]["time_ms"] for s in all_seed_results]
            speedup_vals = [s["horizons"][h_key]["methods"][method]["speedup"] for s in all_seed_results]

            paired_vs_adamw = calc_paired_diff_stats(v_ppl_vals, adamw_ppl_vals)
            paired_vs_frozen = calc_paired_diff_stats(v_ppl_vals, frozen_ppl_vals)

            prep_times = [s["horizons"][h_key]["methods"][method].get("timing_breakdown", {}).get("checkpoint_prep_ms", 0.0) for s in all_seed_results]
            svd_times = [s["horizons"][h_key]["methods"][method].get("timing_breakdown", {}).get("svd_ms", 0.0) for s in all_seed_results]
            forecast_times = [s["horizons"][h_key]["methods"][method].get("timing_breakdown", {}).get("forecast_ms", 0.0) for s in all_seed_results]
            recon_times = [s["horizons"][h_key]["methods"][method].get("timing_breakdown", {}).get("reconstruction_and_load_ms", 0.0) for s in all_seed_results]

            summary["horizons"][h_key]["methods"][method] = {
                "train_loss": calc_stats(t_loss_vals),
                "train_ppl": calc_stats(t_ppl_vals),
                "val_loss": calc_stats(v_loss_vals),
                "val_ppl": calc_stats(v_ppl_vals),
                "gen_gap": calc_stats(gap_vals),
                "cos_sim_w2": calc_stats(cos_vals),
                "rel_l2_w2": calc_stats(l2_vals),
                "time_ms": calc_stats(time_vals),
                "speedup": calc_stats(speedup_vals),
                "paired_diff_vs_adamw_ppl": paired_vs_adamw,
                "paired_diff_vs_frozen_ppl": paired_vs_frozen,
                "timing_breakdown": {
                    "checkpoint_prep_ms": calc_stats(prep_times),
                    "svd_ms": calc_stats(svd_times),
                    "forecast_ms": calc_stats(forecast_times),
                    "reconstruction_and_load_ms": calc_stats(recon_times),
                    "total_latency_ms": calc_stats(time_vals)
                }
            }

    # Save JSON and CSV
    os.makedirs("results", exist_ok=True)
    with open("results/smolgpt_w2_only_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open("results/smolgpt_w2_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/smolgpt_w2_only_per_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "horizon", "equivalent_steps", "method", "train_loss", "train_ppl", "val_loss", "val_ppl", "gen_gap", "cos_sim_w2", "rel_l2_w2", "time_ms", "speedup"])
        for s in all_seed_results:
            seed_id = s["seed"]
            for h in horizons:
                h_key = f"h_{h}"
                eq_steps = s["horizons"][h_key]["equivalent_steps"]
                for method, m_data in s["horizons"][h_key]["methods"].items():
                    writer.writerow([
                        seed_id, h, eq_steps, method,
                        f"{m_data['train_loss']:.4f}",
                        f"{m_data['train_ppl']:.2f}",
                        f"{m_data['val_loss']:.4f}",
                        f"{m_data['val_ppl']:.2f}",
                        f"{m_data['gen_gap']:.4f}",
                        f"{m_data['cos_sim_w2']:.4f}",
                        f"{m_data['rel_l2_w2']:.4f}",
                        f"{m_data['time_ms']:.2f}",
                        f"{m_data['speedup']:.2f}"
                    ])

    print("Saved results to results/smolgpt_w2_only_results.json", flush=True)
    print("Saved summary to results/smolgpt_w2_only_summary.json", flush=True)
    print("Saved CSV to results/smolgpt_w2_only_per_seed.csv", flush=True)

    return all_seed_results, summary


if __name__ == "__main__":
    run_smolgpt_w2_benchmark()
