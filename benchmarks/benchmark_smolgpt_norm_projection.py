"""
Rigorous Weight Trajectory Forecasting Benchmark: Weight Norm Regularization & Basin Projection (Choice B).
Evaluates Global Norm Matching, Tensor-Wise Norm Matching, and Oracle Radial Decay Tracking on Full SmolGPT.
Tests whether constraining weight norms to the reference manifold eliminates the catastrophic blowout.
"""

import os
import sys
import json
import time
import math
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


def project_global_norm(w_cand_t: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Rescales the full weight vector to match target_norm."""
    cand_norm = torch.norm(w_cand_t)
    if cand_norm > 1e-8:
        return w_cand_t * (target_norm / cand_norm)
    return w_cand_t


def project_tensor_wise_norm(model: nn.Module, w_cand_t: torch.Tensor, w_ref_t: torch.Tensor) -> torch.Tensor:
    """Rescales each parameter tensor individually so its Frobenius norm matches the reference tensor norm."""
    offset = 0
    out_parts = []
    for p in model.parameters():
        numel = p.numel()
        cand_p = w_cand_t[offset:offset + numel]
        ref_p = w_ref_t[offset:offset + numel]
        c_norm = torch.norm(cand_p)
        r_norm = torch.norm(ref_p)
        if c_norm > 1e-8:
            out_parts.append(cand_p * (r_norm / c_norm))
        else:
            out_parts.append(cand_p)
        offset += numel
    return torch.cat(out_parts)


def compute_displacement_and_weight_cosine(w_forecast, w_actual, w_ref):
    """
    Independently computes:
    1. Forecast displacement cosine: cos(w_forecast - w_ref, w_actual - w_ref)
    2. Full weight-direction cosine: cos(w_forecast, w_actual)
    """
    if isinstance(w_forecast, torch.Tensor):
        w_f = w_forecast.detach().cpu().numpy().flatten()
    else:
        w_f = np.asarray(w_forecast).flatten()
    if isinstance(w_actual, torch.Tensor):
        w_a = w_actual.detach().cpu().numpy().flatten()
    else:
        w_a = np.asarray(w_actual).flatten()
    if isinstance(w_ref, torch.Tensor):
        w_r = w_ref.detach().cpu().numpy().flatten()
    else:
        w_r = np.asarray(w_ref).flatten()

    # Weight cosine
    norm_f = float(np.linalg.norm(w_f))
    norm_a = float(np.linalg.norm(w_a))
    cos_weight = float(np.dot(w_f, w_a) / (norm_f * norm_a + 1e-12))

    # Displacement cosine
    d_f = w_f - w_r
    d_a = w_a - w_r
    norm_df = float(np.linalg.norm(d_f))
    norm_da = float(np.linalg.norm(d_a))
    if norm_df > 1e-8 and norm_da > 1e-8:
        cos_disp = float(np.dot(d_f, d_a) / (norm_df * norm_da))
    else:
        cos_disp = 0.0

    return cos_disp, cos_weight


def run_smolgpt_norm_benchmark(
    seeds=[42, 43, 44],
    cutoff_step=1000,
    context_checkpoints=32,
    checkpoint_interval=20,
    ranks=[1, 2],
    horizons=[8, 16, 32, 64],  # +160, +320, +640, +1280 steps
    batch_size=16,
    block_size=128,
    lr=1e-3,
    weight_decay=0.01,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" RIGOROUS SMOLGPT WEIGHT NORM REGULARIZATION BENCHMARK (Choice B)")
    print(f" Tests: Global Norm Matching, Tensor-Wise Matching, and Oracle Radial Decay Tracking")
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
        checkpoints_history = []  # shape: (P, context_checkpoints)
        future_checkpoints = {}   # h -> dict

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
                    w_t = get_flat_params(model).cpu().numpy().copy()
                    checkpoints_history.append(w_t)
                    if step == cutoff_step:
                        w_ref_full = w_t.copy()

            # Record future continuation checkpoints
            if step > cutoff_step:
                steps_after = step - cutoff_step
                if steps_after % checkpoint_interval == 0:
                    h_curr = steps_after // checkpoint_interval
                    if h_curr in horizons:
                        w_future_full = get_flat_params(model).cpu().numpy().copy()
                        future_checkpoints[h_curr] = {
                            "w_full": w_future_full,
                            "cum_step_time": float(np.sum(step_times[cutoff_step:step]))
                        }

            if step % 200 == 0:
                v_loss, v_ppl = evaluate_transformer_loss_and_ppl(model, val_batches)
                print(f"  Step {step:4d}/{total_steps} | Batch Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f} | PPL: {v_ppl:.2f}", flush=True)

        train_elapsed = time.perf_counter() - t_train_start
        print(f"Training completed in {train_elapsed:.2f} s ({total_steps / train_elapsed:.1f} steps/s)", flush=True)

        weight_window = np.column_stack(checkpoints_history)  # (1246080, 32)
        w_ref = weight_window[:, -1].copy()
        w_ref_t = torch.from_numpy(w_ref).float()
        ref_global_norm = float(np.linalg.norm(w_ref))

        engine = LowRankTrajectoryEngine(weight_window, ref_checkpoint=w_ref)

        def reset_to_frozen_ref():
            set_flat_params(model, w_ref_t)

        # Evaluate Frozen Checkpoint
        reset_to_frozen_ref()
        frozen_metrics = eval_both(model)

        # Evaluate Matched SWA (over last 16 checkpoints)
        reset_to_frozen_ref()
        t0_swa = time.perf_counter()
        w_swa = np.mean(weight_window[:, -16:], axis=1)
        set_flat_params(model, torch.from_numpy(w_swa).float())
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_swa_load = time.perf_counter() - t0_swa
        swa_metrics = eval_both(model)

        seed_data = {
            "seed": seed,
            "cutoff_step": cutoff_step,
            "ref_global_norm": ref_global_norm,
            "frozen": frozen_metrics,
            "swa": swa_metrics,
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
            w_act_t = torch.from_numpy(w_actual_full).float()
            act_global_norm = float(np.linalg.norm(w_actual_full))
            adamw_time = future_checkpoints[h]["cum_step_time"]

            # 1. Full Continued AdamW
            set_flat_params(model, w_act_t)
            full_adamw_metrics = eval_both(model)
            h_data["methods"]["full_continued_adamw"] = {
                **full_adamw_metrics,
                "weight_norm": act_global_norm,
                "cos_sim": 1.0,
                "cos_sim_displacement": 1.0,
                "cos_sim_weight": 1.0,
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

            # 2. Frozen Checkpoint Baseline
            reset_to_frozen_ref()
            h_data["methods"]["frozen_checkpoint"] = {
                **frozen_metrics,
                "weight_norm": ref_global_norm,
                "cos_sim": 0.0,
                "cos_sim_displacement": 0.0,
                "cos_sim_weight": compute_displacement_and_weight_cosine(w_ref, w_actual_full, w_ref)[1],
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

            # 3. Matched SWA
            reset_to_frozen_ref()
            set_flat_params(model, torch.from_numpy(w_swa).float())
            cos_disp_swa, cos_weight_swa = compute_displacement_and_weight_cosine(w_swa, w_actual_full, w_ref)
            h_data["methods"]["matched_swa"] = {
                **swa_metrics,
                "weight_norm": float(np.linalg.norm(w_swa)),
                "cos_sim": cos_disp_swa,
                "cos_sim_displacement": cos_disp_swa,
                "cos_sim_weight": cos_weight_swa,
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

            # Evaluate Ranks
            for r in ranks:
                # ----------------- Linear -----------------
                w_lin_raw, c_lin, t_lin_f = engine.forecast_linear(r, h)
                w_lin_raw_np = w_lin_raw.numpy()

                # 4a. Linear Unprojected
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_flat_params(model, w_lin_raw)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lin_load = time.perf_counter() - t0_load
                lin_raw_metrics = eval_both(model)
                cos_disp_lin_raw, cos_weight_lin_raw = compute_displacement_and_weight_cosine(w_lin_raw_np, w_actual_full, w_ref)
                t_lin_raw_total = engine.prep_time + engine.svd_time + t_lin_f + t_lin_load

                h_data["methods"][f"linear_unprojected_r{r}"] = {
                    **lin_raw_metrics,
                    "weight_norm": float(np.linalg.norm(w_lin_raw_np)),
                    "cos_sim": cos_disp_lin_raw,
                    "cos_sim_displacement": cos_disp_lin_raw,
                    "cos_sim_weight": cos_weight_lin_raw,
                    "time_ms": t_lin_raw_total * 1000.0,
                    "speedup": float(adamw_time / max(t_lin_raw_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_lin_f * 1000.0,
                        "reconstruction_and_load_ms": t_lin_load * 1000.0,
                        "total_latency_ms": t_lin_raw_total * 1000.0
                    }
                }

                # 4b. Linear Global Norm Matched (||W|| = ||W_ref||)
                t0_norm = time.perf_counter()
                w_lin_global = project_global_norm(w_lin_raw, ref_global_norm)
                reset_to_frozen_ref()
                set_flat_params(model, w_lin_global)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lin_glob_load = time.perf_counter() - t0_norm
                lin_global_metrics = eval_both(model)
                cos_disp_lin_global, cos_weight_lin_global = compute_displacement_and_weight_cosine(w_lin_global, w_actual_full, w_ref)
                t_lin_glob_total = engine.prep_time + engine.svd_time + t_lin_f + t_lin_glob_load

                h_data["methods"][f"linear_global_norm_r{r}"] = {
                    **lin_global_metrics,
                    "weight_norm": float(torch.norm(w_lin_global).item()),
                    "cos_sim": cos_disp_lin_global,
                    "cos_sim_displacement": cos_disp_lin_global,
                    "cos_sim_weight": cos_weight_lin_global,
                    "time_ms": t_lin_glob_total * 1000.0,
                    "speedup": float(adamw_time / max(t_lin_glob_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_lin_f * 1000.0,
                        "reconstruction_and_load_ms": t_lin_glob_load * 1000.0,
                        "total_latency_ms": t_lin_glob_total * 1000.0
                    }
                }

                # 4c. Linear Tensor-Wise Norm Matched (||p_i|| = ||p_{i,ref}||)
                t0_tensor = time.perf_counter()
                w_lin_tensor = project_tensor_wise_norm(model, w_lin_raw, w_ref_t)
                reset_to_frozen_ref()
                set_flat_params(model, w_lin_tensor)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lin_tens_load = time.perf_counter() - t0_tensor
                lin_tensor_metrics = eval_both(model)
                cos_disp_lin_tensor, cos_weight_lin_tensor = compute_displacement_and_weight_cosine(w_lin_tensor, w_actual_full, w_ref)
                t_lin_tens_total = engine.prep_time + engine.svd_time + t_lin_f + t_lin_tens_load

                h_data["methods"][f"linear_tensor_norm_r{r}"] = {
                    **lin_tensor_metrics,
                    "weight_norm": float(torch.norm(w_lin_tensor).item()),
                    "cos_sim": cos_disp_lin_tensor,
                    "cos_sim_displacement": cos_disp_lin_tensor,
                    "cos_sim_weight": cos_weight_lin_tensor,
                    "time_ms": t_lin_tens_total * 1000.0,
                    "speedup": float(adamw_time / max(t_lin_tens_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_lin_f * 1000.0,
                        "reconstruction_and_load_ms": t_lin_tens_load * 1000.0,
                        "total_latency_ms": t_lin_tens_total * 1000.0
                    }
                }

                # 4d. Linear Oracle Radial Norm Matched (||W|| = ||W_actual||)
                t0_oracle = time.perf_counter()
                w_lin_oracle = project_global_norm(w_lin_raw, act_global_norm)
                reset_to_frozen_ref()
                set_flat_params(model, w_lin_oracle)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lin_ora_load = time.perf_counter() - t0_oracle
                lin_oracle_metrics = eval_both(model)
                cos_disp_lin_oracle, cos_weight_lin_oracle = compute_displacement_and_weight_cosine(w_lin_oracle, w_actual_full, w_ref)
                t_lin_ora_total = engine.prep_time + engine.svd_time + t_lin_f + t_lin_ora_load

                h_data["methods"][f"linear_oracle_norm_r{r}"] = {
                    **lin_oracle_metrics,
                    "weight_norm": float(torch.norm(w_lin_oracle).item()),
                    "cos_sim": cos_disp_lin_oracle,
                    "cos_sim_displacement": cos_disp_lin_oracle,
                    "cos_sim_weight": cos_weight_lin_oracle,
                    "time_ms": t_lin_ora_total * 1000.0,
                    "speedup": float(adamw_time / max(t_lin_ora_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_lin_f * 1000.0,
                        "reconstruction_and_load_ms": t_lin_ora_load * 1000.0,
                        "total_latency_ms": t_lin_ora_total * 1000.0
                    }
                }

                # ----------------- TimesFM -----------------
                w_tfm_raw, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device)
                w_tfm_raw_np = w_tfm_raw.numpy()

                # 5a. TimesFM Unprojected
                t0_tfm_load = time.perf_counter()
                reset_to_frozen_ref()
                set_flat_params(model, w_tfm_raw)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_tfm_load_dt = time.perf_counter() - t0_tfm_load
                tfm_raw_metrics = eval_both(model)
                cos_disp_tfm_raw, cos_weight_tfm_raw = compute_displacement_and_weight_cosine(w_tfm_raw_np, w_actual_full, w_ref)
                t_tfm_raw_total = engine.prep_time + engine.svd_time + t_tfm_f + t_tfm_r + t_tfm_load_dt

                h_data["methods"][f"timesfm_unprojected_r{r}"] = {
                    **tfm_raw_metrics,
                    "weight_norm": float(np.linalg.norm(w_tfm_raw_np)),
                    "cos_sim": cos_disp_tfm_raw,
                    "cos_sim_displacement": cos_disp_tfm_raw,
                    "cos_sim_weight": cos_weight_tfm_raw,
                    "time_ms": t_tfm_raw_total * 1000.0,
                    "speedup": float(adamw_time / max(t_tfm_raw_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r + t_tfm_load_dt) * 1000.0,
                        "total_latency_ms": t_tfm_raw_total * 1000.0
                    }
                }

                # 5b. TimesFM Global Norm Matched (||W|| = ||W_ref||)
                t0_tfm_glob = time.perf_counter()
                w_tfm_global = project_global_norm(w_tfm_raw, ref_global_norm)
                reset_to_frozen_ref()
                set_flat_params(model, w_tfm_global)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_tfm_glob_load = time.perf_counter() - t0_tfm_glob
                tfm_global_metrics = eval_both(model)
                cos_disp_tfm_global, cos_weight_tfm_global = compute_displacement_and_weight_cosine(w_tfm_global, w_actual_full, w_ref)
                t_tfm_glob_total = engine.prep_time + engine.svd_time + t_tfm_f + t_tfm_r + t_tfm_glob_load

                h_data["methods"][f"timesfm_global_norm_r{r}"] = {
                    **tfm_global_metrics,
                    "weight_norm": float(torch.norm(w_tfm_global).item()),
                    "cos_sim": cos_disp_tfm_global,
                    "cos_sim_displacement": cos_disp_tfm_global,
                    "cos_sim_weight": cos_weight_tfm_global,
                    "time_ms": t_tfm_glob_total * 1000.0,
                    "speedup": float(adamw_time / max(t_tfm_glob_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r + t_tfm_glob_load) * 1000.0,
                        "total_latency_ms": t_tfm_glob_total * 1000.0
                    }
                }

                # 5c. TimesFM Tensor-Wise Norm Matched (||p_i|| = ||p_{i,ref}||)
                t0_tfm_tens = time.perf_counter()
                w_tfm_tensor = project_tensor_wise_norm(model, w_tfm_raw, w_ref_t)
                reset_to_frozen_ref()
                set_flat_params(model, w_tfm_tensor)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_tfm_tens_load = time.perf_counter() - t0_tfm_tens
                tfm_tensor_metrics = eval_both(model)
                cos_disp_tfm_tensor, cos_weight_tfm_tensor = compute_displacement_and_weight_cosine(w_tfm_tensor, w_actual_full, w_ref)
                t_tfm_tens_total = engine.prep_time + engine.svd_time + t_tfm_f + t_tfm_r + t_tfm_tens_load

                h_data["methods"][f"timesfm_tensor_norm_r{r}"] = {
                    **tfm_tensor_metrics,
                    "weight_norm": float(torch.norm(w_tfm_tensor).item()),
                    "cos_sim": cos_disp_tfm_tensor,
                    "cos_sim_displacement": cos_disp_tfm_tensor,
                    "cos_sim_weight": cos_weight_tfm_tensor,
                    "time_ms": t_tfm_tens_total * 1000.0,
                    "speedup": float(adamw_time / max(t_tfm_tens_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r + t_tfm_tens_load) * 1000.0,
                        "total_latency_ms": t_tfm_tens_total * 1000.0
                    }
                }

                # 5d. TimesFM Oracle Radial Norm Matched (||W|| = ||W_actual||)
                t0_tfm_ora = time.perf_counter()
                w_tfm_oracle = project_global_norm(w_tfm_raw, act_global_norm)
                reset_to_frozen_ref()
                set_flat_params(model, w_tfm_oracle)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_tfm_ora_load = time.perf_counter() - t0_tfm_ora
                tfm_oracle_metrics = eval_both(model)
                cos_disp_tfm_oracle, cos_weight_tfm_oracle = compute_displacement_and_weight_cosine(w_tfm_oracle, w_actual_full, w_ref)
                t_tfm_ora_total = engine.prep_time + engine.svd_time + t_tfm_f + t_tfm_r + t_tfm_ora_load

                h_data["methods"][f"timesfm_oracle_norm_r{r}"] = {
                    **tfm_oracle_metrics,
                    "weight_norm": float(torch.norm(w_tfm_oracle).item()),
                    "cos_sim": cos_disp_tfm_oracle,
                    "cos_sim_displacement": cos_disp_tfm_oracle,
                    "cos_sim_weight": cos_weight_tfm_oracle,
                    "time_ms": t_tfm_ora_total * 1000.0,
                    "speedup": float(adamw_time / max(t_tfm_ora_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine.prep_time * 1000.0,
                        "svd_ms": engine.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r + t_tfm_ora_load) * 1000.0,
                        "total_latency_ms": t_tfm_ora_total * 1000.0
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
            "weight_norm": calc_stats([s["ref_global_norm"] for s in all_seed_results]),
        },
        "swa": {
            "train_loss": calc_stats([s["swa"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["swa"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["swa"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["swa"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["swa"]["gen_gap"] for s in all_seed_results]),
        },
        "horizons": {}
    }

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
            norm_vals = [s["horizons"][h_key]["methods"][method]["weight_norm"] for s in all_seed_results]
            cos_vals = [s["horizons"][h_key]["methods"][method]["cos_sim"] for s in all_seed_results]
            cos_weight_vals = [s["horizons"][h_key]["methods"][method].get("cos_sim_weight", 0.0) for s in all_seed_results]
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
                "weight_norm": calc_stats(norm_vals),
                "cos_sim": calc_stats(cos_vals),
                "cos_sim_displacement": calc_stats(cos_vals),
                "cos_sim_weight": calc_stats(cos_weight_vals),
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
    with open("results/smolgpt_norm_benchmark_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open("results/smolgpt_norm_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/smolgpt_norm_benchmark_per_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "horizon", "equivalent_steps", "method", "train_loss", "train_ppl", "val_loss", "val_ppl", "gen_gap", "weight_norm", "cos_sim", "time_ms", "speedup"])
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
                        f"{m_data['weight_norm']:.4f}",
                        f"{m_data['cos_sim']:.4f}",
                        f"{m_data['time_ms']:.2f}",
                        f"{m_data['speedup']:.2f}"
                    ])

    print("Saved results to results/smolgpt_norm_benchmark_results.json", flush=True)
    print("Saved summary to results/smolgpt_norm_benchmark_summary.json", flush=True)
    print("Saved CSV to results/smolgpt_norm_benchmark_per_seed.csv", flush=True)

    return all_seed_results, summary


if __name__ == "__main__":
    run_smolgpt_norm_benchmark()
