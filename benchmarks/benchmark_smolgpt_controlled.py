import os
import sys
import time
import copy
import json
import csv
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models_transformer import (
    SmolGPT,
    SmolGPTConfig,
    evaluate_transformer_loss_and_ppl,
    get_flat_params,
    set_flat_params,
    get_mlp_params,
    set_mlp_params,
    get_mlp_w2_params,
    set_mlp_w2_params
)
from src.data_transformer import (
    load_tinystories_dataset,
    get_random_batch,
    get_deterministic_val_batches,
    get_deterministic_train_batches
)
from src.low_rank import LowRankTrajectoryEngine
from transformers import TimesFmModelForPrediction

# ==============================================================================
# Helper Functions
# ==============================================================================

def make_batch_from_indices(tokens: np.ndarray, indices: np.ndarray, block_size: int, device: torch.device):
    x_np = np.stack([tokens[i : i + block_size].astype(np.int64) for i in indices])
    y_np = np.stack([tokens[i + 1 : i + block_size + 1].astype(np.int64) for i in indices])
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)

def calc_stats(vals):
    vals = np.array(vals, dtype=np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    se = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    ci95 = float(se * stats.t.ppf(0.975, df=len(vals) - 1)) if len(vals) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95": ci95,
        "min": float(np.min(vals)),
        "max": float(np.max(vals))
    }

def calc_paired_diff_stats(method_vals, baseline_vals):
    diffs = np.array(method_vals, dtype=np.float64) - np.array(baseline_vals, dtype=np.float64)
    res = calc_stats(diffs)
    if len(diffs) > 1 and np.std(diffs) > 1e-12:
        t_stat, p_val = stats.ttest_rel(method_vals, baseline_vals)
        res["p_val"] = float(p_val)
        res["statistically_significant"] = bool(p_val < 0.05)
    else:
        res["p_val"] = 1.0
        res["statistically_significant"] = False
    return res

# ==============================================================================
# Main Benchmark Runner
# ==============================================================================

def run_controlled_benchmark(
    seeds=[42, 43, 44],
    cutoff_step=1000,
    context_checkpoints=32,
    checkpoint_interval=20,
    horizons=[8, 16, 32, 64],
    batch_size=16,
    block_size=128,
    lr=1e-3,
    weight_decay=0.01,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" SCIENTIFICALLY CONTROLLED SMOLGPT CONTINUATION BENCHMARK")
    print(f" Cutoff: Step {cutoff_step} | Horizons: {horizons} | Interval: {checkpoint_interval}")
    print(f" Seeds: {seeds} | Device: {device}")
    print(" Controlled Conditions: Full vs MLP vs W2 x Resumed vs Restarted AdamW")
    print("=" * 80, flush=True)

    # 1. Load Data
    print("Loading TinyStories token dataset...", flush=True)
    train_tokens, val_tokens = load_tinystories_dataset("data/tinystories/tinystories_sub.npz")
    val_batches = get_deterministic_val_batches(val_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device)
    train_batches = get_deterministic_train_batches(train_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device)

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

    # 2. Load TimesFM
    t_tfm_start = time.perf_counter()
    print("Loading TimesFM-2.0-500M...", flush=True)
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()
    timesfm_load_ms = (time.perf_counter() - t_tfm_start) * 1000.0
    print(f"TimesFM loaded in {timesfm_load_ms:.2f} ms", flush=True)

    max_horizon = max(horizons)
    total_steps = cutoff_step + (max_horizon * checkpoint_interval)
    start_context_step = cutoff_step - (context_checkpoints * checkpoint_interval)
    max_idx = len(train_tokens) - block_size - 1

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        t_seed_start = time.perf_counter()
        print(f"\n{'='*40}\n>>> Running Seed {seed} ({seed_idx + 1}/{len(seeds)})\n{'='*40}", flush=True)

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
        base_model = SmolGPT(config).to(device)
        base_opt = optim.AdamW(base_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

        # Pre-generate future minibatch indices for steps 1001 to total_steps
        # This guarantees 100% identical future batches across all branches
        future_batch_indices = {}

        # Context checkpoint buffers
        full_checkpoints_history = []
        mlp_checkpoints_history = []
        w2_checkpoints_history = []

        print(f"Phase 1: Base Training to Cutoff Step {cutoff_step}...", flush=True)
        for step in range(1, cutoff_step + 1):
            base_model.train()
            x, y = get_random_batch(train_tokens, batch_size=batch_size, block_size=block_size, device=device)
            _, loss = base_model(x, y)
            base_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            base_opt.step()

            if start_context_step < step <= cutoff_step:
                if (step - start_context_step) % checkpoint_interval == 0:
                    full_checkpoints_history.append(get_flat_params(base_model).cpu().numpy().copy())
                    mlp_checkpoints_history.append(get_mlp_params(base_model).cpu().numpy().copy())
                    w2_checkpoints_history.append(get_mlp_w2_params(base_model).cpu().numpy().copy())

        # Generate future minibatch indices after cutoff step
        for step in range(cutoff_step + 1, total_steps + 1):
            future_batch_indices[step] = np.random.randint(0, max_idx, size=batch_size)

        # Snapshot cutoff state
        cutoff_model_state = copy.deepcopy(base_model.state_dict())
        cutoff_opt_state = copy.deepcopy(base_opt.state_dict())
        param_name_list = [name for name, _ in base_model.named_parameters()]
        name_to_adam_state = {
            name: copy.deepcopy(cutoff_opt_state['state'][idx])
            for idx, name in enumerate(param_name_list)
            if idx in cutoff_opt_state['state']
        }

        w_ref_full = get_flat_params(base_model).cpu().numpy().copy()
        w_ref_mlp = get_mlp_params(base_model).cpu().numpy().copy()
        w_ref_w2 = get_mlp_w2_params(base_model).cpu().numpy().copy()
        frozen_metrics = eval_both(base_model)
        print(f"Cutoff Step {cutoff_step} reached | Frozen Val Loss: {frozen_metrics['val_loss']:.4f} | Val PPL: {frozen_metrics['val_ppl']:.2f}")

        # ----------------------------------------------------------------------
        # Continuation Branches
        # ----------------------------------------------------------------------
        branch_configs = [
            ("full_resumed", "all", "resumed", 1246080),
            ("full_restarted", "all", "restarted", 1246080),
            ("mlp_resumed", "mlp", "resumed", 523776),
            ("mlp_restarted", "mlp", "restarted", 523776),
            ("w2_resumed", "mlp.w2", "resumed", 174592),
            ("w2_restarted", "mlp.w2", "restarted", 174592)
        ]

        branch_checkpoints = {} # branch_name -> {h: {"model_state": ..., "metrics": ..., "cum_time": ...}}

        for branch_name, target_pat, opt_mode, n_params in branch_configs:
            print(f"  Branch [{branch_name}]: {target_pat} ({n_params:,} params) | AdamW: {opt_mode}...", flush=True)
            b_model = SmolGPT(config).to(device)
            b_model.load_state_dict(cutoff_model_state)

            for name, param in b_model.named_parameters():
                if target_pat == "all":
                    param.requires_grad = True
                elif target_pat == "mlp":
                    param.requires_grad = ("mlp" in name)
                elif target_pat == "mlp.w2":
                    param.requires_grad = ("mlp.w2" in name)

            trainable_params = [p for p in b_model.parameters() if p.requires_grad]
            trainable_names = [n for n, p in b_model.named_parameters() if p.requires_grad]
            assert sum(p.numel() for p in trainable_params) == n_params, f"Param count mismatch in {branch_name}"

            b_opt = optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

            if opt_mode == "resumed":
                if target_pat == "all":
                    b_opt.load_state_dict(cutoff_opt_state)
                else:
                    for p, name in zip(trainable_params, trainable_names):
                        if name in name_to_adam_state:
                            st = name_to_adam_state[name]
                            b_opt.state[p] = {
                                "step": st["step"].clone(),
                                "exp_avg": st["exp_avg"].clone(),
                                "exp_avg_sq": st["exp_avg_sq"].clone()
                            }

            branch_checkpoints[branch_name] = {}
            step_times = []
            t_branch_start = time.perf_counter()

            for step in range(cutoff_step + 1, total_steps + 1):
                t_s0 = time.perf_counter()
                b_model.train()
                indices = future_batch_indices[step]
                x, y = make_batch_from_indices(train_tokens, indices, block_size, device)
                _, loss = b_model(x, y)
                b_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                b_opt.step()
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t_s0)

                steps_after = step - cutoff_step
                if steps_after % checkpoint_interval == 0:
                    h_curr = steps_after // checkpoint_interval
                    if h_curr in horizons:
                        # Regression assertion: verify frozen parameters bitwise unchanged
                        for name, param in b_model.named_parameters():
                            if not param.requires_grad:
                                assert torch.equal(param.data, cutoff_model_state[name]), f"Bitwise leakage detected in {name} of {branch_name}!"

                        b_metrics = eval_both(b_model)
                        cum_time = float(np.sum(step_times))
                        branch_checkpoints[branch_name][h_curr] = {
                            "flat_params": get_flat_params(b_model).cpu().numpy().copy(),
                            "mlp_params": get_mlp_params(b_model).cpu().numpy().copy(),
                            "w2_params": get_mlp_w2_params(b_model).cpu().numpy().copy(),
                            "metrics": b_metrics,
                            "cum_time": cum_time
                        }

        # ----------------------------------------------------------------------
        # Low-Rank Trajectory Engines (for W2 and MLP)
        # ----------------------------------------------------------------------
        w2_window = np.column_stack(w2_checkpoints_history) # (174592, 32)
        mlp_window = np.column_stack(mlp_checkpoints_history) # (523776, 32)
        engine_w2 = LowRankTrajectoryEngine(w2_window, ref_checkpoint=w_ref_w2)
        engine_mlp = LowRankTrajectoryEngine(mlp_window, ref_checkpoint=w_ref_mlp)

        # SWA and EMA on history
        w_w2_swa = np.mean(w2_window[:, -8:], axis=1)
        w_mlp_swa = np.mean(mlp_window[:, -8:], axis=1)

        alpha = 2.0 / (8 + 1)
        w_w2_ema = w2_window[:, -8].copy()
        w_mlp_ema = mlp_window[:, -8].copy()
        for i in range(-7, 0):
            w_w2_ema = alpha * w2_window[:, i] + (1 - alpha) * w_w2_ema
            w_mlp_ema = alpha * mlp_window[:, i] + (1 - alpha) * w_mlp_ema

        # ----------------------------------------------------------------------
        # Assemble Unified Results across Horizons
        # ----------------------------------------------------------------------
        seed_data = {"seed": seed, "horizons": {}}

        def reset_to_frozen_ref():
            set_flat_params(base_model, torch.from_numpy(w_ref_full).float())

        for h in horizons:
            h_data = {"step": cutoff_step + h * checkpoint_interval, "methods": {}}
            full_resumed_time = branch_checkpoints["full_resumed"][h]["cum_time"]
            full_resumed_ppl = branch_checkpoints["full_resumed"][h]["metrics"]["val_ppl"]

            # 1. Training Branches
            for branch_name, target_pat, opt_mode, n_params in branch_configs:
                b_info = branch_checkpoints[branch_name][h]
                h_data["methods"][branch_name] = {
                    **b_info["metrics"],
                    "trainable_params": n_params,
                    "optimizer_mode": opt_mode,
                    "time_ms": b_info["cum_time"] * 1000.0,
                    "diff_ppl_vs_full_resumed": b_info["metrics"]["val_ppl"] - full_resumed_ppl,
                    "speedup": float(full_resumed_time / max(b_info["cum_time"], 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": 0.0,
                        "svd_ms": 0.0,
                        "forecast_ms": 0.0,
                        "reconstruction_and_load_ms": 0.0,
                        "total_latency_ms": b_info["cum_time"] * 1000.0
                    }
                }

            # 2. Frozen Checkpoint Baseline
            reset_to_frozen_ref()
            h_data["methods"]["frozen_checkpoint"] = {
                **frozen_metrics,
                "trainable_params": 0,
                "optimizer_mode": "none",
                "time_ms": 0.0,
                "diff_ppl_vs_full_resumed": frozen_metrics["val_ppl"] - full_resumed_ppl,
                "speedup": 1.0,
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": 0.0
                }
            }

            # 3. Oracle Transplants (Preserved Historical Conditions)
            # W2 transplant: take W2 from full_resumed at horizon h, insert into frozen checkpoint
            reset_to_frozen_ref()
            w_act_w2 = branch_checkpoints["full_resumed"][h]["w2_params"]
            set_mlp_w2_params(base_model, torch.from_numpy(w_act_w2).float())
            w2_trans_metrics = eval_both(base_model)
            h_data["methods"]["oracle_w2_transplant"] = {
                **w2_trans_metrics,
                "trainable_params": 0,
                "optimizer_mode": "oracle_transplant",
                "time_ms": full_resumed_time * 1000.0,
                "diff_ppl_vs_full_resumed": w2_trans_metrics["val_ppl"] - full_resumed_ppl,
                "speedup": 1.0,
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": full_resumed_time * 1000.0
                }
            }

            # MLP transplant: take MLP from full_resumed at horizon h, insert into frozen checkpoint
            reset_to_frozen_ref()
            w_act_mlp = branch_checkpoints["full_resumed"][h]["mlp_params"]
            set_mlp_params(base_model, torch.from_numpy(w_act_mlp).float())
            mlp_trans_metrics = eval_both(base_model)
            h_data["methods"]["oracle_mlp_transplant"] = {
                **mlp_trans_metrics,
                "trainable_params": 0,
                "optimizer_mode": "oracle_transplant",
                "time_ms": full_resumed_time * 1000.0,
                "diff_ppl_vs_full_resumed": mlp_trans_metrics["val_ppl"] - full_resumed_ppl,
                "speedup": 1.0,
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": 0.0,
                    "total_latency_ms": full_resumed_time * 1000.0
                }
            }

            # 4. SWA & EMA Baselines
            t0_load = time.perf_counter()
            reset_to_frozen_ref()
            set_mlp_w2_params(base_model, torch.from_numpy(w_w2_swa).float())
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_swa_load = time.perf_counter() - t0_load
            swa_w2_metrics = eval_both(base_model)
            h_data["methods"]["w2_matched_swa"] = {
                **swa_w2_metrics,
                "trainable_params": 0,
                "optimizer_mode": "none",
                "time_ms": t_swa_load * 1000.0,
                "diff_ppl_vs_full_resumed": swa_w2_metrics["val_ppl"] - full_resumed_ppl,
                "speedup": float(full_resumed_time / max(t_swa_load, 1e-6)),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": t_swa_load * 1000.0,
                    "total_latency_ms": t_swa_load * 1000.0
                }
            }

            t0_load = time.perf_counter()
            reset_to_frozen_ref()
            set_mlp_params(base_model, torch.from_numpy(w_mlp_swa).float())
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_swa_mlp_load = time.perf_counter() - t0_load
            swa_mlp_metrics = eval_both(base_model)
            h_data["methods"]["mlp_matched_swa"] = {
                **swa_mlp_metrics,
                "trainable_params": 0,
                "optimizer_mode": "none",
                "time_ms": t_swa_mlp_load * 1000.0,
                "diff_ppl_vs_full_resumed": swa_mlp_metrics["val_ppl"] - full_resumed_ppl,
                "speedup": float(full_resumed_time / max(t_swa_mlp_load, 1e-6)),
                "timing_breakdown": {
                    "checkpoint_prep_ms": 0.0,
                    "svd_ms": 0.0,
                    "forecast_ms": 0.0,
                    "reconstruction_and_load_ms": t_swa_mlp_load * 1000.0,
                    "total_latency_ms": t_swa_mlp_load * 1000.0
                }
            }

            # 5. Trajectory Forecasting Baselines (Linear & TimesFM on W2 & MLP)
            for r in [2, 16]:
                # W2 Linear
                w_lin_w2, _, t_lin_w2 = engine_w2.forecast_linear(r, h)
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(base_model, w_lin_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_load_w2 = time.perf_counter() - t0_load
                lin_w2_metrics = eval_both(base_model)
                t_lin_w2_total = engine_w2.prep_time + engine_w2.svd_time + t_lin_w2 + t_load_w2
                h_data["methods"][f"w2_linear_r{r}"] = {
                    **lin_w2_metrics,
                    "trainable_params": 0,
                    "optimizer_mode": "none",
                    "time_ms": t_lin_w2_total * 1000.0,
                    "diff_ppl_vs_full_resumed": lin_w2_metrics["val_ppl"] - full_resumed_ppl,
                    "speedup": float(full_resumed_time / max(t_lin_w2_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine_w2.prep_time * 1000.0,
                        "svd_ms": engine_w2.svd_time * 1000.0,
                        "forecast_ms": t_lin_w2 * 1000.0,
                        "reconstruction_and_load_ms": t_load_w2 * 1000.0,
                        "total_latency_ms": t_lin_w2_total * 1000.0
                    }
                }

                # MLP Linear
                w_lin_mlp, _, t_lin_mlp = engine_mlp.forecast_linear(r, h)
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_params(base_model, w_lin_mlp)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_load_mlp = time.perf_counter() - t0_load
                lin_mlp_metrics = eval_both(base_model)
                t_lin_mlp_total = engine_mlp.prep_time + engine_mlp.svd_time + t_lin_mlp + t_load_mlp
                h_data["methods"][f"mlp_linear_r{r}"] = {
                    **lin_mlp_metrics,
                    "trainable_params": 0,
                    "optimizer_mode": "none",
                    "time_ms": t_lin_mlp_total * 1000.0,
                    "diff_ppl_vs_full_resumed": lin_mlp_metrics["val_ppl"] - full_resumed_ppl,
                    "speedup": float(full_resumed_time / max(t_lin_mlp_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine_mlp.prep_time * 1000.0,
                        "svd_ms": engine_mlp.svd_time * 1000.0,
                        "forecast_ms": t_lin_mlp * 1000.0,
                        "reconstruction_and_load_ms": t_load_mlp * 1000.0,
                        "total_latency_ms": t_lin_mlp_total * 1000.0
                    }
                }

                # W2 TimesFM
                w_tfm_w2, _, t_tfm_f_w2, t_tfm_r_w2 = engine_w2.forecast_timesfm(r, h, tfm_model, device)
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_w2_params(base_model, w_tfm_w2)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_load_tfm_w2 = time.perf_counter() - t0_load
                tfm_w2_metrics = eval_both(base_model)
                t_tfm_w2_total = engine_w2.prep_time + engine_w2.svd_time + t_tfm_f_w2 + t_tfm_r_w2 + t_load_tfm_w2
                h_data["methods"][f"w2_timesfm_r{r}"] = {
                    **tfm_w2_metrics,
                    "trainable_params": 0,
                    "optimizer_mode": "none",
                    "time_ms": t_tfm_w2_total * 1000.0,
                    "diff_ppl_vs_full_resumed": tfm_w2_metrics["val_ppl"] - full_resumed_ppl,
                    "speedup": float(full_resumed_time / max(t_tfm_w2_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine_w2.prep_time * 1000.0,
                        "svd_ms": engine_w2.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f_w2 * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r_w2 + t_load_tfm_w2) * 1000.0,
                        "total_latency_ms": t_tfm_w2_total * 1000.0
                    }
                }

                # MLP TimesFM
                w_tfm_mlp, _, t_tfm_f_mlp, t_tfm_r_mlp = engine_mlp.forecast_timesfm(r, h, tfm_model, device)
                t0_load = time.perf_counter()
                reset_to_frozen_ref()
                set_mlp_params(base_model, w_tfm_mlp)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_load_tfm_mlp = time.perf_counter() - t0_load
                tfm_mlp_metrics = eval_both(base_model)
                t_tfm_mlp_total = engine_mlp.prep_time + engine_mlp.svd_time + t_tfm_f_mlp + t_tfm_r_mlp + t_load_tfm_mlp
                h_data["methods"][f"mlp_timesfm_r{r}"] = {
                    **tfm_mlp_metrics,
                    "trainable_params": 0,
                    "optimizer_mode": "none",
                    "time_ms": t_tfm_mlp_total * 1000.0,
                    "diff_ppl_vs_full_resumed": tfm_mlp_metrics["val_ppl"] - full_resumed_ppl,
                    "speedup": float(full_resumed_time / max(t_tfm_mlp_total, 1e-6)),
                    "timing_breakdown": {
                        "checkpoint_prep_ms": engine_mlp.prep_time * 1000.0,
                        "svd_ms": engine_mlp.svd_time * 1000.0,
                        "forecast_ms": t_tfm_f_mlp * 1000.0,
                        "reconstruction_and_load_ms": (t_tfm_r_mlp + t_load_tfm_mlp) * 1000.0,
                        "total_latency_ms": t_tfm_mlp_total * 1000.0
                    }
                }

            seed_data["horizons"][f"h_{h}"] = h_data

        all_seed_results.append(seed_data)
        print(f"Seed {seed} completed in {time.perf_counter() - t_seed_start:.2f} s")

    # --------------------------------------------------------------------------
    # Statistical Summary Aggregation
    # --------------------------------------------------------------------------
    print("\nComputing multi-seed statistical summary with 95% CIs and paired tests...", flush=True)
    summary = {
        "metadata": {
            "version": "3.0",
            "study": "Controlled SmolGPT Continuation Benchmark",
            "cutoff_step": cutoff_step,
            "seeds": seeds,
            "horizons": horizons,
            "checkpoint_interval": checkpoint_interval,
            "timesfm_cold_start_load_ms": timesfm_load_ms
        },
        "horizons": {}
    }

    all_methods = list(all_seed_results[0]["horizons"][f"h_{horizons[0]}"]["methods"].keys())

    for h in horizons:
        h_key = f"h_{h}"
        summary["horizons"][h_key] = {"methods": {}}
        base_ppls = [s["horizons"][h_key]["methods"]["full_resumed"]["val_ppl"] for s in all_seed_results]

        for m_name in all_methods:
            val_ppls = [s["horizons"][h_key]["methods"][m_name]["val_ppl"] for s in all_seed_results]
            val_losses = [s["horizons"][h_key]["methods"][m_name]["val_loss"] for s in all_seed_results]
            train_ppls = [s["horizons"][h_key]["methods"][m_name]["train_ppl"] for s in all_seed_results]
            train_losses = [s["horizons"][h_key]["methods"][m_name]["train_loss"] for s in all_seed_results]
            gen_gaps = [s["horizons"][h_key]["methods"][m_name]["gen_gap"] for s in all_seed_results]
            time_ms = [s["horizons"][h_key]["methods"][m_name]["time_ms"] for s in all_seed_results]
            speedups = [s["horizons"][h_key]["methods"][m_name]["speedup"] for s in all_seed_results]

            meta_proto = all_seed_results[0]["horizons"][h_key]["methods"][m_name]

            summary["horizons"][h_key]["methods"][m_name] = {
                "trainable_params": meta_proto["trainable_params"],
                "optimizer_mode": meta_proto["optimizer_mode"],
                "val_ppl": calc_stats(val_ppls),
                "val_loss": calc_stats(val_losses),
                "train_ppl": calc_stats(train_ppls),
                "train_loss": calc_stats(train_losses),
                "gen_gap": calc_stats(gen_gaps),
                "time_ms": calc_stats(time_ms),
                "speedup": calc_stats(speedups),
                "diff_ppl_vs_full_resumed": calc_paired_diff_stats(val_ppls, base_ppls),
                "timing_breakdown": {
                    stage: calc_stats([s["horizons"][h_key]["methods"][m_name]["timing_breakdown"][stage] for s in all_seed_results])
                    for stage in meta_proto["timing_breakdown"].keys()
                }
            }

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/smolgpt_controlled_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open("results/smolgpt_controlled_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save Flattened CSV
    with open("results/smolgpt_controlled_per_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "horizon", "steps_after_cutoff", "method", "trainable_params",
            "optimizer_mode", "val_loss", "val_ppl", "train_loss", "train_ppl",
            "gen_gap", "time_ms", "speedup", "diff_ppl_vs_full_resumed"
        ])
        for s in all_seed_results:
            seed_val = s["seed"]
            for h in horizons:
                h_data = s["horizons"][f"h_{h}"]
                delta_s = h * checkpoint_interval
                for m_name, m_info in h_data["methods"].items():
                    writer.writerow([
                        seed_val, h, delta_s, m_name, m_info["trainable_params"],
                        m_info["optimizer_mode"], m_info["val_loss"], m_info["val_ppl"],
                        m_info["train_loss"], m_info["train_ppl"], m_info["gen_gap"],
                        m_info["time_ms"], m_info["speedup"], m_info["diff_ppl_vs_full_resumed"]
                    ])

    print("\n[Saved] Controlled results: results/smolgpt_controlled_results.json")
    print("[Saved] Controlled summary: results/smolgpt_controlled_summary.json")
    print("[Saved] Controlled per-seed CSV: results/smolgpt_controlled_per_seed.csv")
    return summary

if __name__ == "__main__":
    run_controlled_benchmark()
