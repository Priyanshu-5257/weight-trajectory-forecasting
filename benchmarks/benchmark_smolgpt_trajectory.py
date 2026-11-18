"""
Rigorous Weight Trajectory Forecasting Benchmark on SmolGPT Language Model (v2.0).
Evaluates Google TimesFM 2.0 500M and Low-Rank Extrapolations across extended ranks
(Rank 1, 2, 4, 8, 16, 24, 32) and reports BOTH Train and Validation Loss/Perplexity
alongside generalization gaps, cosine similarity, and compute efficiency.
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


def get_layer_slices(model):
    slices = {"embedding": [], "attention": [], "mlp": [], "norm": []}
    offset = 0
    for name, p in model.named_parameters():
        num = p.numel()
        rng = (offset, offset + num)
        if "wte" in name or "lm_head" in name:
            slices["embedding"].append(rng)
        elif "attn" in name:
            slices["attention"].append(rng)
        elif "mlp" in name:
            slices["mlp"].append(rng)
        else:
            slices["norm"].append(rng)
        offset += num
    return slices


def calc_layer_cos_sim(w_pred, w_act, w_ref, layer_slices):
    dw_pred = w_pred - w_ref
    dw_act = w_act - w_ref
    results = {}
    for group, rngs in layer_slices.items():
        v1_list = [dw_pred[s:e] for s, e in rngs]
        v2_list = [dw_act[s:e] for s, e in rngs]
        v1 = np.concatenate(v1_list)
        v2 = np.concatenate(v2_list)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            results[group] = float(np.dot(v1, v2) / (n1 * n2))
        else:
            results[group] = 0.0
    return results


def run_smolgpt_benchmark(
    seeds=[42, 43, 44],
    cutoff_step=1000,
    context_checkpoints=32,
    checkpoint_interval=20,
    ranks=[1, 2, 4, 8, 16, 24, 32],
    horizons=[8, 16, 32, 64],  # +160, +320, +640, +1280 steps
    batch_size=16,
    block_size=128,
    lr=1e-3,
    weight_decay=0.01,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" RIGOROUS SMOLGPT WEIGHT TRAJECTORY FORECASTING BENCHMARK (v2.0)")
    print(f" Cutoff Step: {cutoff_step} | Checkpoint Interval: {checkpoint_interval} steps")
    print(f" Context: {context_checkpoints} checkpoints (Steps {cutoff_step - context_checkpoints * checkpoint_interval} -> {cutoff_step})")
    print(f" Seeds: {seeds} | Extended Ranks: {ranks} | Horizons: {horizons}")
    print(f" Metrics: Train & Val Loss, Perplexity, Generalization Gap, Cosine Alignment")
    print(f" Device: {device}")
    print("=" * 80, flush=True)

    # 1. Load Dataset
    print("Loading TinyStories token dataset...", flush=True)
    train_tokens, val_tokens = load_tinystories_dataset("data/tinystories/tinystories_sub.npz")
    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}", flush=True)

    # Pre-generate deterministic validation and training batches (40,960 tokens each)
    val_batches = get_deterministic_val_batches(
        val_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device
    )
    train_batches = get_deterministic_train_batches(
        train_tokens, num_batches=20, batch_size=batch_size, block_size=block_size, device=device
    )
    print(f"Materialized 20 val batches ({len(val_batches) * batch_size * block_size:,} tokens) and 20 train batches ({len(train_batches) * batch_size * block_size:,} tokens)", flush=True)

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
        layer_slices = get_layer_slices(model)
        P = sum(p.numel() for p in model.parameters())

        # Track trajectory checkpoints
        checkpoints_history = []  # shape: (P, context_checkpoints)
        future_checkpoints = {}   # h -> (flat_params, step_time)

        # Initial evaluation
        init_metrics = eval_both(model)
        print(f"Initial: Train Loss = {init_metrics['train_loss']:.4f} (PPL: {init_metrics['train_ppl']:.1f}) | Val Loss = {init_metrics['val_loss']:.4f} (PPL: {init_metrics['val_ppl']:.1f})", flush=True)

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
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_step = time.perf_counter() - t_s0
            step_times.append(dt_step)

            # Record history checkpoints
            if start_context_step < step <= cutoff_step:
                if (step - start_context_step) % checkpoint_interval == 0:
                    w_t = get_flat_params(model).cpu().numpy().copy()
                    checkpoints_history.append(w_t)

            # Record future continuation checkpoints
            if step > cutoff_step:
                steps_after = step - cutoff_step
                if steps_after % checkpoint_interval == 0:
                    h_curr = steps_after // checkpoint_interval
                    if h_curr in horizons:
                        w_future = get_flat_params(model).cpu().numpy().copy()
                        future_checkpoints[h_curr] = {
                            "w": w_future,
                            "train_loss": float(loss.item()),
                            "cum_step_time": float(np.sum(step_times[cutoff_step:step]))
                        }

            if step % 200 == 0:
                v_loss, v_ppl = evaluate_transformer_loss_and_ppl(model, val_batches)
                print(f"  Step {step:4d}/{total_steps} | Batch Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f} | PPL: {v_ppl:.2f}", flush=True)

        train_elapsed = time.perf_counter() - t_train_start
        print(f"Training completed in {train_elapsed:.2f} s ({total_steps / train_elapsed:.1f} steps/s)", flush=True)

        # Assemble historical weight window
        weight_window = np.column_stack(checkpoints_history)  # (P, context_checkpoints)
        w_ref = weight_window[:, -1].copy()

        # Build trajectory engine
        engine = LowRankTrajectoryEngine(weight_window, ref_checkpoint=w_ref)

        # Evaluate Frozen Checkpoint
        set_flat_params(model, torch.from_numpy(w_ref).float())
        frozen_metrics = eval_both(model)

        # Evaluate Matched SWA (over last 16 checkpoints)
        w_swa = np.mean(weight_window[:, -16:], axis=1)
        set_flat_params(model, torch.from_numpy(w_swa).float())
        swa_metrics = eval_both(model)

        # Evaluate Matched EMA (beta=0.9)
        w_ema = weight_window[:, 0].copy()
        beta = 0.9
        for col in range(1, weight_window.shape[1]):
            w_ema = beta * w_ema + (1.0 - beta) * weight_window[:, col]
        set_flat_params(model, torch.from_numpy(w_ema).float())
        ema_metrics = eval_both(model)

        # SVD Rank Diagnostics
        rank_diagnostics = {}
        for r in ranks:
            diag = engine.get_rank_diagnostics(r)
            rank_diagnostics[f"rank_{r}"] = diag

        seed_data = {
            "seed": seed,
            "cutoff_step": cutoff_step,
            "frozen": frozen_metrics,
            "swa": swa_metrics,
            "ema": ema_metrics,
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

            # Ground truth: Continued AdamW
            w_actual = future_checkpoints[h]["w"]
            adamw_time = future_checkpoints[h]["cum_step_time"]
            set_flat_params(model, torch.from_numpy(w_actual).float())
            act_metrics = eval_both(model)

            h_data["methods"]["continued_adamw"] = {
                **act_metrics,
                "cos_sim": 1.0,
                "rel_l2_error": 0.0,
                "time_ms": adamw_time * 1000.0,
                "speedup": 1.0
            }

            # Frozen baseline
            h_data["methods"]["frozen_checkpoint"] = {
                **frozen_metrics,
                "cos_sim": 0.0,
                "rel_l2_error": float(np.linalg.norm(w_ref - w_actual) / (np.linalg.norm(w_actual - w_ref) + 1e-8)),
                "time_ms": 0.0,
                "speedup": float("inf")
            }

            # SWA baseline
            h_data["methods"]["matched_swa"] = {
                **swa_metrics,
                "cos_sim": float(np.dot(w_swa - w_ref, w_actual - w_ref) / (np.linalg.norm(w_swa - w_ref) * np.linalg.norm(w_actual - w_ref) + 1e-8)),
                "rel_l2_error": float(np.linalg.norm(w_swa - w_actual) / (np.linalg.norm(w_actual - w_ref) + 1e-8)),
                "time_ms": 0.0,
                "speedup": float("inf")
            }

            # EMA baseline
            h_data["methods"]["matched_ema"] = {
                **ema_metrics,
                "cos_sim": float(np.dot(w_ema - w_ref, w_actual - w_ref) / (np.linalg.norm(w_ema - w_ref) * np.linalg.norm(w_actual - w_ref) + 1e-8)),
                "rel_l2_error": float(np.linalg.norm(w_ema - w_actual) / (np.linalg.norm(w_actual - w_ref) + 1e-8)),
                "time_ms": 0.0,
                "speedup": float("inf")
            }

            # Weight decay baseline
            w_wd, _, t_wd = engine.forecast_weight_decay(h, eta=lr, weight_decay=weight_decay, interval=checkpoint_interval)
            w_wd_np = w_wd.numpy()
            set_flat_params(model, w_wd)
            wd_metrics = eval_both(model)
            norm_wd = np.linalg.norm(w_wd_np - w_ref)
            norm_act = np.linalg.norm(w_actual - w_ref)
            cos_wd = float(np.dot(w_wd_np - w_ref, w_actual - w_ref) / (norm_wd * norm_act + 1e-8)) if norm_wd > 1e-8 else 0.0
            h_data["methods"]["weight_decay_only"] = {
                **wd_metrics,
                "cos_sim": cos_wd,
                "rel_l2_error": float(np.linalg.norm(w_wd_np - w_actual) / (norm_act + 1e-8)),
                "time_ms": t_wd * 1000.0,
                "speedup": float(adamw_time / max(t_wd, 1e-6))
            }

            # Low-Rank Extrapolations across ranks
            for r in ranks:
                # 1. Linear
                w_lin, c_lin, t_lin = engine.forecast_linear(r, h)
                w_lin_np = w_lin.numpy()
                set_flat_params(model, w_lin)
                lin_metrics = eval_both(model)
                norm_lin = np.linalg.norm(w_lin_np - w_ref)
                cos_lin = float(np.dot(w_lin_np - w_ref, w_actual - w_ref) / (norm_lin * norm_act + 1e-8)) if norm_lin > 1e-8 else 0.0
                layer_cos_lin = calc_layer_cos_sim(w_lin_np, w_actual, w_ref, layer_slices)

                h_data["methods"][f"low_rank_linear_r{r}"] = {
                    **lin_metrics,
                    "cos_sim": cos_lin,
                    "rel_l2_error": float(np.linalg.norm(w_lin_np - w_actual) / (norm_act + 1e-8)),
                    "layer_cos_sim": layer_cos_lin,
                    "time_ms": t_lin * 1000.0,
                    "speedup": float(adamw_time / max(t_lin, 1e-6))
                }

                # 2. Damped (gamma=0.95)
                w_damp, c_damp, t_damp = engine.forecast_damped(r, h, gamma=0.95)
                w_damp_np = w_damp.numpy()
                set_flat_params(model, w_damp)
                damp_metrics = eval_both(model)
                norm_damp = np.linalg.norm(w_damp_np - w_ref)
                cos_damp = float(np.dot(w_damp_np - w_ref, w_actual - w_ref) / (norm_damp * norm_act + 1e-8)) if norm_damp > 1e-8 else 0.0
                layer_cos_damp = calc_layer_cos_sim(w_damp_np, w_actual, w_ref, layer_slices)

                h_data["methods"][f"low_rank_damped_r{r}"] = {
                    **damp_metrics,
                    "cos_sim": cos_damp,
                    "rel_l2_error": float(np.linalg.norm(w_damp_np - w_actual) / (norm_act + 1e-8)),
                    "layer_cos_sim": layer_cos_damp,
                    "time_ms": t_damp * 1000.0,
                    "speedup": float(adamw_time / max(t_damp, 1e-6))
                }

                # 3. TimesFM-2.0-500M
                w_tfm, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device)
                w_tfm_np = w_tfm.numpy()
                set_flat_params(model, w_tfm)
                tfm_metrics = eval_both(model)
                norm_tfm = np.linalg.norm(w_tfm_np - w_ref)
                cos_tfm = float(np.dot(w_tfm_np - w_ref, w_actual - w_ref) / (norm_tfm * norm_act + 1e-8)) if norm_tfm > 1e-8 else 0.0
                layer_cos_tfm = calc_layer_cos_sim(w_tfm_np, w_actual, w_ref, layer_slices)
                total_t_tfm = t_tfm_f + t_tfm_r

                h_data["methods"][f"low_rank_timesfm_r{r}"] = {
                    **tfm_metrics,
                    "cos_sim": cos_tfm,
                    "rel_l2_error": float(np.linalg.norm(w_tfm_np - w_actual) / (norm_act + 1e-8)),
                    "layer_cos_sim": layer_cos_tfm,
                    "time_ms": total_t_tfm * 1000.0,
                    "speedup": float(adamw_time / max(total_t_tfm, 1e-6))
                }

                # 4. Shuffled TimesFM (control across all ranks)
                w_shuf, c_shuf, t_shuf_f, t_shuf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                w_shuf_np = w_shuf.numpy()
                set_flat_params(model, w_shuf)
                shuf_metrics = eval_both(model)
                norm_shuf = np.linalg.norm(w_shuf_np - w_ref)
                cos_shuf = float(np.dot(w_shuf_np - w_ref, w_actual - w_ref) / (norm_shuf * norm_act + 1e-8)) if norm_shuf > 1e-8 else 0.0

                h_data["methods"][f"shuffled_timesfm_r{r}"] = {
                    **shuf_metrics,
                    "cos_sim": cos_shuf,
                    "rel_l2_error": float(np.linalg.norm(w_shuf_np - w_actual) / (norm_act + 1e-8)),
                    "time_ms": (t_shuf_f + t_shuf_r) * 1000.0,
                    "speedup": float(adamw_time / max(t_shuf_f + t_shuf_r, 1e-6))
                }

            seed_data["horizons"][f"h_{h}"] = h_data

        all_seed_results.append(seed_data)
        print(f"Seed {seed} finished in {time.perf_counter() - t_seed_start:.2f} s", flush=True)

    # 3. Compute Multi-Seed Statistical Aggregation
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
        "swa": {
            "train_loss": calc_stats([s["swa"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["swa"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["swa"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["swa"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["swa"]["gen_gap"] for s in all_seed_results]),
        },
        "ema": {
            "train_loss": calc_stats([s["ema"]["train_loss"] for s in all_seed_results]),
            "train_ppl": calc_stats([s["ema"]["train_ppl"] for s in all_seed_results]),
            "val_loss": calc_stats([s["ema"]["val_loss"] for s in all_seed_results]),
            "val_ppl": calc_stats([s["ema"]["val_ppl"] for s in all_seed_results]),
            "gen_gap": calc_stats([s["ema"]["gen_gap"] for s in all_seed_results]),
        },
        "explained_variance_cum": {},
        "horizons": {}
    }

    # Explained variance aggregation
    for r in ranks:
        ev_vals = [s["rank_diagnostics"][f"rank_{r}"]["explained_variance"] for s in all_seed_results]
        summary["explained_variance_cum"][f"rank_{r}"] = calc_stats(ev_vals)

    # Method aggregation per horizon
    first_h_key = f"h_{horizons[0]}"
    all_method_keys = list(all_seed_results[0]["horizons"][first_h_key]["methods"].keys())

    for h in horizons:
        h_key = f"h_{h}"
        summary["horizons"][h_key] = {"methods": {}}
        adamw_ppl_vals = [s["horizons"][h_key]["methods"]["continued_adamw"]["val_ppl"] for s in all_seed_results]
        frozen_ppl_vals = [s["horizons"][h_key]["methods"]["frozen_checkpoint"]["val_ppl"] for s in all_seed_results]

        for method in all_method_keys:
            t_loss_vals = [s["horizons"][h_key]["methods"][method]["train_loss"] for s in all_seed_results]
            t_ppl_vals = [s["horizons"][h_key]["methods"][method]["train_ppl"] for s in all_seed_results]
            v_loss_vals = [s["horizons"][h_key]["methods"][method]["val_loss"] for s in all_seed_results]
            v_ppl_vals = [s["horizons"][h_key]["methods"][method]["val_ppl"] for s in all_seed_results]
            gap_vals = [s["horizons"][h_key]["methods"][method]["gen_gap"] for s in all_seed_results]

            cos_vals = [s["horizons"][h_key]["methods"][method]["cos_sim"] for s in all_seed_results]
            l2_vals = [s["horizons"][h_key]["methods"][method]["rel_l2_error"] for s in all_seed_results]
            time_vals = [s["horizons"][h_key]["methods"][method]["time_ms"] for s in all_seed_results]
            speedup_vals = [s["horizons"][h_key]["methods"][method]["speedup"] for s in all_seed_results]

            paired_vs_adamw = calc_paired_diff_stats(v_ppl_vals, adamw_ppl_vals)
            paired_vs_frozen = calc_paired_diff_stats(v_ppl_vals, frozen_ppl_vals)

            method_summary = {
                "train_loss": calc_stats(t_loss_vals),
                "train_ppl": calc_stats(t_ppl_vals),
                "val_loss": calc_stats(v_loss_vals),
                "val_ppl": calc_stats(v_ppl_vals),
                "gen_gap": calc_stats(gap_vals),
                "cos_sim": calc_stats(cos_vals),
                "rel_l2_error": calc_stats(l2_vals),
                "time_ms": calc_stats(time_vals),
                "speedup": calc_stats(speedup_vals),
                "paired_diff_vs_adamw_ppl": paired_vs_adamw,
                "paired_diff_vs_frozen_ppl": paired_vs_frozen,
            }

            # Aggregate layer-wise cos sim if present
            if "layer_cos_sim" in all_seed_results[0]["horizons"][h_key]["methods"][method]:
                method_summary["layer_cos_sim"] = {}
                for layer_grp in ["embedding", "attention", "mlp", "norm"]:
                    l_vals = [s["horizons"][h_key]["methods"][method]["layer_cos_sim"][layer_grp] for s in all_seed_results]
                    method_summary["layer_cos_sim"][layer_grp] = calc_stats(l_vals)

            summary["horizons"][h_key]["methods"][method] = method_summary

    # Save JSON files
    os.makedirs("results", exist_ok=True)
    with open("results/smolgpt_benchmark_results.json", "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open("results/smolgpt_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV per-seed results
    with open("results/smolgpt_benchmark_per_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "horizon", "equivalent_steps", "method", "train_loss", "train_ppl", "val_loss", "val_ppl", "gen_gap", "cos_sim", "rel_l2_error", "time_ms", "speedup"])
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
                        f"{m_data['cos_sim']:.4f}",
                        f"{m_data['rel_l2_error']:.4f}",
                        f"{m_data['time_ms']:.2f}",
                        f"{m_data['speedup']:.2f}"
                    ])

    print("Saved results to results/smolgpt_benchmark_results.json", flush=True)
    print("Saved summary to results/smolgpt_benchmark_summary.json", flush=True)
    print("Saved CSV to results/smolgpt_benchmark_per_seed.csv", flush=True)

    return all_seed_results, summary


if __name__ == "__main__":
    run_smolgpt_benchmark()
