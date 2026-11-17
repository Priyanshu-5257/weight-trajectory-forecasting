import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TimesFmModelForPrediction
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import GrokkingTransformer, get_flat_params, set_flat_params, eval_grokking_tensors
from src.data import load_modular_addition_dataset
from src.low_rank import LowRankTrajectoryEngine

# ==============================================================================
# Helper Functions
# ==============================================================================

def calc_stats(vals):
    vals = np.array(vals, dtype=np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}

# ==============================================================================
# Main Grokking Trajectory Benchmark Runner
# ==============================================================================

def run_grokking_benchmark(
    seeds=list(range(42, 47)), # 5 paired seeds
    p=97,
    frac_train=0.5,
    cutoff_step=2000,
    context_checkpoints=32,
    checkpoint_interval=25,
    ranks=[1, 2, 4, 8],
    horizons=[8, 16, 32, 50, 64],
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" GROKKING WEIGHT TRAJECTORY FORECASTING BENCHMARK")
    print(f" Modulus P={p} | Frac Train={frac_train*100:.0f}% | Cutoff Step={cutoff_step}")
    print(f" Context: {context_checkpoints} checkpoints every {checkpoint_interval} steps (Steps {cutoff_step - context_checkpoints*checkpoint_interval} -> {cutoff_step})")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizons: {horizons} (Delta steps: {[h*checkpoint_interval for h in horizons]})")
    print(f" Device: {device}")
    print("=" * 80, flush=True)

    # 1. Load TimesFM Model
    t_load_start = time.perf_counter()
    print("Loading TimesFM-2.0-500M...", flush=True)
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()
    print(f"TimesFM loaded in {time.perf_counter() - t_load_start:.2f} s", flush=True)

    max_horizon = max(horizons)
    total_training_steps = cutoff_step + (max_horizon * checkpoint_interval)
    criterion = nn.CrossEntropyLoss()

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n" + "-" * 70, flush=True)
        print(f"--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        print("-" * 70, flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        data = load_modular_addition_dataset(p=p, frac_train=frac_train, seed=seed, device=device)
        train_x, train_y = data["train_data"], data["train_labels"]
        test_x, test_y = data["test_data"], data["test_labels"]

        model = GrokkingTransformer(p=p).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)

        # We will record checkpoints at multiples of checkpoint_interval
        # Start recording when step >= cutoff_step - context_checkpoints * checkpoint_interval
        start_record_step = cutoff_step - (context_checkpoints * checkpoint_interval)
        
        saved_checkpoints = {} # step -> flat_params (cpu)
        step_times = {}

        t0_train = time.perf_counter()
        print(f"Training base model to step {total_training_steps}...", flush=True)
        for step in range(1, total_training_steps + 1):
            t0_step = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0_step
            step_times[step] = dt

            if step >= start_record_step and (step % checkpoint_interval == 0):
                saved_checkpoints[step] = get_flat_params(model).cpu()

            if step % 500 == 0 or step == cutoff_step or step == total_training_steps:
                tr_l, tr_a = eval_grokking_tensors(model, train_x, train_y, criterion)
                te_l, te_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                print(f"  Step {step:5d} | Train: Loss={tr_l:.4f}, Acc={tr_a:5.1f}% | Test: Loss={te_l:.4f}, Acc={te_a:5.1f}%", flush=True)

        # Extract Context Window for SVD: exactly context_checkpoints checkpoints up to cutoff_step
        context_steps = [cutoff_step - (context_checkpoints - 1 - i) * checkpoint_interval for i in range(context_checkpoints)]
        context_tensors = [saved_checkpoints[s] for s in context_steps]
        weight_window_np = torch.stack(context_tensors, dim=1).numpy() # (P, T=32)

        w_cutoff = saved_checkpoints[cutoff_step]
        set_flat_params(model, w_cutoff.to(device))
        val_cutoff_loss, val_cutoff_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
        train_cutoff_loss, train_cutoff_acc = eval_grokking_tensors(model, train_x, train_y, criterion)

        # Baseline: SWA over the context window
        w_swa = torch.mean(torch.stack(context_tensors, dim=0), dim=0)
        set_flat_params(model, w_swa.to(device))
        val_swa_loss, val_swa_acc = eval_grokking_tensors(model, test_x, test_y, criterion)

        # Baseline: EMA over context window (decay=0.9)
        w_ema = context_tensors[0].clone()
        for ckpt in context_tensors[1:]:
            w_ema = 0.9 * w_ema + 0.1 * ckpt
        set_flat_params(model, w_ema.to(device))
        val_ema_loss, val_ema_acc = eval_grokking_tensors(model, test_x, test_y, criterion)

        # Continued Adam ground truth for each horizon
        continued_adam = {}
        for h in horizons:
            target_step = cutoff_step + (h * checkpoint_interval)
            w_cont = saved_checkpoints[target_step]
            set_flat_params(model, w_cont.to(device))
            c_loss, c_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
            time_spent = sum([step_times[s] for s in range(cutoff_step + 1, target_step + 1)])
            continued_adam[f"h_{h}"] = {
                "target_step": target_step,
                "test_loss": c_loss,
                "test_acc": c_acc,
                "grokked": bool(c_acc >= 95.0),
                "train_time_s": float(time_spent)
            }

        # Initialize Low-Rank Trajectory Engine
        engine = LowRankTrajectoryEngine(weight_window_np, ref_checkpoint=w_cutoff.numpy())

        seed_entry = {
            "seed": seed,
            "cutoff_step": cutoff_step,
            "baselines": {
                "frozen": {"test_loss": val_cutoff_loss, "test_acc": val_cutoff_acc, "train_loss": train_cutoff_loss, "train_acc": train_cutoff_acc},
                "swa": {"test_loss": val_swa_loss, "test_acc": val_swa_acc},
                "ema": {"test_loss": val_ema_loss, "test_acc": val_ema_acc},
                "continued_adam": continued_adam
            },
            "svd_time_s": engine.svd_time,
            "rank_diagnostics": {r: engine.get_rank_diagnostics(r) for r in ranks},
            "evaluations": {}
        }

        print(f"\n  [Diagnostics] SVD Time: {engine.svd_time:.4f}s | Explained Var: " + 
              ", ".join([f"r={r}: {engine.get_rank_diagnostics(r)['explained_variance']*100:.1f}%" for r in ranks]), flush=True)
        print(f"  Frozen: Acc={val_cutoff_acc:.2f}% | SWA: Acc={val_swa_acc:.2f}% | EMA: Acc={val_ema_acc:.2f}%", flush=True)

        for r in ranks:
            seed_entry["evaluations"][f"rank_{r}"] = {}
            for h in horizons:
                target_step = cutoff_step + (h * checkpoint_interval)
                w_act = saved_checkpoints[target_step]

                # 1. Linear Extrapolation
                w_lin, c_lin, t_lin = engine.forecast_linear(r, h)
                set_flat_params(model, w_lin.to(device))
                lin_loss, lin_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
                lin_tr_loss, lin_tr_acc = eval_grokking_tensors(model, train_x, train_y, criterion)
                mech_lin = engine.compute_mechanism_metrics(r, c_lin, w_lin, w_swa, w_act)

                # 2. Damped Linear Extrapolation (gamma=0.95)
                w_dmp, c_dmp, t_dmp = engine.forecast_damped(r, h, gamma=0.95)
                set_flat_params(model, w_dmp.to(device))
                dmp_loss, dmp_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
                dmp_tr_loss, dmp_tr_acc = eval_grokking_tensors(model, train_x, train_y, criterion)
                mech_dmp = engine.compute_mechanism_metrics(r, c_dmp, w_dmp, w_swa, w_act)

                # 3. TimesFM (Ordered)
                w_tfm, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=False)
                set_flat_params(model, w_tfm.to(device))
                tfm_loss, tfm_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
                tfm_tr_loss, tfm_tr_acc = eval_grokking_tensors(model, train_x, train_y, criterion)
                mech_tfm = engine.compute_mechanism_metrics(r, c_tfm, w_tfm, w_swa, w_act)

                # 4. TimesFM (Shuffled Control)
                w_shuf, c_shuf, t_shuf_f, t_shuf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                set_flat_params(model, w_shuf.to(device))
                shuf_loss, shuf_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
                mech_shuf = engine.compute_mechanism_metrics(r, c_shuf, w_shuf, w_swa, w_act)

                seed_entry["evaluations"][f"rank_{r}"][f"horizon_{h}"] = {
                    "linear": {
                        "test_loss": lin_loss, "test_acc": lin_acc, "train_loss": lin_tr_loss, "train_acc": lin_tr_acc,
                        "grokked": bool(lin_acc >= 95.0),
                        "time_s": t_lin, "mechanism": mech_lin
                    },
                    "damped": {
                        "test_loss": dmp_loss, "test_acc": dmp_acc, "train_loss": dmp_tr_loss, "train_acc": dmp_tr_acc,
                        "grokked": bool(dmp_acc >= 95.0),
                        "time_s": t_dmp, "mechanism": mech_dmp
                    },
                    "timesfm": {
                        "test_loss": tfm_loss, "test_acc": tfm_acc, "train_loss": tfm_tr_loss, "train_acc": tfm_tr_acc,
                        "grokked": bool(tfm_acc >= 95.0),
                        "forecast_time_s": t_tfm_f, "reconstruct_time_s": t_tfm_r,
                        "mechanism": mech_tfm
                    },
                    "shuffled_timesfm": {
                        "test_loss": shuf_loss, "test_acc": shuf_acc,
                        "grokked": bool(shuf_acc >= 95.0),
                        "mechanism": mech_shuf
                    }
                }

                print(f"    [r={r}, h={h} (+{h*checkpoint_interval} steps)] Cont Acc={continued_adam[f'h_{h}']['test_acc']:.1f}% | "
                      f"Lin={lin_acc:.1f}% (cos={mech_lin['cos_sim_to_continuation']:.2f}) | "
                      f"Dmp={dmp_acc:.1f}% (cos={mech_dmp['cos_sim_to_continuation']:.2f}) | "
                      f"TFM={tfm_acc:.1f}% (cos={mech_tfm['cos_sim_to_continuation']:.2f}) | "
                      f"Shuf={shuf_acc:.1f}%", flush=True)

        all_seed_results.append(seed_entry)

    # ==============================================================================
    # Aggregate Summary Statistics Across Seeds
    # ==============================================================================
    print("\n" + "=" * 80)
    print(" AGGREGATING MULTI-SEED GROKKING BENCHMARK RESULTS")
    print("=" * 80, flush=True)

    summary = {
        "seeds": seeds,
        "p": p,
        "frac_train": frac_train,
        "cutoff_step": cutoff_step,
        "context_checkpoints": context_checkpoints,
        "checkpoint_interval": checkpoint_interval,
        "ranks": ranks,
        "horizons": horizons,
        "baselines": {
            "frozen_acc": calc_stats([s["baselines"]["frozen"]["test_acc"] for s in all_seed_results]),
            "frozen_loss": calc_stats([s["baselines"]["frozen"]["test_loss"] for s in all_seed_results]),
            "swa_acc": calc_stats([s["baselines"]["swa"]["test_acc"] for s in all_seed_results]),
            "swa_loss": calc_stats([s["baselines"]["swa"]["test_loss"] for s in all_seed_results]),
            "ema_acc": calc_stats([s["baselines"]["ema"]["test_acc"] for s in all_seed_results]),
            "ema_loss": calc_stats([s["baselines"]["ema"]["test_loss"] for s in all_seed_results]),
            "continued_adam": {}
        },
        "explained_variance": {
            r: calc_stats([s["rank_diagnostics"][r]["explained_variance"] for s in all_seed_results])
            for r in ranks
        },
        "methods": {"linear": {}, "damped": {}, "timesfm": {}, "shuffled_timesfm": {}}
    }

    for h in horizons:
        c_accs = [s["baselines"]["continued_adam"][f"h_{h}"]["test_acc"] for s in all_seed_results]
        c_losses = [s["baselines"]["continued_adam"][f"h_{h}"]["test_loss"] for s in all_seed_results]
        summary["baselines"]["continued_adam"][f"h_{h}"] = {
            "acc": calc_stats(c_accs),
            "loss": calc_stats(c_losses)
        }

    for method in ["linear", "damped", "timesfm", "shuffled_timesfm"]:
        summary["methods"][method] = {}
        for r in ranks:
            summary["methods"][method][f"rank_{r}"] = {}
            for h in horizons:
                accs = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["test_acc"] for s in all_seed_results]
                losses = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["test_loss"] for s in all_seed_results]
                cos_sims = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["mechanism"]["cos_sim_to_continuation"] for s in all_seed_results]
                summary["methods"][method][f"rank_{r}"][f"horizon_{h}"] = {
                    "acc": calc_stats(accs),
                    "loss": calc_stats(losses),
                    "cos_sim_to_continuation": calc_stats(cos_sims)
                }

    # Save to results/
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    raw_path = os.path.join(ROOT_DIR, "results", "grokking_benchmark_results.json")
    summary_path = os.path.join(ROOT_DIR, "results", "grokking_benchmark_summary.json")

    with open(raw_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] Raw results: {raw_path}")
    print(f"[Saved] Summary statistics: {summary_path}")
    return all_seed_results, summary

if __name__ == "__main__":
    run_grokking_benchmark(
        seeds=list(range(42, 47)), # 5 seeds
        p=97,
        frac_train=0.5,
        cutoff_step=2000,
        context_checkpoints=32,
        checkpoint_interval=25,
        ranks=[1, 2, 4, 8],
        horizons=[8, 16, 32, 50, 64]
    )
