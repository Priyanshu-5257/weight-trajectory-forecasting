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
# Helper Functions: Statistics & Fourier Analysis
# ==============================================================================

def calc_stats(vals):
    vals = np.array(vals, dtype=np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)) if len(vals) > 1 else 0.0
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

def compute_fourier_saliency(w_embed, p=97):
    """
    Computes top-5 DFT frequency power concentration ratio on token embeddings W_E[:p].
    """
    w_embed = w_embed[:p].float().cpu()
    x = torch.arange(p, dtype=torch.float32).unsqueeze(1) # (p, 1)
    max_k = (p - 1) // 2
    powers = []
    for k in range(1, max_k + 1):
        cos_k = torch.cos(2 * np.pi * k * x / p)
        sin_k = torch.sin(2 * np.pi * k * x / p)
        c_proj = (w_embed * cos_k).sum(dim=0)
        s_proj = (w_embed * sin_k).sum(dim=0)
        power_k = (c_proj**2 + s_proj**2).sum().item()
        powers.append(power_k)
    powers = np.array(powers)
    total_power = np.sum(powers) + 1e-12
    top5_power = np.sum(np.sort(powers)[-5:])
    return float(top5_power / total_power)

# ==============================================================================
# Main Grokking Trajectory Benchmark Runner
# ==============================================================================

def run_grokking_benchmark(
    seeds=list(range(42, 47)), # 5 paired seeds
    p=97,
    frac_train=0.5,
    cutoff_step=1500, # Clean pre-grokking floor before any generalization rise
    context_checkpoints=32,
    checkpoint_interval=25,
    ranks=[1, 2, 4, 8],
    horizons=[8, 16, 32, 50, 64, 80, 100], # +200, +400, +800, +1250, +1600, +2000, +2500 steps
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" RIGOROUS GROKKING WEIGHT TRAJECTORY FORECASTING BENCHMARK (v2.0)")
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

        # Extract strictly non-symmetric test pairs (neither (a,b) nor (b,a) in train)
        train_pairs = set([(int(x[0]), int(x[1])) for x in train_x.cpu()])
        non_sym_idx = [i for i, (a, b, _) in enumerate(test_x.cpu()) if (int(b), int(a)) not in train_pairs]
        non_sym_x = test_x[non_sym_idx]
        non_sym_y = test_y[non_sym_idx]

        print(f"Dataset: Train={len(train_x)} | Test All={len(test_x)} | Test Non-Symmetric={len(non_sym_x)} (Pure Generalization)")

        model = GrokkingTransformer(p=p).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)

        start_record_step = cutoff_step - (context_checkpoints * checkpoint_interval)
        saved_checkpoints = {} # step -> flat_params (cpu)
        step_times = {}

        print(f"Training base model through step {total_training_steps}...", flush=True)
        for step in range(1, total_training_steps + 1):
            t0_step = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0_step
            step_times[step] = dt

            if step >= start_record_step and (step % checkpoint_interval == 0):
                saved_checkpoints[step] = get_flat_params(model).cpu()

            if step % 500 == 0 or step == cutoff_step or step == total_training_steps:
                tr_l, tr_a = eval_grokking_tensors(model, train_x, train_y, criterion)
                te_l, te_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                ns_l, ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                fourier_sal = compute_fourier_saliency(model.token_embed.weight.data, p=p)
                print(f"  Step {step:5d} | Train: Loss={tr_l:.4f}, Acc={tr_a:5.1f}% | "
                      f"Test All: {te_a:5.1f}% | Test Non-Sym: {ns_a:5.2f}% | Fourier Sal: {fourier_sal*100:.1f}%", flush=True)

        # Context window for SVD
        context_steps = [cutoff_step - (context_checkpoints - 1 - i) * checkpoint_interval for i in range(context_checkpoints)]
        context_tensors = [saved_checkpoints[s] for s in context_steps]
        weight_window_np = torch.stack(context_tensors, dim=1).numpy() # (P, 32)

        w_cutoff = saved_checkpoints[cutoff_step]
        set_flat_params(model, w_cutoff.to(device))
        val_cutoff_loss, val_cutoff_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
        ns_cutoff_loss, ns_cutoff_acc = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
        tr_cutoff_loss, tr_cutoff_acc = eval_grokking_tensors(model, train_x, train_y, criterion)
        fourier_cutoff = compute_fourier_saliency(model.token_embed.weight.data, p=p)

        # SWA Baseline
        w_swa = torch.mean(torch.stack(context_tensors, dim=0), dim=0)
        set_flat_params(model, w_swa.to(device))
        val_swa_loss, val_swa_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
        ns_swa_loss, ns_swa_acc = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

        # EMA Baseline
        w_ema = context_tensors[0].clone()
        for ckpt in context_tensors[1:]:
            w_ema = 0.9 * w_ema + 0.1 * ckpt
        set_flat_params(model, w_ema.to(device))
        val_ema_loss, val_ema_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
        ns_ema_loss, ns_ema_acc = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

        # Continued Adam ground truth
        continued_adam = {}
        for h in horizons:
            target_step = cutoff_step + (h * checkpoint_interval)
            w_cont = saved_checkpoints[target_step]
            set_flat_params(model, w_cont.to(device))
            c_loss, c_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
            c_ns_loss, c_ns_acc = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
            fourier_cont = compute_fourier_saliency(model.token_embed.weight.data, p=p)
            time_spent = sum([step_times[s] for s in range(cutoff_step + 1, target_step + 1)])
            continued_adam[f"h_{h}"] = {
                "target_step": target_step,
                "test_loss": c_loss, "test_acc": c_acc,
                "test_loss_nonsym": c_ns_loss, "test_acc_nonsym": c_ns_acc,
                "fourier_saliency": fourier_cont,
                "grokked": bool(c_ns_acc >= 90.0),
                "train_time_s": float(time_spent)
            }

        engine = LowRankTrajectoryEngine(weight_window_np, ref_checkpoint=w_cutoff.numpy())

        # Radial displacement analysis: how much displacement is simple radial shrinkage
        D = engine.D
        v_radial = -w_cutoff.numpy() / (np.linalg.norm(w_cutoff.numpy()) + 1e-8)
        radial_var = np.sum((v_radial @ D) ** 2)
        total_var = np.sum(D ** 2) + 1e-12
        radial_var_ratio = float(radial_var / total_var)

        seed_entry = {
            "seed": seed,
            "cutoff_step": cutoff_step,
            "radial_var_ratio": radial_var_ratio,
            "baselines": {
                "frozen": {
                    "test_loss": val_cutoff_loss, "test_acc": val_cutoff_acc,
                    "test_loss_nonsym": ns_cutoff_loss, "test_acc_nonsym": ns_cutoff_acc,
                    "train_loss": tr_cutoff_loss, "train_acc": tr_cutoff_acc,
                    "fourier_saliency": fourier_cutoff
                },
                "swa": {
                    "test_loss": val_swa_loss, "test_acc": val_swa_acc,
                    "test_loss_nonsym": ns_swa_loss, "test_acc_nonsym": ns_swa_acc
                },
                "ema": {
                    "test_loss": val_ema_loss, "test_acc": val_ema_acc,
                    "test_loss_nonsym": ns_ema_loss, "test_acc_nonsym": ns_ema_acc
                },
                "weight_decay_only": {},
                "continued_adam": continued_adam
            },
            "svd_time_s": engine.svd_time,
            "rank_diagnostics": {r: engine.get_rank_diagnostics(r) for r in ranks},
            "evaluations": {}
        }

        print(f"\n  [Diagnostics] SVD Time: {engine.svd_time:.4f}s | Radial Shrinkage Variance: {radial_var_ratio*100:.2f}% | Explained Var: " + 
              ", ".join([f"r={r}: {engine.get_rank_diagnostics(r)['explained_variance']*100:.1f}%" for r in ranks]), flush=True)
        print(f"  Frozen: Acc(All)={val_cutoff_acc:.2f}%, Acc(NonSym)={ns_cutoff_acc:.2f}% | SWA: Acc(NonSym)={ns_swa_acc:.2f}% | EMA: Acc(NonSym)={ns_ema_acc:.2f}%", flush=True)

        # Evaluate pure Weight-Decay baseline for each horizon
        for h in horizons:
            w_dec, c_dec, t_dec = engine.forecast_weight_decay(h, eta=1e-3, weight_decay=1.0, interval=checkpoint_interval)
            set_flat_params(model, w_dec.to(device))
            dec_l, dec_a = eval_grokking_tensors(model, test_x, test_y, criterion)
            dec_ns_l, dec_ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
            w_act = saved_checkpoints[cutoff_step + (h * checkpoint_interval)]
            mech_dec = engine.compute_mechanism_metrics(1, c_dec[:1], w_dec, w_swa, w_act)
            seed_entry["baselines"]["weight_decay_only"][f"h_{h}"] = {
                "test_loss": dec_l, "test_acc": dec_a,
                "test_loss_nonsym": dec_ns_l, "test_acc_nonsym": dec_ns_a,
                "mechanism": mech_dec
            }

        # Grid of ranks and horizons
        for r in ranks:
            seed_entry["evaluations"][f"rank_{r}"] = {}
            for h in horizons:
                target_step = cutoff_step + (h * checkpoint_interval)
                w_act = saved_checkpoints[target_step]

                # 1. Linear Extrapolation
                w_lin, c_lin, t_lin = engine.forecast_linear(r, h)
                set_flat_params(model, w_lin.to(device))
                lin_l, lin_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                lin_ns_l, lin_ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                fourier_lin = compute_fourier_saliency(model.token_embed.weight.data, p=p)
                mech_lin = engine.compute_mechanism_metrics(r, c_lin, w_lin, w_swa, w_act)

                # 2. Damped Linear Extrapolation (gamma=0.95)
                w_dmp, c_dmp, t_dmp = engine.forecast_damped(r, h, gamma=0.95)
                set_flat_params(model, w_dmp.to(device))
                dmp_l, dmp_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                dmp_ns_l, dmp_ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                fourier_dmp = compute_fourier_saliency(model.token_embed.weight.data, p=p)
                mech_dmp = engine.compute_mechanism_metrics(r, c_dmp, w_dmp, w_swa, w_act)

                # 3. TimesFM (with corrected horizon indexing!)
                w_tfm, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=False)
                set_flat_params(model, w_tfm.to(device))
                tfm_l, tfm_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                tfm_ns_l, tfm_ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                fourier_tfm = compute_fourier_saliency(model.token_embed.weight.data, p=p)
                mech_tfm = engine.compute_mechanism_metrics(r, c_tfm, w_tfm, w_swa, w_act)

                # 4. TimesFM (Shuffled History Control)
                w_shuf, c_shuf, t_shuf_f, t_shuf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                set_flat_params(model, w_shuf.to(device))
                shuf_l, shuf_a = eval_grokking_tensors(model, test_x, test_y, criterion)
                shuf_ns_l, shuf_ns_a = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                mech_shuf = engine.compute_mechanism_metrics(r, c_shuf, w_shuf, w_swa, w_act)

                seed_entry["evaluations"][f"rank_{r}"][f"horizon_{h}"] = {
                    "linear": {
                        "test_loss": lin_l, "test_acc": lin_a,
                        "test_loss_nonsym": lin_ns_l, "test_acc_nonsym": lin_ns_a,
                        "fourier_saliency": fourier_lin,
                        "grokked": bool(lin_ns_a >= 90.0),
                        "time_s": t_lin, "mechanism": mech_lin
                    },
                    "damped": {
                        "test_loss": dmp_l, "test_acc": dmp_a,
                        "test_loss_nonsym": dmp_ns_l, "test_acc_nonsym": dmp_ns_a,
                        "fourier_saliency": fourier_dmp,
                        "grokked": bool(dmp_ns_a >= 90.0),
                        "time_s": t_dmp, "mechanism": mech_dmp
                    },
                    "timesfm": {
                        "test_loss": tfm_l, "test_acc": tfm_a,
                        "test_loss_nonsym": tfm_ns_l, "test_acc_nonsym": tfm_ns_a,
                        "fourier_saliency": fourier_tfm,
                        "grokked": bool(tfm_ns_a >= 90.0),
                        "forecast_time_s": t_tfm_f, "reconstruct_time_s": t_tfm_r,
                        "mechanism": mech_tfm
                    },
                    "shuffled_timesfm": {
                        "test_loss": shuf_l, "test_acc": shuf_a,
                        "test_loss_nonsym": shuf_ns_l, "test_acc_nonsym": shuf_ns_a,
                        "grokked": bool(shuf_ns_a >= 90.0),
                        "mechanism": mech_shuf
                    }
                }

                print(f"    [r={r}, h={h} (+{h*checkpoint_interval} steps)] Cont Acc(NS)={continued_adam[f'h_{h}']['test_acc_nonsym']:.1f}% | "
                      f"Lin={lin_ns_a:.1f}% (cos_full={mech_lin['cos_sim_to_continuation']:.2f}) | "
                      f"Dmp={dmp_ns_a:.1f}% (cos_full={mech_dmp['cos_sim_to_continuation']:.2f}) | "
                      f"TFM={tfm_ns_a:.1f}% (cos_full={mech_tfm['cos_sim_to_continuation']:.2f}) | "
                      f"DecayOnly={seed_entry['baselines']['weight_decay_only'][f'h_{h}']['test_acc_nonsym']:.1f}%", flush=True)

        all_seed_results.append(seed_entry)

    # ==============================================================================
    # Aggregate Summary Statistics Across Seeds with Paired Differences
    # ==============================================================================
    print("\n" + "=" * 80)
    print(" AGGREGATING MULTI-SEED GROKKING BENCHMARK RESULTS (v2.0)")
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
        "radial_variance_ratio": calc_stats([s["radial_var_ratio"] for s in all_seed_results]),
        "baselines": {
            "frozen_acc_all": calc_stats([s["baselines"]["frozen"]["test_acc"] for s in all_seed_results]),
            "frozen_acc_nonsym": calc_stats([s["baselines"]["frozen"]["test_acc_nonsym"] for s in all_seed_results]),
            "frozen_loss": calc_stats([s["baselines"]["frozen"]["test_loss"] for s in all_seed_results]),
            "frozen_fourier_saliency": calc_stats([s["baselines"]["frozen"]["fourier_saliency"] for s in all_seed_results]),
            "swa_acc_nonsym": calc_stats([s["baselines"]["swa"]["test_acc_nonsym"] for s in all_seed_results]),
            "swa_diff_from_frozen": calc_paired_diff_stats(
                [s["baselines"]["swa"]["test_acc_nonsym"] for s in all_seed_results],
                [s["baselines"]["frozen"]["test_acc_nonsym"] for s in all_seed_results]
            ),
            "ema_acc_nonsym": calc_stats([s["baselines"]["ema"]["test_acc_nonsym"] for s in all_seed_results]),
            "ema_diff_from_frozen": calc_paired_diff_stats(
                [s["baselines"]["ema"]["test_acc_nonsym"] for s in all_seed_results],
                [s["baselines"]["frozen"]["test_acc_nonsym"] for s in all_seed_results]
            ),
            "weight_decay_only": {},
            "continued_adam": {}
        },
        "explained_variance": {
            r: calc_stats([s["rank_diagnostics"][r]["explained_variance"] for s in all_seed_results])
            for r in ranks
        },
        "methods": {"linear": {}, "damped": {}, "timesfm": {}, "shuffled_timesfm": {}}
    }

    frozen_nonsym_accs = [s["baselines"]["frozen"]["test_acc_nonsym"] for s in all_seed_results]

    for h in horizons:
        c_accs_ns = [s["baselines"]["continued_adam"][f"h_{h}"]["test_acc_nonsym"] for s in all_seed_results]
        c_accs_all = [s["baselines"]["continued_adam"][f"h_{h}"]["test_acc"] for s in all_seed_results]
        c_sal = [s["baselines"]["continued_adam"][f"h_{h}"]["fourier_saliency"] for s in all_seed_results]
        summary["baselines"]["continued_adam"][f"h_{h}"] = {
            "acc_nonsym": calc_stats(c_accs_ns),
            "acc_all": calc_stats(c_accs_all),
            "fourier_saliency": calc_stats(c_sal),
            "diff_from_frozen": calc_paired_diff_stats(c_accs_ns, frozen_nonsym_accs)
        }

        dec_accs_ns = [s["baselines"]["weight_decay_only"][f"h_{h}"]["test_acc_nonsym"] for s in all_seed_results]
        summary["baselines"]["weight_decay_only"][f"h_{h}"] = {
            "acc_nonsym": calc_stats(dec_accs_ns),
            "diff_from_frozen": calc_paired_diff_stats(dec_accs_ns, frozen_nonsym_accs)
        }

    for method in ["linear", "damped", "timesfm", "shuffled_timesfm"]:
        summary["methods"][method] = {}
        for r in ranks:
            summary["methods"][method][f"rank_{r}"] = {}
            for h in horizons:
                accs_ns = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["test_acc_nonsym"] for s in all_seed_results]
                accs_all = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["test_acc"] for s in all_seed_results]
                cos_full = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["mechanism"]["cos_sim_to_continuation"] for s in all_seed_results]
                rel_l2_full = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["mechanism"]["rel_error_to_continuation"] for s in all_seed_results]
                cos_latent = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["mechanism"]["cos_sim_latent"] for s in all_seed_results]
                rad_pred = [s["evaluations"][f"rank_{r}"][f"horizon_{h}"][method]["mechanism"]["radial_alignment_pred"] for s in all_seed_results]

                summary["methods"][method][f"rank_{r}"][f"horizon_{h}"] = {
                    "acc_nonsym": calc_stats(accs_ns),
                    "acc_all": calc_stats(accs_all),
                    "diff_from_frozen": calc_paired_diff_stats(accs_ns, frozen_nonsym_accs),
                    "cos_sim_full": calc_stats(cos_full),
                    "rel_l2_full": calc_stats(rel_l2_full),
                    "cos_sim_latent": calc_stats(cos_latent),
                    "radial_alignment_pred": calc_stats(rad_pred)
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
        cutoff_step=1500, # Clean pre-grokking floor before any generalization rise
        context_checkpoints=32,
        checkpoint_interval=25,
        ranks=[1, 2, 4, 8],
        horizons=[8, 16, 32, 50, 64, 80, 100]
    )
