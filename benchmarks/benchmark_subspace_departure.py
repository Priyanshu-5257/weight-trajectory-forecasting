import os
import sys
import time
import json
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import (
    GrokkingTransformer,
    eval_grokking_tensors,
    get_flat_params,
    set_flat_params
)
from src.data import load_modular_addition_dataset
from src.subspace_analysis import (
    get_layer_parameter_indices,
    compute_svd_basis,
    compute_radial_decay_alignment,
    project_displacement,
    qr_project_displacement
)

# ==============================================================================
# Statistics Helpers
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

# ==============================================================================
# Subspace Departure Benchmark Runner
# ==============================================================================

def run_subspace_departure_benchmark(
    seeds=list(range(42, 47)), # 5 paired seeds
    p=97,
    frac_train=0.5,
    cutoff_step=1500,
    context_checkpoints=32,
    checkpoint_interval=25,
    ranks=[1, 2, 4, 8, 16, 30, 31, 32],
    horizons=[8, 16, 32, 50, 64, 80, 100], # +200, +400, +800, +1250, +1600, +2000, +2500 steps
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" HISTORICAL WEIGHT SUBSPACE DEPARTURE & GROKKING BENCHMARK")
    print(f" Modulus P={p} | Frac Train={frac_train*100:.0f}% | Cutoff Step={cutoff_step}")
    print(f" Context: {context_checkpoints} checkpoints every {checkpoint_interval} steps (Steps {cutoff_step - context_checkpoints*checkpoint_interval} -> {cutoff_step})")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizons: {horizons}")
    print(f" Device: {device}")
    print("=" * 80)

    total_training_steps = cutoff_step + max(horizons) * checkpoint_interval
    criterion = nn.CrossEntropyLoss()
    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n----------------------------------------------------------------------")
        print(f"--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        print(f"----------------------------------------------------------------------")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        data = load_modular_addition_dataset(p=p, frac_train=frac_train, seed=seed)
        train_x = data["train_data"].to(device)
        train_y = data["train_labels"].to(device)
        test_x = data["test_data"].to(device)
        test_y = data["test_labels"].to(device)

        # Strictly non-symmetric test pairs
        train_pairs = set([(int(x[0]), int(x[1])) for x in train_x.cpu()])
        non_sym_idx = [i for i, (a, b, _) in enumerate(test_x.cpu()) if (int(b), int(a)) not in train_pairs]
        non_sym_x = test_x[non_sym_idx]
        non_sym_y = test_y[non_sym_idx]

        model = GrokkingTransformer(p=p).to(device)
        layer_indices = get_layer_parameter_indices(model)
        layer_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in layer_indices.items()}
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)

        start_record_step = cutoff_step - (context_checkpoints * checkpoint_interval)
        saved_checkpoints = {} # step -> flat_params (cpu)

        eval_steps_set = set([cutoff_step + h * checkpoint_interval for h in horizons])
        eval_steps_set.add(cutoff_step)

        print(f"Training base model through step {total_training_steps}...", flush=True)
        t0_train = time.perf_counter()

        for step in range(1, total_training_steps + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()

            # Record context checkpoints
            if step >= start_record_step and step <= cutoff_step and step % checkpoint_interval == 0:
                saved_checkpoints[step] = get_flat_params(model).cpu()

            # Record future horizon checkpoints
            if step in eval_steps_set:
                saved_checkpoints[step] = get_flat_params(model).cpu()

            if step in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
                with torch.no_grad():
                    _, test_acc_all = eval_grokking_tensors(model, test_x, test_y, criterion)
                    _, test_acc_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)
                    print(f"  Step {step:5d} | Train Loss: {loss.item():.4f} | Test (All): {test_acc_all:5.1f}% | Test (NonSym): {test_acc_ns:5.2f}%", flush=True)

        t_train = time.perf_counter() - t0_train
        print(f"Base training finished in {t_train:.2f} s")

        w_ref = saved_checkpoints[cutoff_step] # (P,)
        hist_steps = [cutoff_step - (context_checkpoints - 1 - i) * checkpoint_interval for i in range(context_checkpoints)]
        assert len(hist_steps) == context_checkpoints, f"Expected {context_checkpoints} checkpoints, found {len(hist_steps)}"

        # Construct historical displacement matrix: (P, T)
        D_full = torch.stack([saved_checkpoints[s] - w_ref for s in hist_steps], dim=1)
        U_full, S_full, cum_var_full, diag_full = compute_svd_basis(D_full, return_diagnostics=True)

        # 1. Cosine alignment with radial decay direction
        cos_radial_full = compute_radial_decay_alignment(U_full, w_ref)
        cos_radial_layers = {}
        U_layers = {}
        S_layers = {}
        cum_var_layers = {}
        diag_layers = {}

        for layer_name, idx_tensor in layer_tensors.items():
            D_layer = D_full[idx_tensor]
            w_ref_layer = w_ref[idx_tensor]
            U_l, S_l, cum_var_l, diag_l = compute_svd_basis(D_layer, return_diagnostics=True)
            U_layers[layer_name] = U_l
            S_layers[layer_name] = S_l
            cum_var_layers[layer_name] = cum_var_l
            diag_layers[layer_name] = diag_l
            cos_radial_layers[layer_name] = compute_radial_decay_alignment(U_l, w_ref_layer)

        print(f"  Radial Decay Alignment | Full: {cos_radial_full:.4f} | Embed: {cos_radial_layers['embed']:.4f} | Attn: {cos_radial_layers['attn']:.4f} | MLP: {cos_radial_layers['mlp']:.4f} | Unembed: {cos_radial_layers['unembed']:.4f} | Cond: {diag_full['condition_number']:.1f} | Max Rank: {diag_full['max_rank']}")

        # 2. Evaluation across horizons
        recon_ranks = [1, 2, 4, 8, 16, 30, 31]
        horizon_results = {}
        for h in horizons:
            fut_step = cutoff_step + h * checkpoint_interval
            w_act = saved_checkpoints[fut_step]
            dw_act = w_act - w_ref

            # Evaluate Actual Continuation
            set_flat_params(model, w_act.to(device))
            loss_act, acc_act_all = eval_grokking_tensors(model, test_x, test_y, criterion)
            _, acc_act_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

            h_res = {
                "step": fut_step,
                "delta_steps": h * checkpoint_interval,
                "actual": {
                    "loss": loss_act,
                    "acc_all": acc_act_all,
                    "acc_nonsym": acc_act_ns
                },
                "coverage_full": {},
                "coverage_layers": {l: {} for l in layer_tensors.keys()},
                "rank_accounting": {},
                "reconstructed_models": {},
                "qr_projection_check": {}
            }

            # Independent QR projection on full displacement matrix
            dw_par_qr, dw_perp_qr, cov_qr = qr_project_displacement(D_full, dw_act)
            h_res["qr_projection_check"] = {
                "cov_qr": cov_qr
            }

            # Subspace predictive coverage across ranks
            for r in ranks:
                actual_r = min(r, U_full.shape[1])
                U_r = U_full[:, :actual_r]
                dw_par, dw_perp, cov_full = project_displacement(dw_act, U_r)
                h_res["coverage_full"][f"r_{r}"] = cov_full
                h_res["rank_accounting"][f"r_{r}"] = {
                    "requested_rank": r,
                    "actual_rank": actual_r
                }

                # Layer-wise predictive coverage
                for layer_name, idx_tensor in layer_tensors.items():
                    actual_r_layer = min(r, U_layers[layer_name].shape[1])
                    U_layer_r = U_layers[layer_name][:, :actual_r_layer]
                    dw_layer = dw_act[idx_tensor]
                    _, _, cov_layer = project_displacement(dw_layer, U_layer_r)
                    h_res["coverage_layers"][layer_name][f"r_{r}"] = cov_layer

                # Reconstruct models for all key ranks: r in recon_ranks
                if r in recon_ranks:
                    # Projected continuation model
                    w_proj = w_ref + dw_par
                    set_flat_params(model, w_proj.to(device))
                    loss_proj, acc_proj_all = eval_grokking_tensors(model, test_x, test_y, criterion)
                    _, acc_proj_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

                    # Orthogonal continuation model
                    w_orth = w_ref + dw_perp
                    set_flat_params(model, w_orth.to(device))
                    loss_orth, acc_orth_all = eval_grokking_tensors(model, test_x, test_y, criterion)
                    _, acc_orth_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

                    h_res["reconstructed_models"][f"r_{r}"] = {
                        "requested_rank": r,
                        "actual_rank": actual_r,
                        "projected": {
                            "loss": loss_proj,
                            "acc_all": acc_proj_all,
                            "acc_nonsym": acc_proj_ns
                        },
                        "orthogonal": {
                            "loss": loss_orth,
                            "acc_all": acc_orth_all,
                            "acc_nonsym": acc_orth_ns
                        }
                    }

            # Check residual between max rank SVD and QR
            max_r_key = f"r_{min(31, max(ranks))}"
            if max_r_key in h_res["coverage_full"]:
                h_res["qr_projection_check"]["cov_svd_max"] = h_res["coverage_full"][max_r_key]
                h_res["qr_projection_check"]["abs_diff"] = abs(h_res["coverage_full"][max_r_key] - cov_qr)

            # Layer-selective ablation at rank 8:
            # Test whether keeping layer l actual while projecting all other layers recovers generalization
            h_res["layer_selective_ablation_r8"] = {}
            U_8_full = U_full[:, :min(8, U_full.shape[1])]
            dw_proj_8, _, _ = project_displacement(dw_act, U_8_full)

            for layer_name, idx_tensor in layer_tensors.items():
                # Hybrid A: Layer l is ACTUAL, all other layers are PROJECTED
                w_hybrid_actual_l = (w_ref + dw_proj_8).clone()
                w_hybrid_actual_l[idx_tensor] = w_act[idx_tensor]
                set_flat_params(model, w_hybrid_actual_l.to(device))
                _, acc_hybrid_act_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

                # Hybrid B: Layer l is PROJECTED, all other layers are ACTUAL
                w_hybrid_proj_l = w_act.clone()
                w_hybrid_proj_l[idx_tensor] = (w_ref + dw_proj_8)[idx_tensor]
                set_flat_params(model, w_hybrid_proj_l.to(device))
                _, acc_hybrid_proj_ns = eval_grokking_tensors(model, non_sym_x, non_sym_y, criterion)

                h_res["layer_selective_ablation_r8"][layer_name] = {
                    "layer_actual_rest_projected_acc_nonsym": acc_hybrid_act_ns,
                    "layer_projected_rest_actual_acc_nonsym": acc_hybrid_proj_ns
                }

            horizon_results[f"h_{h}"] = h_res
            print(f"    [h={h:3d} (+{h*checkpoint_interval:4d}s)] Actual Acc(NS): {acc_act_ns:5.1f}% | Cov(r=8): {h_res['coverage_full']['r_8']*100:5.1f}% | Cov(r=31): {h_res['coverage_full'].get('r_31', 0.0)*100:5.1f}% | Proj(r=31) Acc(NS): {h_res['reconstructed_models'].get('r_31', {}).get('projected', {}).get('acc_nonsym', 0.0):5.1f}% | QR Diff: {h_res['qr_projection_check'].get('abs_diff', 0.0):.2e}")

        seed_data = {
            "seed": seed,
            "diagnostics_full": diag_full,
            "diagnostics_layers": diag_layers,
            "effective_rank_full": diag_full["r_mach_64"],
            "effective_rank_layers": {l: diag_layers[l]["r_mach_64"] for l in layer_tensors.keys()},
            "cos_radial_full": cos_radial_full,
            "cos_radial_layers": cos_radial_layers,
            "exp_var_full": {f"r_{r}": float(cum_var_full[min(r, len(cum_var_full))-1].item()) for r in ranks},
            "exp_var_layers": {
                l: {f"r_{r}": float(cum_var_layers[l][min(r, len(cum_var_layers[l]))-1].item()) for r in ranks} for l in layer_tensors.keys()
            },
            "horizons": horizon_results
        }
        all_seed_results.append(seed_data)

    # ==============================================================================
    # Aggregate Multi-Seed Summary Statistics
    # ==============================================================================
    print("\n" + "=" * 80)
    print(" AGGREGATING MULTI-SEED SUBSPACE DEPARTURE METRICS")
    print("=" * 80)

    summary = {
        "metadata": {
            "version": "2.2",
            "fix": "float64 SVD & QR verification; explicit numerical and energy rank spectra",
            "cutoff_step": cutoff_step,
            "eval_split": "nonsym_pairs",
            "seeds": seeds,
            "checkpoint_interval": checkpoint_interval,
            "context_checkpoints": context_checkpoints
        },
        "seeds": seeds,
        "p": p,
        "frac_train": frac_train,
        "cutoff_step": cutoff_step,
        "checkpoint_interval": checkpoint_interval,
        "ranks": ranks,
        "recon_ranks": recon_ranks,
        "horizons": horizons,
        "spectrum_diagnostics_full": {
            "condition_number": calc_stats([s["diagnostics_full"]["condition_number"] for s in all_seed_results]),
            "r_mach_64": calc_stats([s["diagnostics_full"]["r_mach_64"] for s in all_seed_results]),
            "r_mach_32": calc_stats([s["diagnostics_full"]["r_mach_32"] for s in all_seed_results]),
            "r_rel_1e3": calc_stats([s["diagnostics_full"]["r_rel_1e3"] for s in all_seed_results]),
            "r_abs_1e6": calc_stats([s["diagnostics_full"]["r_abs_1e6"] for s in all_seed_results]),
            "r_energy_99": calc_stats([s["diagnostics_full"]["r_energy_99"] for s in all_seed_results]),
            "r_energy_999": calc_stats([s["diagnostics_full"]["r_energy_999"] for s in all_seed_results]),
            "r_energy_9999": calc_stats([s["diagnostics_full"]["r_energy_9999"] for s in all_seed_results]),
            "singular_values_mean": [float(np.mean([s["diagnostics_full"]["singular_values"][i] for s in all_seed_results])) for i in range(len(all_seed_results[0]["diagnostics_full"]["singular_values"]))],
            "singular_values_std": [float(np.std([s["diagnostics_full"]["singular_values"][i] for s in all_seed_results])) for i in range(len(all_seed_results[0]["diagnostics_full"]["singular_values"]))]
        },
        "effective_rank_full": calc_stats([s["effective_rank_full"] for s in all_seed_results]),
        "cos_radial_full": calc_stats([s["cos_radial_full"] for s in all_seed_results]),
        "cos_radial_layers": {
            l: calc_stats([s["cos_radial_layers"][l] for s in all_seed_results]) for l in ["embed", "attn", "mlp", "unembed"]
        },
        "exp_var_full": {
            f"r_{r}": calc_stats([s["exp_var_full"][f"r_{r}"] for s in all_seed_results]) for r in ranks
        },
        "exp_var_layers": {
            l: {
                f"r_{r}": calc_stats([s["exp_var_layers"][l][f"r_{r}"] for s in all_seed_results]) for r in ranks
            } for l in ["embed", "attn", "mlp", "unembed"]
        },
        "coverage_full": {f"r_{r}": {} for r in ranks},
        "coverage_layers": {l: {f"r_{r}": {} for r in ranks} for l in ["embed", "attn", "mlp", "unembed"]},
        "actual_continuation": {},
        "reconstructed_models": {
            f"r_{r}": {"projected": {}, "orthogonal": {}, "paired_diff_projected_vs_actual": {}} for r in recon_ranks
        },
        "qr_vs_svd_max_diff": {},
        "layer_selective_ablation_r8": {l: {} for l in ["embed", "attn", "mlp", "unembed"]}
    }

    for h in horizons:
        # Actual continuation
        summary["actual_continuation"][f"h_{h}"] = {
            "acc_nonsym": calc_stats([s["horizons"][f"h_{h}"]["actual"]["acc_nonsym"] for s in all_seed_results]),
            "acc_all": calc_stats([s["horizons"][f"h_{h}"]["actual"]["acc_all"] for s in all_seed_results]),
            "loss": calc_stats([s["horizons"][f"h_{h}"]["actual"]["loss"] for s in all_seed_results])
        }

        # QR vs SVD diff
        summary["qr_vs_svd_max_diff"][f"h_{h}"] = calc_stats([s["horizons"][f"h_{h}"]["qr_projection_check"].get("abs_diff", 0.0) for s in all_seed_results])

        # Subspace predictive coverage across ranks
        for r in ranks:
            covs_full = [s["horizons"][f"h_{h}"]["coverage_full"][f"r_{r}"] for s in all_seed_results]
            summary["coverage_full"][f"r_{r}"][f"h_{h}"] = calc_stats(covs_full)

            for l in ["embed", "attn", "mlp", "unembed"]:
                covs_layer = [s["horizons"][f"h_{h}"]["coverage_layers"][l][f"r_{r}"] for s in all_seed_results]
                summary["coverage_layers"][l][f"r_{r}"][f"h_{h}"] = calc_stats(covs_layer)

        # Reconstructed models for r in recon_ranks
        for r in recon_ranks:
            act_accs = [s["horizons"][f"h_{h}"]["actual"]["acc_nonsym"] for s in all_seed_results]
            proj_accs = [s["horizons"][f"h_{h}"]["reconstructed_models"][f"r_{r}"]["projected"]["acc_nonsym"] for s in all_seed_results]
            orth_accs = [s["horizons"][f"h_{h}"]["reconstructed_models"][f"r_{r}"]["orthogonal"]["acc_nonsym"] for s in all_seed_results]

            summary["reconstructed_models"][f"r_{r}"]["projected"][f"h_{h}"] = calc_stats(proj_accs)
            summary["reconstructed_models"][f"r_{r}"]["orthogonal"][f"h_{h}"] = calc_stats(orth_accs)
            summary["reconstructed_models"][f"r_{r}"]["paired_diff_projected_vs_actual"][f"h_{h}"] = calc_paired_diff_stats(proj_accs, act_accs)

        # Layer-selective ablation at r=8
        for l in ["embed", "attn", "mlp", "unembed"]:
            act_l = [s["horizons"][f"h_{h}"]["layer_selective_ablation_r8"][l]["layer_actual_rest_projected_acc_nonsym"] for s in all_seed_results]
            proj_l = [s["horizons"][f"h_{h}"]["layer_selective_ablation_r8"][l]["layer_projected_rest_actual_acc_nonsym"] for s in all_seed_results]
            summary["layer_selective_ablation_r8"][l][f"h_{h}"] = {
                "layer_actual_rest_projected": calc_stats(act_l),
                "layer_projected_rest_actual": calc_stats(proj_l)
            }

    # Save outputs
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    raw_path = os.path.join(results_dir, "subspace_departure_results.json")
    summary_path = os.path.join(results_dir, "subspace_departure_summary.json")

    with open(raw_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] Raw results: {raw_path}")
    print(f"[Saved] Summary statistics: {summary_path}")
    return summary

if __name__ == "__main__":
    run_subspace_departure_benchmark(
        seeds=list(range(42, 47)),
        p=97,
        frac_train=0.5,
        cutoff_step=1500,
        context_checkpoints=32,
        checkpoint_interval=25,
        ranks=[1, 2, 4, 8, 16, 30, 31, 32],
        horizons=[8, 16, 32, 50, 64, 80, 100]
    )
