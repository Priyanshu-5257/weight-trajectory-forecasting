"""
Publication Figure Generator for SmolGPT Weight Trajectory Forecasting Benchmark.
Generates:
1. smolgpt_ppl_vs_horizon.png (Perplexity vs Horizon across methods with 95% CIs)
2. smolgpt_explained_variance_and_spectrum.png (Singular value spectrum & cumulative variance)
3. smolgpt_cosine_alignment_and_layer_breakdown.png (Directional cosine alignment full & per layer)
4. smolgpt_compute_quality_tradeoff.png (Perplexity vs Compute Latency)
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Artifact directory
ARTIFACT_DIR = "/home/prinshu/.gemini/antigravity-cli/brain/89fca131-5bb0-4d9a-a7fe-bd7e0105d3a5"

# High-quality publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "figure.dpi": 300
})


def copy_to_artifacts(src_path, filename):
    if os.path.exists(ARTIFACT_DIR):
        dst = os.path.join(ARTIFACT_DIR, filename)
        shutil.copy2(src_path, dst)
        print(f"Copied to artifact: {dst}")


def generate_smolgpt_figures():
    raw_path = os.path.join(RESULTS_DIR, "smolgpt_benchmark_results.json")
    summary_path = os.path.join(RESULTS_DIR, "smolgpt_benchmark_summary.json")

    if not os.path.exists(raw_path) or not os.path.exists(summary_path):
        print(f"Results files not found at {summary_path}. Run benchmark first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    with open(raw_path) as f:
        raw_results = json.load(f)

    h_keys = list(summary["horizons"].keys())
    delta_steps = [int(hk.replace("h_", "")) * 20 for hk in h_keys]  # 20 steps per checkpoint interval

    # --------------------------------------------------------------------------
    # Figure 1: Perplexity vs Horizon across methods
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Baseline curves
    adamw_ppl = [summary["horizons"][hk]["methods"]["continued_adamw"]["val_ppl"]["mean"] for hk in h_keys]
    adamw_ci = [summary["horizons"][hk]["methods"]["continued_adamw"]["val_ppl"]["ci95"] for hk in h_keys]

    frozen_ppl = [summary["frozen"]["val_ppl"]["mean"]] * len(delta_steps)
    frozen_ci = [summary["frozen"]["val_ppl"]["ci95"]] * len(delta_steps)

    swa_ppl = [summary["swa"]["val_ppl"]["mean"]] * len(delta_steps)
    swa_ci = [summary["swa"]["val_ppl"]["ci95"]] * len(delta_steps)

    wd_ppl = [summary["horizons"][hk]["methods"]["weight_decay_only"]["val_ppl"]["mean"] for hk in h_keys]

    # Forecast curves (Rank 2 and Rank 4)
    lin2_ppl = [summary["horizons"][hk]["methods"]["low_rank_linear_r2"]["val_ppl"]["mean"] for hk in h_keys]
    lin2_ci = [summary["horizons"][hk]["methods"]["low_rank_linear_r2"]["val_ppl"]["ci95"] for hk in h_keys]

    damp2_ppl = [summary["horizons"][hk]["methods"]["low_rank_damped_r2"]["val_ppl"]["mean"] for hk in h_keys]
    damp2_ci = [summary["horizons"][hk]["methods"]["low_rank_damped_r2"]["val_ppl"]["ci95"] for hk in h_keys]

    tfm2_ppl = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]
    tfm2_ci = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_ppl"]["ci95"] for hk in h_keys]

    shuf2_ppl = [summary["horizons"][hk]["methods"]["shuffled_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]

    # Plot on Left: Perplexity
    ax1.plot(delta_steps, adamw_ppl, "o-", color="#1a73e8", label="Continued AdamW (Ground Truth)", linewidth=2.8)
    ax1.fill_between(delta_steps, np.array(adamw_ppl) - np.array(adamw_ci), np.array(adamw_ppl) + np.array(adamw_ci), color="#1a73e8", alpha=0.15)

    ax1.axhline(frozen_ppl[0], color="#5f6368", linestyle="--", label="Frozen Checkpoint (W_ref)", alpha=0.8)
    ax1.axhline(swa_ppl[0], color="#9c27b0", linestyle=":", label="Matched SWA (last 16 ckpts)", alpha=0.8)

    ax1.plot(delta_steps, tfm2_ppl, "s-", color="#e91e63", label="TimesFM-2.0 (Rank 2)", linewidth=2.4)
    ax1.fill_between(delta_steps, np.array(tfm2_ppl) - np.array(tfm2_ci), np.array(tfm2_ppl) + np.array(tfm2_ci), color="#e91e63", alpha=0.15)

    ax1.plot(delta_steps, damp2_ppl, "^-", color="#ff9800", label="Damped Extrapolation (Rank 2)", linewidth=2.0)
    ax1.plot(delta_steps, lin2_ppl, "d--", color="#4caf50", label="Linear Extrapolation (Rank 2)", linewidth=2.0)
    ax1.plot(delta_steps, shuf2_ppl, "x:", color="#795548", label="Shuffled TimesFM (Control)", alpha=0.7)

    ax1.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax1.set_ylabel("Validation Perplexity (PPL, lower is better)")
    ax1.set_title("SmolGPT Language Model Perplexity vs Horizon")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Plot on Right: Validation Loss
    adamw_loss = [summary["horizons"][hk]["methods"]["continued_adamw"]["val_loss"]["mean"] for hk in h_keys]
    frozen_loss = [summary["frozen"]["val_loss"]["mean"]] * len(delta_steps)
    tfm2_loss = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_loss"]["mean"] for hk in h_keys]
    damp2_loss = [summary["horizons"][hk]["methods"]["low_rank_damped_r2"]["val_loss"]["mean"] for hk in h_keys]
    lin2_loss = [summary["horizons"][hk]["methods"]["low_rank_linear_r2"]["val_loss"]["mean"] for hk in h_keys]

    ax2.plot(delta_steps, adamw_loss, "o-", color="#1a73e8", label="Continued AdamW", linewidth=2.8)
    ax2.axhline(frozen_loss[0], color="#5f6368", linestyle="--", label="Frozen W_ref", alpha=0.8)
    ax2.plot(delta_steps, tfm2_loss, "s-", color="#e91e63", label="TimesFM-2.0 (Rank 2)", linewidth=2.4)
    ax2.plot(delta_steps, damp2_loss, "^-", color="#ff9800", label="Damped (Rank 2)", linewidth=2.0)
    ax2.plot(delta_steps, lin2_loss, "d--", color="#4caf50", label="Linear (Rank 2)", linewidth=2.0)

    ax2.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax2.set_ylabel("Validation Cross-Entropy Loss")
    ax2.set_title("Validation Cross-Entropy Loss vs Horizon")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "smolgpt_ppl_vs_horizon.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig1_path, "smolgpt_ppl_vs_horizon.png")
    print("Generated smolgpt_ppl_vs_horizon.png")

    # --------------------------------------------------------------------------
    # Figure 2: Subspace Spectrum and Cumulative Explained Variance
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # Mean singular values across seeds
    all_s = [s["singular_values"] for s in raw_results]
    mean_s = np.mean(all_s, axis=0)
    indices = np.arange(1, len(mean_s) + 1)

    ax1.plot(indices, mean_s, "o-", color="#1a73e8", markersize=6, linewidth=2.2)
    ax1.set_xlabel("Singular Value Index (i)")
    ax1.set_ylabel("Singular Value (sigma_i)")
    ax1.set_title("SmolGPT Checkpoint Displacement Spectrum (P=1.25M)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Cumulative explained variance
    all_ev = [s["explained_variance_ratio"] for s in raw_results]
    mean_ev_cum = np.mean(np.cumsum(all_ev, axis=1), axis=0) * 100.0

    ax2.plot(indices, mean_ev_cum, "s-", color="#e91e63", markersize=6, linewidth=2.2)
    ax2.axhline(90.0, color="#757575", linestyle="--", alpha=0.7, label="90% Variance")
    ax2.axhline(99.0, color="#4caf50", linestyle=":", alpha=0.7, label="99% Variance")
    for r in [1, 2, 4, 8, 16]:
        ax2.annotate(f"r={r}: {mean_ev_cum[r-1]:.1f}%", (r, mean_ev_cum[r-1]),
                     textcoords="offset points", xytext=(8, -12), fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e91e63", alpha=0.8))

    ax2.set_xlabel("Truncation Rank (r)")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Subspace Variance Concentration in SmolGPT")
    ax2.set_ylim([50, 102])
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "smolgpt_explained_variance_and_spectrum.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig2_path, "smolgpt_explained_variance_and_spectrum.png")
    print("Generated smolgpt_explained_variance_and_spectrum.png")

    # --------------------------------------------------------------------------
    # Figure 3: Full-Parameter & Layer-Wise Cosine Similarity
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # Full parameter cosine similarity
    for method_key, label, col, marker in [
        ("low_rank_timesfm_r2", "TimesFM (Rank 2)", "#e91e63", "s-"),
        ("low_rank_damped_r2", "Damped (Rank 2)", "#ff9800", "^-"),
        ("low_rank_linear_r2", "Linear (Rank 2)", "#4caf50", "d--"),
        ("weight_decay_only", "Weight Decay Only", "#795548", "x:"),
        ("shuffled_timesfm_r2", "Shuffled TimesFM", "#9e9e9e", "v:")
    ]:
        cos_means = [summary["horizons"][hk]["methods"][method_key]["cos_sim"]["mean"] for hk in h_keys]
        cos_cis = [summary["horizons"][hk]["methods"][method_key]["cos_sim"]["ci95"] for hk in h_keys]
        ax1.plot(delta_steps, cos_means, marker, color=col, label=label, linewidth=2.0)
        ax1.fill_between(delta_steps, np.array(cos_means) - np.array(cos_cis), np.array(cos_means) + np.array(cos_cis), color=col, alpha=0.1)

    ax1.set_xlabel("Forecast Horizon (Steps)")
    ax1.set_ylabel("Full-Parameter Cosine Similarity to Continuation")
    ax1.set_title("Directional Alignment vs Optimization Trajectory")
    ax1.set_ylim([-0.2, 1.05])
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Layer-wise breakdown for TimesFM (Rank 2) at largest horizon h=64
    hk_max = h_keys[-1]
    layers = ["embedding", "attention", "mlp", "norm"]
    layer_names = ["Embeddings (42%)", "Attention (16%)", "MLP (42%)", "RMSNorm (0.1%)"]
    tfm_layer_cos = [summary["horizons"][hk_max]["methods"]["low_rank_timesfm_r2"]["layer_cos_sim"][l]["mean"] for l in layers]
    damp_layer_cos = [summary["horizons"][hk_max]["methods"]["low_rank_damped_r2"]["layer_cos_sim"][l]["mean"] for l in layers]
    lin_layer_cos = [summary["horizons"][hk_max]["methods"]["low_rank_linear_r2"]["layer_cos_sim"][l]["mean"] for l in layers]

    x_idx = np.arange(len(layers))
    w = 0.25
    ax2.bar(x_idx - w, tfm_layer_cos, width=w, color="#e91e63", label="TimesFM (r=2)")
    ax2.bar(x_idx, damp_layer_cos, width=w, color="#ff9800", label="Damped (r=2)")
    ax2.bar(x_idx + w, lin_layer_cos, width=w, color="#4caf50", label="Linear (r=2)")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(layer_names)
    ax2.set_ylabel(f"Cosine Similarity at +{delta_steps[-1]} Steps")
    ax2.set_title("Layer-Wise Alignment to Continuation")
    ax2.set_ylim([-0.1, 1.05])
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "smolgpt_cosine_alignment_and_layer_breakdown.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig3_path, "smolgpt_cosine_alignment_and_layer_breakdown.png")
    print("Generated smolgpt_cosine_alignment_and_layer_breakdown.png")

    # --------------------------------------------------------------------------
    # Figure 4: Compute Overhead & Speedup Tradeoff
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.0))

    methods_plot = [
        ("continued_adamw", "Continued AdamW", "#1a73e8", "o"),
        ("low_rank_timesfm_r2", "TimesFM (r=2)", "#e91e63", "s"),
        ("low_rank_damped_r2", "Damped (r=2)", "#ff9800", "^"),
        ("low_rank_linear_r2", "Linear (r=2)", "#4caf50", "d"),
        ("weight_decay_only", "Weight Decay Only", "#795548", "x")
    ]

    for m_key, m_label, col, mark in methods_plot:
        # At horizon h=32 (+640 steps)
        hk_target = "h_32"
        t_ms = summary["horizons"][hk_target]["methods"][m_key]["time_ms"]["mean"]
        ppl = summary["horizons"][hk_target]["methods"][m_key]["val_ppl"]["mean"]
        ax.scatter(t_ms, ppl, color=col, s=120, marker=mark, label=m_label, zorder=5)
        ax.annotate(f"{m_label}\n({t_ms:.1f}ms, PPL={ppl:.1f})", (t_ms, ppl),
                    textcoords="offset points", xytext=(8, -8), fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    ax.axhline(summary["frozen"]["val_ppl"]["mean"], color="#5f6368", linestyle="--", label="Frozen W_ref", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Execution Time (ms, log scale)")
    ax.set_ylabel("Validation Perplexity at +640 Steps (lower is better)")
    ax.set_title("Compute vs Validation Quality Tradeoff on SmolGPT")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "smolgpt_compute_quality_tradeoff.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig4_path, "smolgpt_compute_quality_tradeoff.png")
    print("Generated smolgpt_compute_quality_tradeoff.png")


if __name__ == "__main__":
    generate_smolgpt_figures()
