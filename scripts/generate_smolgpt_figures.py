"""
Publication Figure Generator for SmolGPT Weight Trajectory Forecasting Benchmark (v2.0).
Generates:
1. smolgpt_train_vs_val_loss.png (Train Loss vs Val Loss across horizons and methods)
2. smolgpt_performance_vs_rank.png (Perplexity vs Truncation Rank r in [1, 2, 4, 8, 16, 24, 32])
3. smolgpt_ppl_vs_horizon.png (Perplexity vs Horizon across methods with 95% CIs)
4. smolgpt_explained_variance_and_spectrum.png (Singular value spectrum & cumulative variance)
5. smolgpt_cosine_alignment_and_layer_breakdown.png (Directional cosine alignment full & per layer)
6. smolgpt_compute_quality_tradeoff.png (Perplexity vs Compute Latency)
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
    delta_steps = [int(hk.replace("h_", "")) * 20 for hk in h_keys]
    ranks = summary["ranks"]

    # --------------------------------------------------------------------------
    # Figure 1: Train Loss vs Val Loss across Horizons
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods_plot = [
        ("continued_adamw", "Continued AdamW", "#1a73e8", "o-", 2.6),
        ("matched_swa", "Matched SWA (last 16)", "#9c27b0", ":", 2.2),
        ("low_rank_timesfm_r2", "TimesFM (Rank 2)", "#e91e63", "s-", 2.2),
        ("low_rank_damped_r2", "Damped (Rank 2)", "#ff9800", "^-", 2.0),
        ("low_rank_linear_r2", "Linear (Rank 2)", "#4caf50", "d--", 2.0),
        ("shuffled_timesfm_r2", "Shuffled TimesFM", "#795548", "x:", 1.8),
    ]

    # Left: Train Cross-Entropy Loss
    frozen_train = summary["frozen"]["train_loss"]["mean"]
    swa_train = summary["swa"]["train_loss"]["mean"]

    for m_key, m_label, col, lstyle, lwidth in methods_plot:
        if m_key == "matched_swa":
            ax1.axhline(swa_train, color=col, linestyle=":", label=m_label, alpha=0.8, linewidth=lwidth)
            continue
        vals = [summary["horizons"][hk]["methods"][m_key]["train_loss"]["mean"] for hk in h_keys]
        cis = [summary["horizons"][hk]["methods"][m_key]["train_loss"]["ci95"] for hk in h_keys]
        # Clip display for exploding linear
        vals_clipped = [min(v, 4.0) for v in vals]
        ax1.plot(delta_steps, vals_clipped, lstyle, color=col, label=m_label, linewidth=lwidth)
        if m_key in ["continued_adamw", "low_rank_timesfm_r2"]:
            ax1.fill_between(delta_steps, np.array(vals) - np.array(cis), np.array(vals) + np.array(cis), color=col, alpha=0.12)

    ax1.axhline(frozen_train, color="#5f6368", linestyle="--", label="Frozen Checkpoint", alpha=0.8)
    ax1.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax1.set_ylabel("Train Cross-Entropy Loss (40,960 tokens)")
    ax1.set_title("Train Loss vs Forecast Horizon")
    ax1.set_ylim([2.2, 3.8])
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Right: Validation Cross-Entropy Loss
    frozen_val = summary["frozen"]["val_loss"]["mean"]
    swa_val = summary["swa"]["val_loss"]["mean"]

    for m_key, m_label, col, lstyle, lwidth in methods_plot:
        if m_key == "matched_swa":
            ax2.axhline(swa_val, color=col, linestyle=":", label=m_label, alpha=0.8, linewidth=lwidth)
            continue
        vals = [summary["horizons"][hk]["methods"][m_key]["val_loss"]["mean"] for hk in h_keys]
        cis = [summary["horizons"][hk]["methods"][m_key]["val_loss"]["ci95"] for hk in h_keys]
        vals_clipped = [min(v, 4.0) for v in vals]
        ax2.plot(delta_steps, vals_clipped, lstyle, color=col, label=m_label, linewidth=lwidth)
        if m_key in ["continued_adamw", "low_rank_timesfm_r2"]:
            ax2.fill_between(delta_steps, np.array(vals) - np.array(cis), np.array(vals) + np.array(cis), color=col, alpha=0.12)

    ax2.axhline(frozen_val, color="#5f6368", linestyle="--", label="Frozen Checkpoint", alpha=0.8)
    ax2.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax2.set_ylabel("Validation Cross-Entropy Loss (40,960 tokens)")
    ax2.set_title("Validation Loss vs Forecast Horizon")
    ax2.set_ylim([2.4, 3.8])
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "smolgpt_train_vs_val_loss.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig1_path, "smolgpt_train_vs_val_loss.png")
    print("Generated smolgpt_train_vs_val_loss.png")

    # --------------------------------------------------------------------------
    # Figure 2: Performance vs Extended Rank (r in [1, 2, 4, 8, 16, 24, 32])
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, hk, h_title in [(ax1, "h_8", "+160 Steps (Short Horizon)"), (ax2, "h_32", "+640 Steps (Long Horizon)")]:
        frozen_ppl = summary["frozen"]["val_ppl"]["mean"]
        adamw_ppl = summary["horizons"][hk]["methods"]["continued_adamw"]["val_ppl"]["mean"]
        swa_ppl = summary["swa"]["val_ppl"]["mean"]

        tfm_ppl = [summary["horizons"][hk]["methods"][f"low_rank_timesfm_r{r}"]["val_ppl"]["mean"] for r in ranks]
        tfm_ci = [summary["horizons"][hk]["methods"][f"low_rank_timesfm_r{r}"]["val_ppl"]["ci95"] for r in ranks]

        shuf_ppl = [summary["horizons"][hk]["methods"][f"shuffled_timesfm_r{r}"]["val_ppl"]["mean"] for r in ranks]

        damp_ppl = [summary["horizons"][hk]["methods"][f"low_rank_damped_r{r}"]["val_ppl"]["mean"] for r in ranks]

        lin_ppl = [min(summary["horizons"][hk]["methods"][f"low_rank_linear_r{r}"]["val_ppl"]["mean"], 100.0) for r in ranks]

        ax.plot(ranks, tfm_ppl, "s-", color="#e91e63", label="TimesFM-2.0 (Ordered)", linewidth=2.4)
        ax.fill_between(ranks, np.array(tfm_ppl) - np.array(tfm_ci), np.array(tfm_ppl) + np.array(tfm_ci), color="#e91e63", alpha=0.15)

        ax.plot(ranks, shuf_ppl, "x:", color="#795548", label="Shuffled TimesFM (Control)", linewidth=2.0)
        ax.plot(ranks, damp_ppl, "^-", color="#ff9800", label="Damped Extrapolation", linewidth=2.0)
        ax.plot(ranks, lin_ppl, "d--", color="#4caf50", label="Linear Extrapolation", linewidth=1.8)

        ax.axhline(adamw_ppl, color="#1a73e8", linestyle="-", label=f"Continued AdamW ({adamw_ppl:.2f})", linewidth=2.2)
        ax.axhline(frozen_ppl, color="#5f6368", linestyle="--", label=f"Frozen Checkpoint ({frozen_ppl:.2f})", alpha=0.8)
        ax.axhline(swa_ppl, color="#9c27b0", linestyle=":", label=f"Matched SWA ({swa_ppl:.2f})", alpha=0.8)

        ax.set_xlabel("Truncation Rank (r)")
        ax.set_ylabel("Validation Perplexity (PPL, lower is better)")
        ax.set_title(f"Validation PPL vs Rank at {h_title}")
        ax.set_xticks(ranks)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "smolgpt_performance_vs_rank.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig2_path, "smolgpt_performance_vs_rank.png")
    print("Generated smolgpt_performance_vs_rank.png")

    # --------------------------------------------------------------------------
    # Figure 3: Perplexity vs Horizon across methods
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    adamw_ppl = [summary["horizons"][hk]["methods"]["continued_adamw"]["val_ppl"]["mean"] for hk in h_keys]
    adamw_ci = [summary["horizons"][hk]["methods"]["continued_adamw"]["val_ppl"]["ci95"] for hk in h_keys]

    frozen_ppl = [summary["frozen"]["val_ppl"]["mean"]] * len(delta_steps)
    swa_ppl = [summary["swa"]["val_ppl"]["mean"]] * len(delta_steps)

    tfm1_ppl = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r1"]["val_ppl"]["mean"] for hk in h_keys]
    tfm2_ppl = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]
    tfm2_ci = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_ppl"]["ci95"] for hk in h_keys]
    tfm16_ppl = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r16"]["val_ppl"]["mean"] for hk in h_keys]
    tfm32_ppl = [summary["horizons"][hk]["methods"]["low_rank_timesfm_r32"]["val_ppl"]["mean"] for hk in h_keys]

    shuf2_ppl = [summary["horizons"][hk]["methods"]["shuffled_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]
    damp2_ppl = [summary["horizons"][hk]["methods"]["low_rank_damped_r2"]["val_ppl"]["mean"] for hk in h_keys]
    lin2_ppl = [min(summary["horizons"][hk]["methods"]["low_rank_linear_r2"]["val_ppl"]["mean"], 100.0) for hk in h_keys]

    ax1.plot(delta_steps, adamw_ppl, "o-", color="#1a73e8", label="Continued AdamW", linewidth=2.8)
    ax1.fill_between(delta_steps, np.array(adamw_ppl) - np.array(adamw_ci), np.array(adamw_ppl) + np.array(adamw_ci), color="#1a73e8", alpha=0.15)

    ax1.axhline(frozen_ppl[0], color="#5f6368", linestyle="--", label="Frozen Checkpoint", alpha=0.8)
    ax1.axhline(swa_ppl[0], color="#9c27b0", linestyle=":", label="Matched SWA (last 16 ckpts)", alpha=0.8)

    ax1.plot(delta_steps, tfm2_ppl, "s-", color="#e91e63", label="TimesFM (Rank 2)", linewidth=2.4)
    ax1.fill_between(delta_steps, np.array(tfm2_ppl) - np.array(tfm2_ci), np.array(tfm2_ppl) + np.array(tfm2_ci), color="#e91e63", alpha=0.15)

    ax1.plot(delta_steps, damp2_ppl, "^-", color="#ff9800", label="Damped (Rank 2)", linewidth=2.0)
    ax1.plot(delta_steps, lin2_ppl, "d--", color="#4caf50", label="Linear (Rank 2)", linewidth=2.0)
    ax1.plot(delta_steps, shuf2_ppl, "x:", color="#795548", label="Shuffled TimesFM (Control)", alpha=0.7)

    ax1.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax1.set_ylabel("Validation Perplexity (PPL, lower is better)")
    ax1.set_title("SmolGPT Perplexity vs Horizon")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Right: TimesFM Rank Comparison across Horizons
    ax2.plot(delta_steps, tfm1_ppl, "o-", color="#9c27b0", label="TimesFM (Rank 1)", linewidth=2.0)
    ax2.plot(delta_steps, tfm2_ppl, "s-", color="#e91e63", label="TimesFM (Rank 2)", linewidth=2.4)
    ax2.plot(delta_steps, tfm16_ppl, "^-", color="#ff5722", label="TimesFM (Rank 16)", linewidth=2.0)
    ax2.plot(delta_steps, tfm32_ppl, "d-", color="#3f51b5", label="TimesFM (Rank 32, Full)", linewidth=2.0)
    ax2.axhline(frozen_ppl[0], color="#5f6368", linestyle="--", label="Frozen Checkpoint", alpha=0.8)
    ax2.axhline(swa_ppl[0], color="#4caf50", linestyle=":", label="Matched SWA", alpha=0.8)

    ax2.set_xlabel("Forecast Horizon (Future Training Steps)")
    ax2.set_ylabel("Validation Perplexity (PPL)")
    ax2.set_title("TimesFM Forecast Quality Across Ranks")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "smolgpt_ppl_vs_horizon.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig3_path, "smolgpt_ppl_vs_horizon.png")
    print("Generated smolgpt_ppl_vs_horizon.png")

    # --------------------------------------------------------------------------
    # Figure 4: Subspace Spectrum and Cumulative Explained Variance
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    all_s = [s["singular_values"] for s in raw_results]
    mean_s = np.mean(all_s, axis=0)
    indices = np.arange(1, len(mean_s) + 1)

    ax1.plot(indices, mean_s, "o-", color="#1a73e8", markersize=6, linewidth=2.2)
    ax1.set_xlabel("Singular Value Index (i)")
    ax1.set_ylabel("Singular Value (sigma_i)")
    ax1.set_title("SmolGPT Checkpoint Displacement Spectrum (P=1.25M)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    all_ev = [s["explained_variance_ratio"] for s in raw_results]
    mean_ev_cum = np.mean(np.cumsum(all_ev, axis=1), axis=0) * 100.0

    ax2.plot(indices, mean_ev_cum, "s-", color="#e91e63", markersize=6, linewidth=2.2)
    ax2.axhline(90.0, color="#757575", linestyle="--", alpha=0.7, label="90% Variance")
    ax2.axhline(99.0, color="#4caf50", linestyle=":", alpha=0.7, label="99% Variance")
    for r in [1, 2, 4, 8, 16, 24, 32]:
        ax2.annotate(f"r={r}: {mean_ev_cum[r-1]:.1f}%", (r, mean_ev_cum[r-1]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8.5,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e91e63", alpha=0.8))

    ax2.set_xlabel("Truncation Rank (r)")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Subspace Variance Concentration in SmolGPT")
    ax2.set_ylim([50, 102])
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "smolgpt_explained_variance_and_spectrum.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig4_path, "smolgpt_explained_variance_and_spectrum.png")
    print("Generated smolgpt_explained_variance_and_spectrum.png")

    # --------------------------------------------------------------------------
    # Figure 5: Directional Cosine Alignment & Layer Breakdown
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

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
    ax2.set_ylim([-0.2, 1.05])
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "smolgpt_cosine_alignment_and_layer_breakdown.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig5_path, "smolgpt_cosine_alignment_and_layer_breakdown.png")
    print("Generated smolgpt_cosine_alignment_and_layer_breakdown.png")

    # --------------------------------------------------------------------------
    # Figure 6: Compute Overhead & Speedup Tradeoff
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.0))

    methods_plot_scatter = [
        ("continued_adamw", "Continued AdamW", "#1a73e8", "o"),
        ("low_rank_timesfm_r2", "TimesFM (r=2)", "#e91e63", "s"),
        ("low_rank_timesfm_r32", "TimesFM (r=32)", "#3f51b5", "P"),
        ("low_rank_damped_r2", "Damped (r=2)", "#ff9800", "^"),
        ("low_rank_linear_r2", "Linear (r=2)", "#4caf50", "d"),
        ("weight_decay_only", "Weight Decay Only", "#795548", "x")
    ]

    for m_key, m_label, col, mark in methods_plot_scatter:
        hk_target = "h_32"
        t_ms = summary["horizons"][hk_target]["methods"][m_key]["time_ms"]["mean"]
        ppl = summary["horizons"][hk_target]["methods"][m_key]["val_ppl"]["mean"]
        ax.scatter(t_ms, ppl, color=col, s=120, marker=mark, label=m_label, zorder=5)
        ax.annotate(f"{m_label}\n({t_ms:.1f}ms, PPL={ppl:.1f})", (t_ms, ppl),
                    textcoords="offset points", xytext=(8, -8), fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    ax.axhline(summary["frozen"]["val_ppl"]["mean"], color="#5f6368", linestyle="--", label="Frozen W_ref", alpha=0.7)
    ax.axhline(summary["swa"]["val_ppl"]["mean"], color="#9c27b0", linestyle=":", label="Matched SWA", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Execution Time (ms, log scale)")
    ax.set_ylabel("Validation Perplexity at +640 Steps (lower is better)")
    ax.set_title("Compute vs Validation Quality Tradeoff on SmolGPT")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "smolgpt_compute_quality_tradeoff.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig6_path, "smolgpt_compute_quality_tradeoff.png")
    print("Generated smolgpt_compute_quality_tradeoff.png")


if __name__ == "__main__":
    generate_smolgpt_figures()
