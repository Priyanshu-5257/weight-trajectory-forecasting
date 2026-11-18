"""
Publication Figure Generator for SmolGPT Weight Norm Regularization Benchmark (Choice B).
Generates:
1. smolgpt_norm_linear_stabilization.png (Perplexity vs Horizon: Unprojected vs Global vs Tensor vs Oracle)
2. smolgpt_norm_timesfm_impact.png (Perplexity vs Horizon for TimesFM under norm matching)
3. smolgpt_norm_weight_norm_evolution.png (Weight vector norm ||W|| evolution across horizons)
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

ARTIFACT_DIR = "/home/prinshu/.gemini/antigravity-cli/brain/89fca131-5bb0-4d9a-a7fe-bd7e0105d3a5"

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


def generate_smolgpt_norm_figures():
    norm_summary_path = os.path.join(RESULTS_DIR, "smolgpt_norm_benchmark_summary.json")

    if not os.path.exists(norm_summary_path):
        print(f"Results file not found at {norm_summary_path}. Run benchmark first.")
        return

    with open(norm_summary_path) as f:
        summary = json.load(f)

    h_keys = list(summary["horizons"].keys())
    delta_steps = [int(hk.replace("h_", "")) * 20 for hk in h_keys]

    frozen_val = summary["frozen"]["val_ppl"]["mean"]
    swa_val = summary["swa"]["val_ppl"]["mean"]
    ref_norm = summary["frozen"]["weight_norm"]["mean"]

    # --------------------------------------------------------------------------
    # Figure 1: Linear Extrapolation Stabilization via Norm Matching
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Rank 1 Linear variants
    r1_methods = [
        ("linear_unprojected_r1", "Unprojected Linear (r=1)", "#f44336", "x--", 2.0),
        ("linear_global_norm_r1", "Global Norm Matched (r=1)", "#ff9800", "s-", 2.2),
        ("linear_tensor_norm_r1", "Tensor-Wise Norm Matched (r=1)", "#9c27b0", "^-", 2.2),
        ("linear_oracle_norm_r1", "Oracle Radial Norm Matched (r=1)", "#009688", "d-", 2.2),
    ]

    for m_key, m_label, col, lstyle, lwidth in r1_methods:
        vals = [summary["horizons"][hk]["methods"][m_key]["val_ppl"]["mean"] for hk in h_keys]
        # Clip unprojected for display
        vals_clipped = [min(v, 60.0) for v in vals]
        ax1.plot(delta_steps, vals_clipped, lstyle, color=col, label=m_label, linewidth=lwidth)

    ax1.axhline(frozen_val, color="#5f6368", linestyle="--", label=f"Frozen Ref (PPL = {frozen_val:.2f})")
    ax1.axhline(swa_val, color="#673ab7", linestyle=":", label=f"Matched SWA (PPL = {swa_val:.2f})")
    ax1.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps)")
    ax1.set_ylabel("Validation Perplexity (40,960 tokens)")
    ax1.set_title("Rank 1: Linear Extrapolation under Norm Projections", fontweight="bold")
    ax1.set_xticks(delta_steps)
    ax1.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
    ax1.set_ylim(10.0, 50.0)
    ax1.legend(loc="upper left")

    # Rank 2 Linear variants
    r2_methods = [
        ("linear_unprojected_r2", "Unprojected Linear (r=2)", "#f44336", "x--", 2.0),
        ("linear_global_norm_r2", "Global Norm Matched (r=2)", "#ff9800", "s-", 2.2),
        ("linear_tensor_norm_r2", "Tensor-Wise Norm Matched (r=2)", "#9c27b0", "^-", 2.2),
        ("linear_oracle_norm_r2", "Oracle Radial Norm Matched (r=2)", "#009688", "d-", 2.2),
    ]

    for m_key, m_label, col, lstyle, lwidth in r2_methods:
        vals = [summary["horizons"][hk]["methods"][m_key]["val_ppl"]["mean"] for hk in h_keys]
        vals_clipped = [min(v, 60.0) for v in vals]
        ax2.plot(delta_steps, vals_clipped, lstyle, color=col, label=m_label, linewidth=lwidth)

    ax2.axhline(frozen_val, color="#5f6368", linestyle="--", label=f"Frozen Ref (PPL = {frozen_val:.2f})")
    ax2.axhline(swa_val, color="#673ab7", linestyle=":", label=f"Matched SWA (PPL = {swa_val:.2f})")
    ax2.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps)")
    ax2.set_ylabel("Validation Perplexity (40,960 tokens)")
    ax2.set_title("Rank 2: Linear Extrapolation under Norm Projections", fontweight="bold")
    ax2.set_xticks(delta_steps)
    ax2.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
    ax2.set_ylim(10.0, 50.0)
    ax2.legend(loc="upper left")

    fig.suptitle("Choice B: Does Weight Norm Regularization Cure Linear Blowout on Full Transformer?", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "smolgpt_norm_linear_stabilization.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig1_path, "smolgpt_norm_linear_stabilization.png")

    # --------------------------------------------------------------------------
    # Figure 2: TimesFM Performance under Norm Matching
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    tfm_methods = [
        ("full_continued_adamw", "Full Continued AdamW", "#1a73e8", "o-", 2.6),
        ("timesfm_unprojected_r1", "TimesFM Unprojected (r=1)", "#e91e63", "s-", 2.2),
        ("timesfm_global_norm_r1", "TimesFM Global Norm Matched (r=1)", "#d81b60", "^-", 2.2),
        ("timesfm_tensor_norm_r1", "TimesFM Tensor-Wise Norm Matched (r=1)", "#ad1457", "v-", 2.2),
        ("timesfm_oracle_norm_r1", "TimesFM Oracle Radial Norm Matched (r=1)", "#009688", "d-", 2.2),
        ("matched_swa", "Matched SWA (Last 16)", "#9c27b0", ":", 2.2),
    ]

    for m_key, m_label, col, lstyle, lwidth in tfm_methods:
        if m_key == "matched_swa":
            ax.axhline(swa_val, color=col, linestyle=":", label=m_label, linewidth=lwidth)
            continue
        vals = [summary["horizons"][hk]["methods"][m_key]["val_ppl"]["mean"] for hk in h_keys]
        ax.plot(delta_steps, vals, lstyle, color=col, label=m_label, linewidth=lwidth)

    ax.axhline(frozen_val, color="#5f6368", linestyle="--", label=f"Frozen Ref (PPL = {frozen_val:.2f})")
    ax.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps into Future)")
    ax.set_ylabel("Validation Perplexity (40,960 tokens)")
    ax.set_title("SmolGPT Choice B: TimesFM Trajectory Forecasting under Norm Projections", fontsize=13, fontweight="bold")
    ax.set_xticks(delta_steps)
    ax.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
    ax.set_ylim(10.0, 25.0)
    ax.legend(loc="upper left")
    plt.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "smolgpt_norm_timesfm_impact.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig2_path, "smolgpt_norm_timesfm_impact.png")

    # --------------------------------------------------------------------------
    # Figure 3: Weight Vector Norm Evolution Across Horizons
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))

    act_norms = [summary["horizons"][hk]["methods"]["full_continued_adamw"]["weight_norm"]["mean"] for hk in h_keys]
    lin_norms = [summary["horizons"][hk]["methods"]["linear_unprojected_r1"]["weight_norm"]["mean"] for hk in h_keys]
    tfm_norms = [summary["horizons"][hk]["methods"]["timesfm_unprojected_r1"]["weight_norm"]["mean"] for hk in h_keys]
    glob_norms = [summary["horizons"][hk]["methods"]["linear_global_norm_r1"]["weight_norm"]["mean"] for hk in h_keys]

    ax.plot(delta_steps, lin_norms, "x--", color="#f44336", label=r"Unprojected Linear (r=1) $\|W\|$", linewidth=2.4)
    ax.plot(delta_steps, tfm_norms, "s-", color="#e91e63", label=r"Unprojected TimesFM (r=1) $\|W\|$", linewidth=2.2)
    ax.plot(delta_steps, act_norms, "o-", color="#1a73e8", label=r"Actual AdamW $\|W\|$ (Natural Decay)", linewidth=2.6)
    ax.axhline(ref_norm, color="#5f6368", linestyle="--", label=rf"Frozen Reference $\|W_{{\text{{ref}}}}\|$ ({ref_norm:.2f})")
    ax.plot(delta_steps, glob_norms, ":", color="#ff9800", label=r"Global Norm Matched ($\|W\| = \|W_{\text{ref}}\|$)", linewidth=2.0)

    ax.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps)")
    ax.set_ylabel(r"Weight Vector $L_2$ Norm $\|W\|_2$")
    ax.set_title("Weight Norm Drift: Radial Expansion vs. Natural Optimization Decay", fontsize=13, fontweight="bold")
    ax.set_xticks(delta_steps)
    ax.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
    ax.legend(loc="center right")
    plt.tight_layout()

    fig3_path = os.path.join(FIGURES_DIR, "smolgpt_norm_weight_norm_evolution.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig3_path, "smolgpt_norm_weight_norm_evolution.png")

    print("All Choice B norm regularization figures successfully generated!")


if __name__ == "__main__":
    generate_smolgpt_norm_figures()
