"""
Publication Figure Generator for SmolGPT MLP-Only Weight Trajectory Forecasting Benchmark (Choice A Option 1).
Generates:
1. smolgpt_mlp_ppl_vs_horizon.png (Perplexity vs Horizon across methods with 95% CIs)
2. smolgpt_mlp_rank_comparison.png (Perplexity vs Truncation Rank r in [1, 2, 4, 8, 16])
3. smolgpt_mlp_train_val_gap.png (Train & Val Loss across horizons)
4. smolgpt_mlp_vs_full_comparison.png (Comparison of Full Model vs MLP-Only Forecasting)
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


def generate_smolgpt_mlp_figures():
    mlp_summary_path = os.path.join(RESULTS_DIR, "smolgpt_mlp_only_summary.json")
    full_summary_path = os.path.join(RESULTS_DIR, "smolgpt_benchmark_summary.json")

    if not os.path.exists(mlp_summary_path):
        print(f"Results file not found at {mlp_summary_path}. Run benchmark first.")
        return

    with open(mlp_summary_path) as f:
        summary = json.load(f)

    full_summary = None
    if os.path.exists(full_summary_path):
        with open(full_summary_path) as f:
            full_summary = json.load(f)

    h_keys = list(summary["horizons"].keys())
    delta_steps = [int(hk.replace("h_", "")) * 20 for hk in h_keys]
    ranks = summary["ranks"]

    # --------------------------------------------------------------------------
    # Figure 1: Perplexity vs Horizon (Choice A Option 1)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    frozen_val = summary["frozen"]["val_ppl"]["mean"]
    swa_val = summary["swa_mlp"]["val_ppl"]["mean"]

    ax.axhline(frozen_val, color="#5f6368", linestyle="--", linewidth=1.8, label=f"Frozen Ref at Step 1000 (PPL = {frozen_val:.2f})")
    ax.axhline(swa_val, color="#9c27b0", linestyle=":", linewidth=2.2, label=f"MLP Matched SWA (PPL = {swa_val:.2f})")

    methods_to_plot = [
        ("full_continued_adamw", "Full Continued AdamW (All Weights)", "#1a73e8", "o-", 2.6),
        ("mlp_only_actual_continuation", "MLP-Only Actual AdamW (Attn/Embed Frozen)", "#009688", "s--", 2.4),
        ("mlp_timesfm_r1", "MLP TimesFM (Rank 1)", "#e91e63", "^-", 2.2),
        ("mlp_timesfm_r2", "MLP TimesFM (Rank 2)", "#d81b60", "v-", 2.2),
        ("mlp_damped_r2", "MLP Damped (Rank 2)", "#ff9800", "d-", 2.0),
        ("mlp_linear_r2", "MLP Linear (Rank 2)", "#f44336", "x--", 2.0),
    ]

    for m_key, m_label, col, lstyle, lwidth in methods_to_plot:
        vals = [summary["horizons"][hk]["methods"][m_key]["val_ppl"]["mean"] for hk in h_keys]
        cis = [summary["horizons"][hk]["methods"][m_key]["val_ppl"]["ci95"] for hk in h_keys]
        # clip display for linear if high
        vals_clipped = [min(v, 60.0) for v in vals]
        ax.plot(delta_steps, vals_clipped, lstyle, color=col, label=m_label, linewidth=lwidth)
        if m_key in ["full_continued_adamw", "mlp_only_actual_continuation", "mlp_timesfm_r1", "mlp_timesfm_r2"]:
            ax.fill_between(delta_steps, np.array(vals) - np.array(cis), np.array(vals) + np.array(cis), color=col, alpha=0.12)

    ax.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps into Future)")
    ax.set_ylabel("Validation Perplexity (40,960 tokens)")
    ax.set_title("SmolGPT Choice A Option 1: Forecasting Isolated strictly to MLP Linear Blocks\n(Token Embeddings & Attention strictly Frozen at Step 1000)", fontsize=13, fontweight="bold")
    ax.set_xticks(delta_steps)
    ax.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
    ax.set_ylim(10.0, 35.0)
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "smolgpt_mlp_ppl_vs_horizon.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig1_path, "smolgpt_mlp_ppl_vs_horizon.png")

    # --------------------------------------------------------------------------
    # Figure 2: Perplexity vs Rank in MLP Subspace (at Horizon h=8 and h=16)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_curr, h_val, h_title in [(ax1, 8, "Short Horizon h=8 (+160 steps)"), (ax2, 16, "Medium Horizon h=16 (+320 steps)")]:
        hk = f"h_{h_val}"
        tfm_ppls = [summary["horizons"][hk]["methods"][f"mlp_timesfm_r{r}"]["val_ppl"]["mean"] for r in ranks]
        lin_ppls = [min(summary["horizons"][hk]["methods"][f"mlp_linear_r{r}"]["val_ppl"]["mean"], 50.0) for r in ranks]
        damp_ppls = [summary["horizons"][hk]["methods"][f"mlp_damped_r{r}"]["val_ppl"]["mean"] for r in ranks]
        shuf_ppls = [summary["horizons"][hk]["methods"][f"mlp_shuffled_timesfm_r{r}"]["val_ppl"]["mean"] for r in ranks]

        ax_curr.plot(ranks, tfm_ppls, "o-", color="#e91e63", label="MLP TimesFM-2.0", linewidth=2.4)
        ax_curr.plot(ranks, damp_ppls, "^-", color="#ff9800", label="MLP Damped (γ=0.95)", linewidth=2.0)
        ax_curr.plot(ranks, shuf_ppls, "x:", color="#795548", label="MLP Shuffled TimesFM", linewidth=1.8)
        ax_curr.plot(ranks, lin_ppls, "d--", color="#f44336", label="MLP Linear (clipped)", linewidth=2.0)

        ax_curr.axhline(frozen_val, color="#5f6368", linestyle="--", label="Frozen Ref (Step 1000)", alpha=0.8)
        act_mlp = summary["horizons"][hk]["methods"]["mlp_only_actual_continuation"]["val_ppl"]["mean"]
        ax_curr.axhline(act_mlp, color="#009688", linestyle="-.", label=f"MLP Actual Continuation ({act_mlp:.2f})", alpha=0.8)

        ax_curr.set_xlabel("SVD Truncation Rank $r$")
        ax_curr.set_title(h_title, fontweight="bold")
        ax_curr.set_xticks(ranks)
        ax_curr.set_xticklabels([str(r) for r in ranks])
        ax_curr.legend(loc="upper right", framealpha=0.9)

    ax1.set_ylabel("Validation Perplexity")
    fig.suptitle("SmolGPT MLP Subspace: Perplexity vs Truncation Rank $r$", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "smolgpt_mlp_rank_comparison.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig2_path, "smolgpt_mlp_rank_comparison.png")

    # --------------------------------------------------------------------------
    # Figure 3: Train Loss vs Val Loss across Horizons (MLP-Only)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    plot_keys = [
        ("full_continued_adamw", "Full Continued AdamW", "#1a73e8", "o-"),
        ("mlp_only_actual_continuation", "MLP Actual AdamW", "#009688", "s--"),
        ("mlp_timesfm_r1", "MLP TimesFM (Rank 1)", "#e91e63", "^-"),
        ("mlp_damped_r2", "MLP Damped (Rank 2)", "#ff9800", "d-"),
        ("mlp_linear_r2", "MLP Linear (Rank 2)", "#f44336", "x:"),
    ]

    for m_key, m_label, col, lstyle in plot_keys:
        t_vals = [summary["horizons"][hk]["methods"][m_key]["train_loss"]["mean"] for hk in h_keys]
        v_vals = [summary["horizons"][hk]["methods"][m_key]["val_loss"]["mean"] for hk in h_keys]
        ax1.plot(delta_steps, [min(v, 4.0) for v in t_vals], lstyle, color=col, label=m_label)
        ax2.plot(delta_steps, [min(v, 4.0) for v in v_vals], lstyle, color=col, label=m_label)

    ax1.axhline(summary["frozen"]["train_loss"]["mean"], color="#5f6368", linestyle="--", label="Frozen Ref")
    ax2.axhline(summary["frozen"]["val_loss"]["mean"], color="#5f6368", linestyle="--", label="Frozen Ref")

    ax1.set_title("MLP-Only: Train Cross-Entropy Loss", fontweight="bold")
    ax2.set_title("MLP-Only: Validation Cross-Entropy Loss", fontweight="bold")
    ax1.set_xlabel(r"Forecast Horizon $\Delta t$ (Steps)")
    ax2.set_xlabel(r"Forecast Horizon $\Delta t$ (Steps)")
    ax1.set_ylabel("Train Loss (40,960 tokens)")
    ax2.set_ylabel("Validation Loss (40,960 tokens)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.tight_layout()

    fig3_path = os.path.join(FIGURES_DIR, "smolgpt_mlp_train_val_gap.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    copy_to_artifacts(fig3_path, "smolgpt_mlp_train_val_gap.png")

    # --------------------------------------------------------------------------
    # Figure 4: Full Model vs MLP-Only Comparison
    # --------------------------------------------------------------------------
    if full_summary is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Full TimesFM vs MLP TimesFM
        full_tfm_r2 = [full_summary["horizons"][hk]["methods"]["low_rank_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]
        mlp_tfm_r2 = [summary["horizons"][hk]["methods"]["mlp_timesfm_r2"]["val_ppl"]["mean"] for hk in h_keys]
        mlp_tfm_r1 = [summary["horizons"][hk]["methods"]["mlp_timesfm_r1"]["val_ppl"]["mean"] for hk in h_keys]

        # Full Linear vs MLP Linear
        full_lin_r2 = [min(full_summary["horizons"][hk]["methods"]["low_rank_linear_r2"]["val_ppl"]["mean"], 50.0) for hk in h_keys]
        mlp_lin_r2 = [min(summary["horizons"][hk]["methods"]["mlp_linear_r2"]["val_ppl"]["mean"], 50.0) for hk in h_keys]

        ax.plot(delta_steps, full_tfm_r2, "o--", color="#e91e63", alpha=0.6, label="Full Model TimesFM (Rank 2)")
        ax.plot(delta_steps, mlp_tfm_r2, "o-", color="#c2185b", linewidth=2.5, label="MLP-Only TimesFM (Rank 2)")
        ax.plot(delta_steps, mlp_tfm_r1, "^-", color="#880e4f", linewidth=2.5, label="MLP-Only TimesFM (Rank 1)")

        ax.plot(delta_steps, full_lin_r2, "x--", color="#f44336", alpha=0.6, label="Full Model Linear (Rank 2, blown up)")
        ax.plot(delta_steps, mlp_lin_r2, "x-", color="#b71c1c", linewidth=2.0, label="MLP-Only Linear (Rank 2)")

        ax.axhline(frozen_val, color="#5f6368", linestyle="--", label="Frozen Ref (Step 1000)")
        ax.axhline(summary["horizons"][h_keys[-1]]["methods"]["full_continued_adamw"]["val_ppl"]["mean"], color="#1a73e8", linestyle=":", label="Full Continued AdamW (Target)")

        ax.set_xlabel(r"Forecast Horizon $\Delta t$ (Training Steps)")
        ax.set_ylabel("Validation Perplexity (40,960 tokens)")
        ax.set_title("Full Model Forecasting vs MLP-Isolated Forecasting (Choice A Option 1)", fontsize=13, fontweight="bold")
        ax.set_xticks(delta_steps)
        ax.set_xticklabels([f"+{s}\n(h={s//20})" for s in delta_steps])
        ax.set_ylim(10.0, 35.0)
        ax.legend(loc="upper left")
        plt.tight_layout()

        fig4_path = os.path.join(FIGURES_DIR, "smolgpt_mlp_vs_full_comparison.png")
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        copy_to_artifacts(fig4_path, "smolgpt_mlp_vs_full_comparison.png")

    print("All MLP figures successfully generated!")


if __name__ == "__main__":
    generate_smolgpt_mlp_figures()
