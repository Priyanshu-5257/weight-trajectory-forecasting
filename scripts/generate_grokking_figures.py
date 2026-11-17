import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# High-quality academic styling
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
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.dpi": 300
})

def generate_grokking_figures():
    raw_path = os.path.join(RESULTS_DIR, "grokking_benchmark_results.json")
    summary_path = os.path.join(RESULTS_DIR, "grokking_benchmark_summary.json")

    if not os.path.exists(raw_path) or not os.path.exists(summary_path):
        print("Results files not found. Please run the benchmark first.")
        return

    with open(raw_path) as f:
        raw_data = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    ranks = summary["ranks"]
    horizons = summary["horizons"]
    checkpoint_interval = summary["checkpoint_interval"]
    delta_steps = [h * checkpoint_interval for h in horizons]

    # --------------------------------------------------------------------------
    # Figure 1: Non-Symmetric Generalization vs Horizon (Pre-Grokking Cutoff = 1500)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Pure Non-Symmetric Generalization
    cont_acc_ns = [summary["baselines"]["continued_adam"][f"h_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    cont_ci_ns = [summary["baselines"]["continued_adam"][f"h_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    lin_acc_ns = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    lin_ci_ns = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    dmp_acc_ns = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    dmp_ci_ns = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    tfm_acc_ns = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    tfm_ci_ns = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    dec_acc_ns = [summary["baselines"]["weight_decay_only"][f"h_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    dec_ci_ns = [summary["baselines"]["weight_decay_only"][f"h_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    frozen_ns = summary["baselines"]["frozen_acc_nonsym"]["mean"]
    swa_ns = summary["baselines"]["swa_acc_nonsym"]["mean"]

    ax1.axhline(frozen_ns, color="#5f6368", linestyle="--", label=f"Frozen Checkpoint @ 1500 ({frozen_ns:.2f}%)")
    ax1.axhline(swa_ns, color="#9c27b0", linestyle=":", label=f"Matched SWA ({swa_ns:.2f}%)")

    ax1.errorbar(delta_steps, cont_acc_ns, yerr=cont_ci_ns, label="Continued AdamW (Ground Truth)", color="#34a853", marker="o", capsize=3)
    ax1.errorbar(delta_steps, lin_acc_ns, yerr=lin_ci_ns, label="Low-Rank Linear (Rank 2)", color="#ff6d01", marker="s", capsize=3)
    ax1.errorbar(delta_steps, dmp_acc_ns, yerr=dmp_ci_ns, label="Low-Rank Damped (Rank 2, γ=0.95)", color="#e91e63", marker="^", capsize=3)
    ax1.errorbar(delta_steps, tfm_acc_ns, yerr=tfm_ci_ns, label="Low-Rank TimesFM (Rank 2, Fixed Index)", color="#1a73e8", marker="D", capsize=3)
    ax1.errorbar(delta_steps, dec_acc_ns, yerr=dec_ci_ns, label="Weight Decay Only Control", color="#795548", linestyle="--", marker="x", capsize=3)

    ax1.set_xlabel("Forecast Horizon (+Steps beyond Cutoff 1500)")
    ax1.set_ylabel("Non-Symmetric Test Accuracy (%) [Pure Generalization]")
    ax1.set_title("Strict Generalization (Excluding Commutative Pairs)")
    ax1.set_ylim(-2, 102)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: All Test Pairs vs Non-Symmetric Pairs (Illustrating the Memorization Artifact)
    cont_acc_all = [summary["baselines"]["continued_adam"][f"h_{h}"]["acc_all"]["mean"] for h in horizons]
    tfm_acc_all = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["acc_all"]["mean"] for h in horizons]
    frozen_all = summary["baselines"]["frozen_acc_all"]["mean"]

    ax2.plot(delta_steps, cont_acc_all, color="#34a853", marker="o", label="Continued AdamW (All Test Pairs)")
    ax2.plot(delta_steps, cont_acc_ns, color="#34a853", linestyle="--", marker="o", alpha=0.6, label="Continued AdamW (Non-Symmetric Only)")
    ax2.plot(delta_steps, tfm_acc_all, color="#1a73e8", marker="D", label="TimesFM Rank 2 (All Test Pairs)")
    ax2.plot(delta_steps, tfm_acc_ns, color="#1a73e8", linestyle="--", marker="D", alpha=0.6, label="TimesFM Rank 2 (Non-Symmetric Only)")
    ax2.axhline(50.0, color="#d32f2f", linestyle=":", label="Commutative Memorization Baseline (~50%)")
    ax2.axhline(frozen_all, color="#5f6368", linestyle="--", label=f"Frozen All Pairs ({frozen_all:.1f}%)")

    ax2.set_xlabel("Forecast Horizon (+Steps beyond Cutoff 1500)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Commutativity Artifact: All Pairs vs Non-Symmetric Pairs")
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "grokking_accuracy_vs_horizon.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[Saved] {f1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Subspace Dimensionality & Radial Shrinkage Variance
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: SVD Explained Variance
    exp_vars = [summary["explained_variance"][str(r)]["mean"] * 100 for r in ranks]
    exp_ci = [summary["explained_variance"][str(r)]["ci95"] * 100 for r in ranks]

    bars = ax1.bar([f"Rank {r}" for r in ranks], exp_vars, yerr=exp_ci, capsize=5, color="#1a73e8", alpha=0.85, edgecolor="#1557b0", width=0.45)
    ax1.set_ylabel("Cumulative Explained Variance (%)")
    ax1.set_title("Pre-Grokking Checkpoint SVD Explained Variance (Cutoff=1500)")
    ax1.set_ylim(85, 102)

    for bar, val in zip(bars, exp_vars):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{val:.2f}%", ha='center', va='bottom', fontweight='bold', color="#1557b0")
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.6)

    # Panel 2: Radial Alignment of Predictions (Projection along -W_ref)
    lin_rad = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["radial_alignment_pred"]["mean"] for h in horizons]
    tfm_rad = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["radial_alignment_pred"]["mean"] for h in horizons]
    dmp_rad = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["radial_alignment_pred"]["mean"] for h in horizons]

    ax2.plot(delta_steps, lin_rad, label="Linear Extrapolation (Rank 2)", color="#ff6d01", marker="s")
    ax2.plot(delta_steps, dmp_rad, label="Damped Extrapolation (Rank 2)", color="#e91e63", marker="^")
    ax2.plot(delta_steps, tfm_rad, label="TimesFM (Rank 2)", color="#1a73e8", marker="D")
    ax2.axhline(1.0, color="#795548", linestyle="--", label="Pure Radial Shrinkage (-W_ref)")

    ax2.set_xlabel("Forecast Horizon (+Steps beyond Cutoff 1500)")
    ax2.set_ylabel("Radial Alignment Ratio cos(ΔW, -W_ref)")
    ax2.set_title("Pre-Grokking Trajectory Direction: Radial Weight Decay")
    ax2.set_ylim(-0.2, 1.1)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "grokking_subspace_explained_variance.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"[Saved] {f2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Full-Parameter Space Alignment in R^P (P = 227,200)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    lin_cos_full = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["cos_sim_full"]["mean"] for h in horizons]
    dmp_cos_full = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["cos_sim_full"]["mean"] for h in horizons]
    tfm_cos_full = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["cos_sim_full"]["mean"] for h in horizons]

    lin_cos_lat = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["cos_sim_latent"]["mean"] for h in horizons]
    tfm_cos_lat = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["cos_sim_latent"]["mean"] for h in horizons]

    ax1.plot(delta_steps, lin_cos_full, label="Linear (Full R^P)", color="#ff6d01", marker="s")
    ax1.plot(delta_steps, dmp_cos_full, label="Damped (Full R^P)", color="#e91e63", marker="^")
    ax1.plot(delta_steps, tfm_cos_full, label="TimesFM (Full R^P)", color="#1a73e8", marker="D")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Forecast Horizon (+Steps beyond Cutoff)")
    ax1.set_ylabel("Full Cosine Similarity cos_sim(ΔW_pred, ΔW_actual)")
    ax1.set_title("Full-Parameter Alignment in R^227,200")
    ax1.set_ylim(-0.2, 1.05)
    ax1.legend(loc="lower left", frameon=True)

    # Latent vs Full parameter comparison
    ax2.plot(delta_steps, tfm_cos_lat, label="TimesFM Latent Alignment (in R^r)", color="#1a73e8", linestyle="--", marker="o")
    ax2.plot(delta_steps, tfm_cos_full, label="TimesFM Parameter Alignment (in R^P)", color="#1a73e8", marker="D")
    ax2.plot(delta_steps, lin_cos_lat, label="Linear Latent Alignment (in R^r)", color="#ff6d01", linestyle="--", marker="o")
    ax2.plot(delta_steps, lin_cos_full, label="Linear Parameter Alignment (in R^P)", color="#ff6d01", marker="s")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Forecast Horizon (+Steps beyond Cutoff)")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Latent Subspace vs Full Parameter Alignment")
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "grokking_cosine_alignment.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"[Saved] {f3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Fourier Circuit Saliency Emergence
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fourier_saliency = [summary["baselines"]["continued_adam"][f"h_{h}"]["fourier_saliency"]["mean"] * 100 for h in horizons]
    fourier_ci = [summary["baselines"]["continued_adam"][f"h_{h}"]["fourier_saliency"]["ci95"] * 100 for h in horizons]

    ax.errorbar(delta_steps, fourier_saliency, yerr=fourier_ci, label="Continued AdamW (Embeddings Fourier Saliency)", color="#673ab7", marker="o", capsize=4)
    ax.axhline(summary["baselines"]["continued_adam"]["h_8"]["fourier_saliency"]["mean"] * 100, color="gray", linestyle=":", label="Initial Pre-Grokking Fourier Power")
    ax.set_xlabel("Steps beyond Pre-Grokking Cutoff 1500")
    ax.set_ylabel("Top-5 2D-DFT Power Ratio (%) on W_E[:p]")
    ax.set_title("Fourier Modular Circuit Formation Across Grokking Transition")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "grokking_fourier_circuit_saliency.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"[Saved] {f4_path}")

if __name__ == "__main__":
    generate_grokking_figures()
