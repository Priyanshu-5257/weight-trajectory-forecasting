import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
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

    # --------------------------------------------------------------------------
    # Figure 1: Subspace Explained Variance & Trajectory Collapse
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = summary["ranks"]
    exp_vars = [summary["explained_variance"][str(r)]["mean"] * 100 for r in ranks]
    exp_ci = [summary["explained_variance"][str(r)]["ci95"] * 100 for r in ranks]

    bars = ax.bar([f"Rank {r}" for r in ranks], exp_vars, yerr=exp_ci, capsize=5, color="#1a73e8", alpha=0.85, edgecolor="#1557b0", width=0.5)
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Grokking Subspace Dimensionality (SVD of 32 Checkpoint Displacements)")
    ax.set_ylim(80, 102)

    for bar, val in zip(bars, exp_vars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', color="#1557b0")

    ax.axhline(100, color="gray", linestyle="--", alpha=0.6)
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "grokking_subspace_explained_variance.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[Saved] {f1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Test Accuracy vs Horizon Across Methods (Rank 2)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    horizons = summary["horizons"]
    delta_steps = [h * summary["checkpoint_interval"] for h in horizons]

    cont_acc = [summary["baselines"]["continued_adam"][f"h_{h}"]["acc"]["mean"] for h in horizons]
    cont_ci = [summary["baselines"]["continued_adam"][f"h_{h}"]["acc"]["ci95"] for h in horizons]

    lin_acc = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["acc"]["mean"] for h in horizons]
    lin_ci = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["acc"]["ci95"] for h in horizons]

    dmp_acc = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["acc"]["mean"] for h in horizons]
    dmp_ci = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["acc"]["ci95"] for h in horizons]

    tfm_acc = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["acc"]["mean"] for h in horizons]
    tfm_ci = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["acc"]["ci95"] for h in horizons]

    frozen_val = summary["baselines"]["frozen_acc"]["mean"]
    swa_val = summary["baselines"]["swa_acc"]["mean"]

    ax.axhline(frozen_val, color="#5f6368", linestyle="--", label=f"Frozen Checkpoint ({frozen_val:.1f}%)")
    ax.axhline(swa_val, color="#ea4335", linestyle=":", label=f"Matched SWA ({swa_val:.1f}%)")

    ax.errorbar(delta_steps, cont_acc, yerr=cont_ci, label="Continued AdamW (Ground Truth)", color="#34a853", marker="o", capsize=4)
    ax.errorbar(delta_steps, lin_acc, yerr=lin_ci, label="Low-Rank Linear (Rank 2)", color="#ff6d01", marker="s", capsize=4)
    ax.errorbar(delta_steps, dmp_acc, yerr=dmp_ci, label="Low-Rank Damped (Rank 2, γ=0.95)", color="#9334e6", marker="^", capsize=4)
    ax.errorbar(delta_steps, tfm_acc, yerr=tfm_ci, label="Low-Rank TimesFM (Rank 2)", color="#1a73e8", marker="D", capsize=4)

    ax.set_xlabel("Forecast Horizon (+Steps beyond Cutoff)")
    ax.set_ylabel("Test Accuracy on Modular Addition (%)")
    ax.set_title("Grokking Trajectory Forecasting vs Continued Optimization")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "grokking_accuracy_vs_horizon.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"[Saved] {f2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Continuation Cosine Alignment vs Horizon
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lin_cos = [summary["methods"]["linear"]["rank_2"][f"horizon_{h}"]["cos_sim_to_continuation"]["mean"] for h in horizons]
    dmp_cos = [summary["methods"]["damped"]["rank_2"][f"horizon_{h}"]["cos_sim_to_continuation"]["mean"] for h in horizons]
    tfm_cos = [summary["methods"]["timesfm"]["rank_2"][f"horizon_{h}"]["cos_sim_to_continuation"]["mean"] for h in horizons]

    ax.plot(delta_steps, lin_cos, label="Low-Rank Linear (Rank 2)", color="#ff6d01", marker="s")
    ax.plot(delta_steps, dmp_cos, label="Low-Rank Damped (Rank 2)", color="#9334e6", marker="^")
    ax.plot(delta_steps, tfm_cos, label="Low-Rank TimesFM (Rank 2)", color="#1a73e8", marker="D")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(1.0, color="#34a853", linestyle=":", alpha=0.8, label="Perfect Alignment (+1.0)")
    ax.set_xlabel("Forecast Horizon (+Steps beyond Cutoff)")
    ax.set_ylabel("Cosine Similarity to Actual Optimization Trajectory")
    ax.set_title("Mechanistic Directional Alignment with Continued AdamW")
    ax.set_ylim(-1.1, 1.15)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "grokking_cosine_alignment.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"[Saved] {f3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Performance Across Low-Rank Subspace (Ranks 1, 2, 4, 8)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    h_target = 16 # +400 steps
    lin_rank_acc = [summary["methods"]["linear"][f"rank_{r}"][f"horizon_{h_target}"]["acc"]["mean"] for r in ranks]
    dmp_rank_acc = [summary["methods"]["damped"][f"rank_{r}"][f"horizon_{h_target}"]["acc"]["mean"] for r in ranks]
    tfm_rank_acc = [summary["methods"]["timesfm"][f"rank_{r}"][f"horizon_{h_target}"]["acc"]["mean"] for r in ranks]

    x = np.arange(len(ranks))
    width = 0.25

    ax.bar(x - width, lin_rank_acc, width, label="Linear Extrapolation", color="#ff6d01", alpha=0.9)
    ax.bar(x, dmp_rank_acc, width, label="Damped Extrapolation", color="#9334e6", alpha=0.9)
    ax.bar(x + width, tfm_rank_acc, width, label="TimesFM", color="#1a73e8", alpha=0.9)

    ax.axhline(frozen_val, color="#5f6368", linestyle="--", label=f"Frozen Checkpoint ({frozen_val:.1f}%)")
    ax.axhline(swa_val, color="#ea4335", linestyle=":", label=f"Matched SWA ({swa_val:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks])
    ax.set_ylabel("Test Accuracy (%) at Horizon +400 Steps")
    ax.set_title("Low-Rank Subspace Performance Comparison (+400 Steps)")
    ax.set_ylim(45, 65)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "grokking_rank_comparison.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"[Saved] {f4_path}")

if __name__ == "__main__":
    generate_grokking_figures()
