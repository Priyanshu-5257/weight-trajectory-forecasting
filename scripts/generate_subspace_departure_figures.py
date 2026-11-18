import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

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

def generate_subspace_departure_figures():
    raw_path = os.path.join(RESULTS_DIR, "subspace_departure_results.json")
    summary_path = os.path.join(RESULTS_DIR, "subspace_departure_summary.json")

    if not os.path.exists(raw_path) or not os.path.exists(summary_path):
        print("Results files not found. Please run the benchmark first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    ranks = summary["ranks"]
    horizons = summary["horizons"]
    checkpoint_interval = summary["checkpoint_interval"]
    delta_steps = [h * checkpoint_interval for h in horizons]

    # --------------------------------------------------------------------------
    # Figure 1: Full-Model Predictive Coverage Collapse vs Horizon
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {1: "#9e9e9e", 2: "#ff9800", 4: "#4caf50", 8: "#1a73e8", 16: "#9c27b0", 30: "#009688", 31: "#3f51b5", 32: "#e91e63"}
    plot_ranks = [r for r in ranks if r in colors]

    for r in plot_ranks:
        cov_means = [summary["coverage_full"][f"r_{r}"][f"h_{h}"]["mean"] * 100 for h in horizons]
        cov_ci = [summary["coverage_full"][f"r_{r}"][f"h_{h}"]["ci95"] * 100 for h in horizons]
        ax1.errorbar(delta_steps, cov_means, yerr=cov_ci, label=f"Rank {r}", color=colors[r], marker="o", capsize=3)

    ax1.set_xlabel("Steps beyond Pre-Grokking Cutoff (Step 1500)")
    ax1.set_ylabel("Predictive Subspace Coverage (%) [||ΔW_proj||² / ||ΔW_act||²]")
    ax1.set_title("Historical Subspace Loses Predictive Coverage as Grokking Emerges")
    ax1.set_ylim(20, 102)
    ax1.legend(loc="lower left", frameon=True)

    # Panel 2: Orthogonal Energy Fraction (1 - Coverage) for r=8, 32 vs Actual Generalization
    orth_8 = [(1.0 - summary["coverage_full"]["r_8"][f"h_{h}"]["mean"]) * 100 for h in horizons]
    orth_32 = [(1.0 - summary["coverage_full"]["r_32"][f"h_{h}"]["mean"]) * 100 for h in horizons]
    act_acc = [summary["actual_continuation"][f"h_{h}"]["acc_nonsym"]["mean"] for h in horizons]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(delta_steps, orth_8, color="#d32f2f", marker="s", label="Orthogonal Displacement (Rank 8 Basis)")
    l2 = ax2.plot(delta_steps, orth_32, color="#7b1fa2", marker="^", label="Orthogonal Displacement (Full Rank 32 Basis)")
    l3 = ax2_twin.plot(delta_steps, act_acc, color="#2e7d32", linestyle="--", marker="o", label="Actual Non-Symmetric Test Acc (Right Axis)")

    ax2.set_xlabel("Steps beyond Pre-Grokking Cutoff (Step 1500)")
    ax2.set_ylabel("Orthogonal Component Energy (%) [||ΔW_perp||² / ||ΔW_act||²]", color="#d32f2f")
    ax2_twin.set_ylabel("Strict Non-Symmetric Test Accuracy (%)", color="#2e7d32")
    ax2.set_title("Grokking Coincides with Departure from Historical Subspace")
    ax2.set_ylim(-2, 75)
    ax2_twin.set_ylim(-2, 102)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left", frameon=True)

    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "subspace_departure_coverage_vs_horizon.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[Saved] {f1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Layer-wise Predictive Coverage Collapse
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    layer_colors = {"embed": "#1a73e8", "attn": "#e91e63", "mlp": "#ff9800", "unembed": "#4caf50"}
    layer_names = {"embed": "Embeddings (Token & Pos)", "attn": "Self-Attention (Q, K, V, Out)", "mlp": "MLP Feedforward", "unembed": "Unembedding Head"}

    for l in ["embed", "attn", "mlp", "unembed"]:
        cov_l = [summary["coverage_layers"][l]["r_8"][f"h_{h}"]["mean"] * 100 for h in horizons]
        ci_l = [summary["coverage_layers"][l]["r_8"][f"h_{h}"]["ci95"] * 100 for h in horizons]
        ax.errorbar(delta_steps, cov_l, yerr=ci_l, label=f"{layer_names[l]} (Rank 8)", color=layer_colors[l], marker="o", capsize=3)

    ax.set_xlabel("Steps beyond Pre-Grokking Cutoff (Step 1500)")
    ax.set_ylabel("Predictive Subspace Coverage (%) at Rank 8")
    ax.set_title("Layer Breakdown: Attention & MLP Strongly Rotate Out of Historical Basis")
    ax.set_ylim(10, 102)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "subspace_departure_layer_breakdown.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"[Saved] {f2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Reconstructed Model Accuracy (Actual vs Projected vs Orthogonal)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    act_acc = [summary["actual_continuation"][f"h_{h}"]["acc_nonsym"]["mean"] for h in horizons]
    act_ci = [summary["actual_continuation"][f"h_{h}"]["acc_nonsym"]["ci95"] for h in horizons]

    proj_2 = [summary["reconstructed_models"]["r_2"]["projected"][f"h_{h}"]["mean"] for h in horizons]
    proj_2_ci = [summary["reconstructed_models"]["r_2"]["projected"][f"h_{h}"]["ci95"] for h in horizons]

    proj_8 = [summary["reconstructed_models"]["r_8"]["projected"][f"h_{h}"]["mean"] for h in horizons]
    proj_8_ci = [summary["reconstructed_models"]["r_8"]["projected"][f"h_{h}"]["ci95"] for h in horizons]

    proj_32 = [summary["reconstructed_models"]["r_32"]["projected"][f"h_{h}"]["mean"] for h in horizons]
    proj_32_ci = [summary["reconstructed_models"]["r_32"]["projected"][f"h_{h}"]["ci95"] for h in horizons]

    orth_8_acc = [summary["reconstructed_models"]["r_8"]["orthogonal"][f"h_{h}"]["mean"] for h in horizons]
    orth_8_ci = [summary["reconstructed_models"]["r_8"]["orthogonal"][f"h_{h}"]["ci95"] for h in horizons]

    # Panel 1: Accuracy comparison
    ax1.errorbar(delta_steps, act_acc, yerr=act_ci, label="Actual Continuation W(t+h)", color="#2e7d32", marker="o", capsize=3)
    ax1.errorbar(delta_steps, proj_8, yerr=proj_8_ci, label="Projected onto Historical Basis W_proj (Rank 8)", color="#d32f2f", marker="s", capsize=3)
    ax1.errorbar(delta_steps, proj_32, yerr=proj_32_ci, label="Projected onto Full Basis W_proj (Rank 32)", color="#ff9800", marker="^", capsize=3)
    ax1.errorbar(delta_steps, orth_8_acc, yerr=orth_8_ci, label="Orthogonal Continuation W_orth (Rank 8)", color="#9c27b0", marker="D", capsize=3)

    ax1.set_xlabel("Steps beyond Pre-Grokking Cutoff (Step 1500)")
    ax1.set_ylabel("Strict Non-Symmetric Test Accuracy (%)")
    ax1.set_title("Oracle Subspace Projection Fails: Generalization Requires Departure")
    ax1.set_ylim(-2, 102)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Paired difference (W_proj - W_act)
    diff_8 = [summary["reconstructed_models"]["r_8"]["paired_diff_projected_vs_actual"][f"h_{h}"]["mean"] for h in horizons]
    diff_8_ci = [summary["reconstructed_models"]["r_8"]["paired_diff_projected_vs_actual"][f"h_{h}"]["ci95"] for h in horizons]

    diff_32 = [summary["reconstructed_models"]["r_32"]["paired_diff_projected_vs_actual"][f"h_{h}"]["mean"] for h in horizons]
    diff_32_ci = [summary["reconstructed_models"]["r_32"]["paired_diff_projected_vs_actual"][f"h_{h}"]["ci95"] for h in horizons]

    ax2.errorbar(delta_steps, diff_8, yerr=diff_8_ci, label="Δ(Projected r=8 - Actual Continuation)", color="#d32f2f", marker="s", capsize=3)
    ax2.errorbar(delta_steps, diff_32, yerr=diff_32_ci, label="Δ(Projected r=32 - Actual Continuation)", color="#ff9800", marker="^", capsize=3)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)

    ax2.set_xlabel("Steps beyond Pre-Grokking Cutoff (Step 1500)")
    ax2.set_ylabel("Paired Accuracy Drop Δ (%)")
    ax2.set_title("Generalization Penalty of Subspace Projection (95% CI)")
    ax2.set_ylim(-105, 5)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "subspace_departure_reconstructed_accuracy.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"[Saved] {f3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Radial Decay Alignment & Layer Causal Ablation
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Radial Decay Alignment cos(u1, -w_ref)
    groups = ["Full Model", "Embeddings", "Attention", "MLP", "Unembed"]
    rad_vals = [
        summary["cos_radial_full"]["mean"],
        summary["cos_radial_layers"]["embed"]["mean"],
        summary["cos_radial_layers"]["attn"]["mean"],
        summary["cos_radial_layers"]["mlp"]["mean"],
        summary["cos_radial_layers"]["unembed"]["mean"]
    ]
    rad_ci = [
        summary["cos_radial_full"]["ci95"],
        summary["cos_radial_layers"]["embed"]["ci95"],
        summary["cos_radial_layers"]["attn"]["ci95"],
        summary["cos_radial_layers"]["mlp"]["ci95"],
        summary["cos_radial_layers"]["unembed"]["ci95"]
    ]

    b_colors = ["#1a73e8", "#4285f4", "#ea4335", "#fbbc04", "#34a853"]
    bars = ax1.bar(groups, rad_vals, yerr=rad_ci, capsize=4, color=b_colors, alpha=0.85, width=0.5)
    ax1.set_ylabel("Cosine Alignment |<u1, -W_ref>|")
    ax1.set_title("Pre-Grokking SVD Direction u1 vs Radial Weight Decay Direction")
    ax1.set_ylim(0, 1.05)

    for bar, val in zip(bars, rad_vals):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{val:.3f}", ha='center', va='bottom', fontweight='bold')

    # Panel 2: Layer Causal Ablation at h=100 (Step 4000)
    h_final = "h_100"
    l_ablation_groups = ["Embeddings", "Attention", "MLP", "Unembedding"]
    act_l_vals = [summary["layer_selective_ablation_r8"][l][h_final]["layer_actual_rest_projected"]["mean"] for l in ["embed", "attn", "mlp", "unembed"]]
    proj_l_vals = [summary["layer_selective_ablation_r8"][l][h_final]["layer_projected_rest_actual"]["mean"] for l in ["embed", "attn", "mlp", "unembed"]]

    x = np.arange(len(l_ablation_groups))
    w = 0.35

    ax2.bar(x - w/2, proj_l_vals, width=w, label="Layer Projected (Rest Actual)", color="#d32f2f", alpha=0.85)
    ax2.bar(x + w/2, act_l_vals, width=w, label="Layer Actual (Rest Projected)", color="#2e7d32", alpha=0.85)
    ax2.axhline(summary["actual_continuation"][h_final]["acc_nonsym"]["mean"], color="#2e7d32", linestyle="--", label=f"Full Actual ({summary['actual_continuation'][h_final]['acc_nonsym']['mean']:.1f}%)")
    ax2.axhline(summary["reconstructed_models"]["r_8"]["projected"][h_final]["mean"], color="#d32f2f", linestyle=":", label=f"Full Projected ({summary['reconstructed_models']['r_8']['projected'][h_final]['mean']:.1f}%)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(l_ablation_groups)
    ax2.set_ylabel("Strict Non-Symmetric Test Accuracy (%) at Step 4000")
    ax2.set_title("Layer Causal Necessity at Grokking Transition (Rank 8 Basis)")
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "subspace_departure_radial_and_causal_ablation.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"[Saved] {f4_path}")

if __name__ == "__main__":
    generate_subspace_departure_figures()
