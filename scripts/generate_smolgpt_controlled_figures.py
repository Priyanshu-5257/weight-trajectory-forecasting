import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

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

def generate_smolgpt_controlled_figures():
    summary_path = os.path.join(RESULTS_DIR, "smolgpt_controlled_summary.json")
    if not os.path.exists(summary_path):
        print("smolgpt_controlled_summary.json not found!")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    horizons = summary["metadata"]["horizons"]
    interval = summary["metadata"]["checkpoint_interval"]
    delta_steps = [h * interval for h in horizons]

    # --------------------------------------------------------------------------
    # Figure 1: PPL vs Horizon across Controlled Conditions
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    methods_plot = [
        ("full_resumed", "Full Resumed AdamW (1.25M)", "#2e7d32", "o", "-"),
        ("full_restarted", "Full Restarted AdamW (1.25M)", "#4caf50", "s", "--"),
        ("mlp_resumed", "MLP Resumed AdamW (524k)", "#1565c0", "^", "-"),
        ("mlp_restarted", "MLP Restarted AdamW (524k)", "#42a5f5", "v", "--"),
        ("w2_resumed", "W2 Resumed AdamW (175k)", "#7b1fa2", "D", "-"),
        ("w2_restarted", "W2 Restarted AdamW (175k)", "#ba68c8", "d", "--"),
        ("oracle_w2_transplant", "Oracle W2 Transplant (0k)", "#ff9800", "x", ":"),
        ("oracle_mlp_transplant", "Oracle MLP Transplant (0k)", "#d32f2f", "P", ":"),
        ("frozen_checkpoint", "Frozen Step-1000 Baseline", "#757575", "*", ":")
    ]

    for m_key, label, col, marker, ls in methods_plot:
        ppls = [summary["horizons"][f"h_{h}"]["methods"][m_key]["val_ppl"]["mean"] for h in horizons]
        cis = [summary["horizons"][f"h_{h}"]["methods"][m_key]["val_ppl"]["ci95"] for h in horizons]
        ax1.errorbar(delta_steps, ppls, yerr=cis, label=label, color=col, marker=marker, linestyle=ls, capsize=3)

    ax1.set_xlabel("Steps after Cutoff (Interval = 20)")
    ax1.set_ylabel("Validation Perplexity (Lower is Better)")
    ax1.set_title("Controlled SmolGPT Continuation (Seeds 42, 43, 44)")
    ax1.set_ylim(11.0, 36.0)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Panel 2: Paired Difference vs Full Resumed AdamW
    diff_methods = [
        ("full_restarted", "Full Restarted vs Resumed", "#4caf50", "s", "--"),
        ("mlp_resumed", "MLP Resumed vs Full Resumed", "#1565c0", "^", "-"),
        ("mlp_restarted", "MLP Restarted vs Full Resumed", "#42a5f5", "v", "--"),
        ("w2_resumed", "W2 Resumed vs Full Resumed", "#7b1fa2", "D", "-"),
        ("w2_restarted", "W2 Restarted vs Full Resumed", "#ba68c8", "d", "--"),
        ("oracle_w2_transplant", "Oracle W2 Transplant vs Full Resumed", "#ff9800", "x", ":"),
        ("frozen_checkpoint", "Frozen Checkpoint vs Full Resumed", "#757575", "*", ":")
    ]

    for m_key, label, col, marker, ls in diff_methods:
        diffs = [summary["horizons"][f"h_{h}"]["methods"][m_key]["diff_ppl_vs_full_resumed"]["mean"] for h in horizons]
        cis = [summary["horizons"][f"h_{h}"]["methods"][m_key]["diff_ppl_vs_full_resumed"]["ci95"] for h in horizons]
        ax2.errorbar(delta_steps, diffs, yerr=cis, label=label, color=col, marker=marker, linestyle=ls, capsize=3)

    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Steps after Cutoff (Interval = 20)")
    ax2.set_ylabel("Δ Validation Perplexity (vs Full Resumed AdamW)")
    ax2.set_title("Perplexity Gap vs Uninterrupted Full-Model Training")
    ax2.set_ylim(-1.0, 6.0)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_fig1 = os.path.join(FIGURES_DIR, "smolgpt_controlled_ppl_comparison.png")
    fig.savefig(out_fig1, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_fig1}")

    # --------------------------------------------------------------------------
    # Figure 2: Latency vs Quality Frontier at Horizon h=64
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    h64_methods = summary["horizons"]["h_64"]["methods"]

    scatter_items = [
        ("full_resumed", "Full Resumed AdamW", "#2e7d32", "o", 100),
        ("mlp_resumed", "MLP Resumed AdamW", "#1565c0", "^", 100),
        ("w2_resumed", "W2 Resumed AdamW", "#7b1fa2", "D", 100),
        ("mlp_restarted", "MLP Restarted AdamW", "#42a5f5", "v", 80),
        ("w2_restarted", "W2 Restarted AdamW", "#ba68c8", "d", 80),
        ("oracle_w2_transplant", "Oracle W2 Transplant", "#ff9800", "x", 80),
        ("w2_matched_swa", "W2 Matched SWA", "#009688", "p", 100),
        ("mlp_matched_swa", "MLP Matched SWA", "#00796b", "P", 100),
        ("w2_timesfm_r16", "W2 TimesFM (r=16)", "#e91e63", "*", 120),
        ("w2_linear_r16", "W2 Linear (r=16)", "#f44336", "s", 90),
        ("frozen_checkpoint", "Frozen Checkpoint", "#757575", "X", 90)
    ]

    for m_key, label, col, marker, sz in scatter_items:
        info = h64_methods[m_key]
        t_sec = info["time_ms"]["mean"] / 1000.0
        ppl = info["val_ppl"]["mean"]
        ax.scatter([t_sec], [ppl], color=col, marker=marker, s=sz, label=label, zorder=5)
        ax.annotate(label, (t_sec, ppl), textcoords="offset points", xytext=(8, -2), fontsize=8.5)

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Computation Time (seconds, log scale)")
    ax.set_ylabel("Validation Perplexity at h=64 (+1280 steps)")
    ax.set_title("Pareto Trade-off: Compute Latency vs Validation Quality (h=64)")
    ax.set_ylim(11.0, 38.0)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    out_fig2 = os.path.join(FIGURES_DIR, "smolgpt_controlled_latency_vs_quality.png")
    fig.savefig(out_fig2, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_fig2}")

if __name__ == "__main__":
    generate_smolgpt_controlled_figures()
