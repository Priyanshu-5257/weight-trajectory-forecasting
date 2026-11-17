import os
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SUMMARY = os.path.join(ROOT_DIR, "results", "feasibility_benchmark_summary.json")
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def generate_all_feasibility_figures(summary_path=None):
    if summary_path is None:
        summary_path = DEFAULT_SUMMARY if os.path.exists(DEFAULT_SUMMARY) else "feasibility_benchmark_summary.json"

    if not os.path.exists(summary_path):
        print(f"Summary file {summary_path} not found. Skipping plot generation.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    reg = data["california_housing"]
    mnist = data["mnist"]
    bench = data["compute_benchmarks"]
    ranks = [1, 4, 8, 16]
    horizons = [8, 16, 32]

    # --------------------------------------------------------------------------
    # Figure 1: Rank vs. Quality (California Housing and MNIST)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # California Housing (Horizon 16)
    h_key = "h_16"
    lin_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["linear"]["test_mse"]["mean"] for r in ranks]
    dmp_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["damped"]["test_mse"]["mean"] for r in ranks]
    tfm_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["test_mse"]["mean"] for r in ranks]
    shf_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["timesfm_shuffled"]["test_mse"]["mean"] for r in ranks]

    f_reg = reg["baselines"]["frozen"]["test_mse"]["mean"]
    swa_reg = reg["baselines"]["swa"]["test_mse"]["mean"]
    cont_reg = reg["baselines"]["continued"][h_key]["test_mse"]["mean"]

    ax1.plot(ranks, lin_reg, marker='x', linestyle='--', color='#d62728', linewidth=2, label='Low-Rank Linear')
    ax1.plot(ranks, dmp_reg, marker='^', linestyle='-', color='#ff7f0e', linewidth=2, label='Low-Rank Damped Linear')
    ax1.plot(ranks, tfm_reg, marker='o', linestyle='-', color='#9467bd', linewidth=2.5, label='Low-Rank TimesFM')
    ax1.plot(ranks, shf_reg, marker='s', linestyle=':', color='#8c564b', linewidth=2, label='TimesFM (Shuffled Control)')

    ax1.axhline(f_reg, color='#7f7f7f', linestyle='--', label=f'Frozen Checkpoint ({f_reg:.4f})')
    ax1.axhline(cont_reg, color='#1f77b4', linestyle='-', label=f'Continued Adam h=16 ({cont_reg:.4f})')
    ax1.axhline(swa_reg, color='#2ca02c', linestyle='-.', label=f'Matched SWA ({swa_reg:.4f})')

    ax1.set_xlabel("Update Subspace Rank (r)", fontweight='bold')
    ax1.set_ylabel("Held-Out Test MSE (Lower is Better)", fontweight='bold')
    ax1.set_title("California Housing: Rank vs. Test MSE (Horizon 16, 10 Seeds)", fontweight='bold')
    ax1.set_xticks(ranks)
    ax1.set_ylim(0.27, 0.45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # MNIST (Horizon 16)
    lin_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["linear"]["test_loss"]["mean"] for r in ranks]
    dmp_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["damped"]["test_loss"]["mean"] for r in ranks]
    tfm_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["test_loss"]["mean"] for r in ranks]
    shf_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["timesfm_shuffled"]["test_loss"]["mean"] for r in ranks]

    f_m = mnist["baselines"]["frozen"]["test_loss"]["mean"]
    swa_m = mnist["baselines"]["swa"]["test_loss"]["mean"]
    cont_m = mnist["baselines"]["continued"][h_key]["test_loss"]["mean"]

    ax2.plot(ranks, lin_m, marker='x', linestyle='--', color='#d62728', linewidth=2, label='Low-Rank Linear')
    ax2.plot(ranks, dmp_m, marker='^', linestyle='-', color='#ff7f0e', linewidth=2, label='Low-Rank Damped Linear')
    ax2.plot(ranks, tfm_m, marker='o', linestyle='-', color='#9467bd', linewidth=2.5, label='Low-Rank TimesFM')
    ax2.plot(ranks, shf_m, marker='s', linestyle=':', color='#8c564b', linewidth=2, label='TimesFM (Shuffled Control)')

    ax2.axhline(f_m, color='#7f7f7f', linestyle='--', label=f'Frozen Checkpoint ({f_m:.4f})')
    ax2.axhline(cont_m, color='#1f77b4', linestyle='-', label=f'Continued SGD h=16 ({cont_m:.4f})')
    ax2.axhline(swa_m, color='#2ca02c', linestyle='-.', label=f'Matched SWA ({swa_m:.4f})')

    ax2.set_xlabel("Update Subspace Rank (r)", fontweight='bold')
    ax2.set_ylabel("Held-Out Test Cross-Entropy Loss (Lower is Better)", fontweight='bold')
    ax2.set_title("MNIST Plateau: Rank vs. Test Loss (Horizon 16, 10 Seeds)", fontweight='bold')
    ax2.set_xticks(ranks)
    ax2.set_ylim(0.16, 1.20)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feasibility_rank_vs_quality.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 2: Quality vs. Compute Pareto Frontier
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))

    # Latencies in milliseconds
    t_swa = 0.05 # < 0.1 ms
    t_sgd16 = mnist["baselines"]["continued"]["h_16"]["train_time_s"]["mean"] * 1000.0 # ~160 ms
    t_sgd32 = mnist["baselines"]["continued"]["h_32"]["train_time_s"]["mean"] * 1000.0 # ~320 ms
    t_tfm_val_sel = mnist["validation_selected_methods"]["timesfm"]["latency_s"]["mean"] * 1000.0 # ~50-60 ms
    t_tfm_shuf = mnist["validation_selected_methods"]["timesfm_shuffled"]["latency_s"]["mean"] * 1000.0
    t_lin = mnist["validation_selected_methods"]["linear"]["latency_s"]["mean"] * 1000.0
    t_frozen = 0.001

    loss_frozen = f_m
    loss_swa = swa_m
    loss_sgd16 = cont_m
    loss_sgd32 = mnist["baselines"]["continued"]["h_32"]["test_loss"]["mean"]
    loss_tfm = mnist["validation_selected_methods"]["timesfm"]["test_loss"]["mean"]
    loss_tfm_shuf = mnist["validation_selected_methods"]["timesfm_shuffled"]["test_loss"]["mean"]
    loss_lin = mnist["validation_selected_methods"]["linear"]["test_loss"]["mean"]

    points = [
        ("Frozen Checkpoint", t_frozen, loss_frozen, '#7f7f7f', 's', 9),
        ("Matched SWA (Online)", t_swa, loss_swa, '#2ca02c', '*', 14),
        ("Continued SGD (h=16)", t_sgd16, loss_sgd16, '#1f77b4', 'o', 9),
        ("Continued SGD (h=32)", t_sgd32, loss_sgd32, '#1f77b4', 'D', 9),
        ("Val-Selected Low-Rank TimesFM", t_tfm_val_sel, loss_tfm, '#9467bd', 'o', 11),
        ("TimesFM (Shuffled Control)", t_tfm_shuf, loss_tfm_shuf, '#8c564b', '^', 9),
    ]

    for label, lat, loss_val, col, marker, sz in points:
        plt.scatter(lat, loss_val, color=col, marker=marker, s=sz*sz, label=label, zorder=4)
        offset_y = 0.003 if "SWA" not in label else -0.005
        plt.annotate(f"{label}\n({lat:.1f}ms, {loss_val:.4f})", (lat, loss_val),
                     textcoords="offset points", xytext=(0, 10 if "SWA" not in label else -25),
                     ha='center', fontsize=8.5, fontweight='bold', color=col)

    plt.xscale('log')
    plt.xlabel("End-to-End Latency (ms, Log Scale)", fontweight='bold')
    plt.ylabel("Test Cross-Entropy Loss (Lower is Better)", fontweight='bold')
    plt.title("Quality vs. Compute Frontier on MNIST Optimization Plateau", fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.xlim(0.0005, 1000)
    plt.ylim(0.170, 0.220)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feasibility_quality_vs_compute.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 3: Measured Latency Breakdown
    # --------------------------------------------------------------------------
    fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(13, 5))

    # TimesFM Coordinate Forecasting Latency (Median vs p95 across ranks)
    bar_ranks = [str(r) for r in ranks]
    x_pos = np.arange(len(bar_ranks))
    w = 0.25

    for idx, h in enumerate(horizons):
        medians = [bench["timesfm_forecast_ms"][f"rank_{r}"][f"h_{h}"]["median"] for r in ranks]
        p95s = [bench["timesfm_forecast_ms"][f"rank_{r}"][f"h_{h}"]["p95"] for r in ranks]
        yerr = [p95 - med for p95, med in zip(p95s, medians)]
        bx1.bar(x_pos + (idx - 1)*w, medians, width=w, yerr=[[0]*len(ranks), yerr], capsize=4, label=f'Horizon {h}')

    bx1.set_xlabel("Subspace Rank (r)", fontweight='bold')
    bx1.set_ylabel("Latency (ms, Median with p95 upper bar)", fontweight='bold')
    bx1.set_title("TimesFM Coordinate Forecasting Latency", fontweight='bold')
    bx1.set_xticks(x_pos)
    bx1.set_xticklabels([f"Rank {r}" for r in ranks])
    bx1.grid(axis='y', linestyle='--', alpha=0.5)
    bx1.legend()

    # Pipeline Stage Breakdown for Rank 4, Horizon 16
    stages = ["Storage Overhead\n(<1 KB)", "SVD Decomposition\n(64 Checkpoints)", "TimesFM Coordinate\nForecasting (r=4, h=16)", "Weight\nReconstruction", "Total Forecasting\nPipeline", "Continued SGD\n(16 Steps)"]
    stage_times = [
        0.001,
        bench["svd_ms"]["median"],
        bench["timesfm_forecast_ms"]["rank_4"]["h_16"]["median"],
        bench["reconstruction_ms_r4"]["median"],
        bench["svd_ms"]["median"] + bench["timesfm_forecast_ms"]["rank_4"]["h_16"]["median"] + bench["reconstruction_ms_r4"]["median"],
        mnist["baselines"]["continued"]["h_16"]["train_time_s"]["mean"] * 1000.0
    ]
    colors = ['#7f7f7f', '#17becf', '#9467bd', '#e377c2', '#ff7f0e', '#1f77b4']
    bars = bx2.bar(stages, stage_times, color=colors, alpha=0.85)
    bx2.set_ylabel("Execution Time (ms)", fontweight='bold')
    bx2.set_title("End-to-End Pipeline Latency Breakdown vs. SGD", fontweight='bold')
    bx2.grid(axis='y', linestyle='--', alpha=0.5)
    for b, val in zip(bars, stage_times):
        bx2.text(b.get_x() + b.get_width()/2.0, val + 3, f"{val:.1f}ms", ha='center', va='bottom', fontweight='bold', fontsize=8.5)

    bx2.set_ylim(0, max(stage_times) * 1.25)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feasibility_latency_breakdown.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 4: Mechanistic Probes (Alignment to SWA vs Continuation Direction)
    # --------------------------------------------------------------------------
    fig, (cx1, cx2) = plt.subplots(1, 2, figsize=(13, 5))

    # Cosine Similarity to SWA across Ranks (Horizon 16)
    cos_swa_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["cos_sim_to_swa"]["mean"] for r in ranks]
    cos_swa_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["cos_sim_to_swa"]["mean"] for r in ranks]

    cx1.plot(ranks, cos_swa_reg, marker='o', linewidth=2.5, color='#1f77b4', label='California Housing')
    cx1.plot(ranks, cos_swa_m, marker='^', linewidth=2.5, color='#ff7f0e', label='MNIST')
    cx1.set_xlabel("Subspace Rank (r)", fontweight='bold')
    cx1.set_ylabel("Cosine Similarity to SWA Displacement", fontweight='bold')
    cx1.set_title("TimesFM Displacement Vector Alignment to SWA (h=16)", fontweight='bold')
    cx1.set_xticks(ranks)
    cx1.set_ylim(0.0, 1.05)
    cx1.grid(True, linestyle='--', alpha=0.5)
    cx1.legend()
    for r, val in zip(ranks, cos_swa_reg):
        cx1.annotate(f"{val:.2f}", (r, val), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=8.5, color='#1f77b4')
    for r, val in zip(ranks, cos_swa_m):
        cx1.annotate(f"{val:.2f}", (r, val), textcoords="offset points", xytext=(10, -12), ha='center', fontsize=8.5, color='#ff7f0e')

    # Cosine Similarity to Actual Continuation Direction
    cos_cont_reg = [reg["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["cos_sim_to_continuation"]["mean"] for r in ranks]
    cos_cont_m = [mnist["grid_evaluations"][f"rank_{r}"][h_key]["timesfm"]["cos_sim_to_continuation"]["mean"] for r in ranks]

    cx2.plot(ranks, cos_cont_reg, marker='s', linewidth=2.5, color='#1f77b4', label='California Housing')
    cx2.plot(ranks, cos_cont_m, marker='D', linewidth=2.5, color='#ff7f0e', label='MNIST')
    cx2.axhline(0.0, color='gray', linestyle='--', alpha=0.7)
    cx2.set_xlabel("Subspace Rank (r)", fontweight='bold')
    cx2.set_ylabel("Cosine Similarity to Actual Future Continuation", fontweight='bold')
    cx2.set_title("Directional Extrapolation Alignment (Forecast vs. Actual SGD/Adam)", fontweight='bold')
    cx2.set_xticks(ranks)
    cx2.set_ylim(-0.5, 0.8)
    cx2.grid(True, linestyle='--', alpha=0.5)
    cx2.legend()
    for r, val in zip(ranks, cos_cont_reg):
        cx2.annotate(f"{val:.2f}", (r, val), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=8.5, color='#1f77b4')
    for r, val in zip(ranks, cos_cont_m):
        cx2.annotate(f"{val:.2f}", (r, val), textcoords="offset points", xytext=(10, 8), ha='center', fontsize=8.5, color='#ff7f0e')

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feasibility_mechanism_alignment.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    generate_all_feasibility_figures()
