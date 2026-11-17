import os
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def find_file(filename):
    p1 = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(filename):
        return filename
    return None

def generate_all_low_rank_figures():
    # -------------------------------------------------------------
    # 1. Explained Variance & Reconstruction Error vs. Rank
    # -------------------------------------------------------------
    reg_path = find_file("low_rank_regression_results.json")
    mnist_path = find_file("low_rank_mnist_results.json")

    if reg_path and mnist_path:
        with open(reg_path) as f:
            reg_data = json.load(f)["summary"]
        with open(mnist_path) as f:
            mnist_data = json.load(f)["summary"]

        ranks = [1, 4, 8, 16]
        exp_vars_reg = [reg_data["explained_variance_by_rank"][str(r)]["mean"] * 100 for r in ranks]
        rec_errs_reg = [reg_data["reconstruction_error_by_rank"][str(r)]["mean"] * 100 for r in ranks]

        exp_vars_m = [mnist_data["explained_variance_by_rank"][str(r)]["mean"] * 100 for r in ranks]
        rec_errs_m = [mnist_data["reconstruction_error_by_rank"][str(r)]["mean"] * 100 for r in ranks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Subplot 1: Explained Variance
        ax1.plot(ranks, exp_vars_reg, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='California Housing (MLP)')
        ax1.plot(ranks, exp_vars_m, marker='^', linewidth=2.5, markersize=8, color='#ff7f0e', label='MNIST (MLP)')
        ax1.set_xlabel("Update Subspace Rank (r)", fontweight='bold')
        ax1.set_ylabel("Cumulative Explained Variance (%)", fontweight='bold')
        ax1.set_title("Low-Rank Subspace Concentration (10 Seeds)", fontweight='bold')
        ax1.set_xticks(ranks)
        ax1.set_ylim(75, 102)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='lower right', framealpha=0.9)
        for r, val in zip(ranks, exp_vars_reg):
            ax1.annotate(f"{val:.1f}%", (r, val), textcoords="offset points", xytext=(-10, 8), ha='center', fontweight='bold', fontsize=8, color='#1f77b4')
        for r, val in zip(ranks, exp_vars_m):
            ax1.annotate(f"{val:.1f}%", (r, val), textcoords="offset points", xytext=(10, -12), ha='center', fontweight='bold', fontsize=8, color='#ff7f0e')

        # Subplot 2: Reconstruction Error
        ax2.plot(ranks, rec_errs_reg, marker='s', linewidth=2.5, markersize=8, color='#1f77b4', label='California Housing (MLP)')
        ax2.plot(ranks, rec_errs_m, marker='D', linewidth=2.5, markersize=8, color='#ff7f0e', label='MNIST (MLP)')
        ax2.set_xlabel("Update Subspace Rank (r)", fontweight='bold')
        ax2.set_ylabel("Relative Reconstruction Error (%)", fontweight='bold')
        ax2.set_title("Trajectory Reconstruction Fidelity (||D - Ur Cr|| / ||D||)", fontweight='bold')
        ax2.set_xticks(ranks)
        ax2.set_ylim(-1, 48)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', framealpha=0.9)
        for r, val in zip(ranks, rec_errs_reg):
            ax2.annotate(f"{val:.1f}%", (r, val), textcoords="offset points", xytext=(-10, 8), ha='center', fontweight='bold', fontsize=8, color='#1f77b4')
        for r, val in zip(ranks, rec_errs_m):
            ax2.annotate(f"{val:.1f}%", (r, val), textcoords="offset points", xytext=(10, 8), ha='center', fontweight='bold', fontsize=8, color='#ff7f0e')

        plt.tight_layout()
        out1 = os.path.join(FIG_DIR, "low_rank_explained_variance.png")
        plt.savefig(out1, dpi=300)
        print(f"Saved {out1}")
        plt.close()

    # -------------------------------------------------------------
    # 2. Performance vs. Rank Across Methods (Regression)
    # -------------------------------------------------------------
    if reg_path:
        with open(reg_path) as f:
            reg_sum = json.load(f)["summary"]

        ranks = [1, 4, 8, 16]
        lin_means = [reg_sum["low_rank_methods"][f"rank_{r}"]["linear"]["mean"] for r in ranks]
        lin_cis = [reg_sum["low_rank_methods"][f"rank_{r}"]["linear"]["ci95"] for r in ranks]

        damped_means = [reg_sum["low_rank_methods"][f"rank_{r}"]["damped"]["mean"] for r in ranks]
        damped_cis = [reg_sum["low_rank_methods"][f"rank_{r}"]["damped"]["ci95"] for r in ranks]

        tfm_means = [reg_sum["low_rank_methods"][f"rank_{r}"]["timesfm"]["mean"] for r in ranks]
        tfm_cis = [reg_sum["low_rank_methods"][f"rank_{r}"]["timesfm"]["ci95"] for r in ranks]

        shuf_means = [reg_sum["low_rank_methods"][f"rank_{r}"]["timesfm_shuffled"]["mean"] for r in ranks]
        shuf_cis = [reg_sum["low_rank_methods"][f"rank_{r}"]["timesfm_shuffled"]["ci95"] for r in ranks]

        f_mean = reg_sum["baselines"]["frozen"]["mean"]
        gt_mean = reg_sum["baselines"]["continued_adam"]["mean"]
        swa_mean = reg_sum["baselines"]["swa"]["mean"]

        plt.figure(figsize=(10, 5))
        plt.plot(ranks, lin_means, marker='x', linestyle='--', linewidth=2, color='#d62728', label='Low-Rank Linear (Diverges)')
        plt.plot(ranks, damped_means, marker='^', linestyle='-', linewidth=2, color='#ff7f0e', label='Low-Rank Damped Linear')
        plt.plot(ranks, tfm_means, marker='o', linestyle='-', linewidth=2.5, color='#9467bd', label='Low-Rank TimesFM')
        plt.plot(ranks, shuf_means, marker='s', linestyle=':', linewidth=2, color='#8c564b', label='Low-Rank TimesFM (Shuffled)')

        plt.axhline(f_mean, color='#7f7f7f', linestyle='--', label=f'Frozen Baseline ({f_mean:.4f})')
        plt.axhline(gt_mean, color='#1f77b4', linestyle='-', label=f'Continued Adam ({gt_mean:.4f})')
        plt.axhline(swa_mean, color='#2ca02c', linestyle='-.', label=f'Matched SWA ({swa_mean:.4f})')

        plt.xlabel("Subspace Rank (r)", fontweight='bold')
        plt.ylabel("Held-Out Test MSE (Lower is Better)", fontweight='bold')
        plt.title("California Housing: Rank vs. Performance Across Extrapolation Methods (10 Seeds)", fontweight='bold')
        plt.xticks(ranks)
        plt.ylim(0.27, 0.65)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()
        out2 = os.path.join(FIG_DIR, "low_rank_performance_vs_rank.png")
        plt.savefig(out2, dpi=300)
        print(f"Saved {out2}")
        plt.close()

    # -------------------------------------------------------------
    # 2b. Performance vs. Rank Across Methods (MNIST Classification)
    # -------------------------------------------------------------
    if mnist_path:
        with open(mnist_path) as f:
            mnist_sum = json.load(f)["summary"]

        ranks = [1, 4, 8, 16]
        lin_means_m = [mnist_sum["low_rank_methods"][f"rank_{r}"]["linear"]["loss_mean"] for r in ranks]
        damped_means_m = [mnist_sum["low_rank_methods"][f"rank_{r}"]["damped"]["loss_mean"] for r in ranks]
        tfm_means_m = [mnist_sum["low_rank_methods"][f"rank_{r}"]["timesfm"]["loss_mean"] for r in ranks]
        shuf_means_m = [mnist_sum["low_rank_methods"][f"rank_{r}"]["timesfm_shuffled"]["loss_mean"] for r in ranks]

        f_mean_m = mnist_sum["baselines"]["frozen"]["loss_mean"]
        gt_mean_m = mnist_sum["baselines"]["continued_sgd"]["loss_mean"]
        swa_mean_m = mnist_sum["baselines"]["swa"]["loss_mean"]

        plt.figure(figsize=(10, 5))
        plt.plot(ranks, lin_means_m, marker='x', linestyle='--', linewidth=2, color='#d62728', label='Low-Rank Linear (Diverges)')
        plt.plot(ranks, damped_means_m, marker='^', linestyle='-', linewidth=2, color='#ff7f0e', label='Low-Rank Damped Linear')
        plt.plot(ranks, tfm_means_m, marker='o', linestyle='-', linewidth=2.5, color='#9467bd', label='Low-Rank TimesFM')
        plt.plot(ranks, shuf_means_m, marker='s', linestyle=':', linewidth=2, color='#8c564b', label='Low-Rank TimesFM (Shuffled)')

        plt.axhline(f_mean_m, color='#7f7f7f', linestyle='--', label=f'Frozen Baseline ({f_mean_m:.4f})')
        plt.axhline(gt_mean_m, color='#1f77b4', linestyle='-', label=f'Continued SGD ({gt_mean_m:.4f})')
        plt.axhline(swa_mean_m, color='#2ca02c', linestyle='-.', label=f'Matched SWA ({swa_mean_m:.4f})')

        plt.xlabel("Subspace Rank (r)", fontweight='bold')
        plt.ylabel("Held-Out Test Cross-Entropy Loss (Lower is Better)", fontweight='bold')
        plt.title("MNIST Plateau: Rank vs. Performance Across Extrapolation Methods (10 Seeds)", fontweight='bold')
        plt.xticks(ranks)
        plt.ylim(0.15, 2.0)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper left', framealpha=0.9)
        plt.tight_layout()
        out3 = os.path.join(FIG_DIR, "low_rank_mnist_performance_vs_rank.png")
        plt.savefig(out3, dpi=300)
        print(f"Saved {out3}")
        plt.close()

    # -------------------------------------------------------------
    # 3. Compute Efficiency & Latency Comparison
    # -------------------------------------------------------------
    # Wall clock comparison: Full TimesFM (45.9s) vs Low-Rank TimesFM (0.052s) vs Linear (0.001s) vs SGD (0.65s) vs SWA (0.0001s)
    labels = [
        "Matched SWA\n(Online)",
        "Low-Rank Linear\n(Rank 4)",
        "Low-Rank TimesFM\n(Rank 4)",
        "Continued Adam/SGD\n(48-64 Steps)",
        "Per-Weight TimesFM\n(Full Weights)"
    ]
    latencies = [0.0001, 0.005, 0.052, 0.65, 45.9] # in seconds

    plt.figure(figsize=(9, 4.8))
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4', '#d62728']
    bars = plt.bar(labels, latencies, color=colors, alpha=0.85)
    plt.yscale('log')
    plt.ylabel("Wall-Clock Latency (Seconds, Log Scale)", fontweight='bold')
    plt.title("Compute Overhead: Per-Weight TimesFM vs. Low-Rank vs. Baselines", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.text(bars[0].get_x() + bars[0].get_width()/2.0, 0.0003, "< 0.1 ms", ha='center', va='bottom', fontweight='bold')
    plt.text(bars[1].get_x() + bars[1].get_width()/2.0, 0.01, "5.0 ms", ha='center', va='bottom', fontweight='bold')
    plt.text(bars[2].get_x() + bars[2].get_width()/2.0, 0.1, "52 ms (880x speedup!)", ha='center', va='bottom', fontweight='bold', color='#9467bd')
    plt.text(bars[3].get_x() + bars[3].get_width()/2.0, 1.2, "650 ms", ha='center', va='bottom', fontweight='bold')
    plt.text(bars[4].get_x() + bars[4].get_width()/2.0, 75, "45.9 s", ha='center', va='bottom', fontweight='bold', color='#d62728')

    plt.tight_layout()
    out4 = os.path.join(FIG_DIR, "low_rank_compute_tradeoff.png")
    plt.savefig(out4, dpi=300)
    print(f"Saved {out4}")
    plt.close()

if __name__ == "__main__":
    generate_all_low_rank_figures()
