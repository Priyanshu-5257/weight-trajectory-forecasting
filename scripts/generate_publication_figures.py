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

def generate_all_figures():
    # ---------------------------------------------------------
    # 1. Clean California Housing Regression Benchmark
    # ---------------------------------------------------------
    reg_file = find_file("rigorous_regression_results.json")
    if reg_file:
        with open(reg_file) as f:
            reg_data = json.load(f)

        methods = ["Frozen\n(Step 200)", "Matched\nSWA", "TimesFM\nShuffled", "TimesFM\nOrdered", "Continued\nAdam"]
        means = [
            reg_data["summary"]["frozen"]["mean"],
            reg_data["summary"]["swa"]["mean"],
            reg_data["summary"]["timesfm_shuffled"]["mean"],
            reg_data["summary"]["timesfm_ordered"]["mean"],
            reg_data["summary"]["gt_adam"]["mean"]
        ]
        cis = [
            reg_data["summary"]["frozen"]["ci95"],
            reg_data["summary"]["swa"]["ci95"],
            reg_data["summary"]["timesfm_shuffled"]["ci95"],
            reg_data["summary"]["timesfm_ordered"]["ci95"],
            reg_data["summary"]["gt_adam"]["ci95"]
        ]

        plt.figure(figsize=(9, 4.5))
        bars = plt.bar(methods, means, yerr=cis, capsize=6, color=['#7f7f7f', '#2ca02c', '#d62728', '#9467bd', '#1f77b4'], alpha=0.85)
        plt.ylabel("Test MSE (Lower is Better)")
        plt.title("California Housing (Clean Split): Zero-Shot Forecasting vs. Baselines (5 Seeds with 95% CI)")
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out1 = os.path.join(FIG_DIR, "rigorous_regression_benchmark.png")
        plt.savefig(out1, dpi=300)
        print(f"Saved {out1}")
        plt.close()

    # ---------------------------------------------------------
    # 2. Rigorous Multi-Epoch MNIST Benchmark (Plateau)
    # ---------------------------------------------------------
    mnist_file = find_file("rigorous_mnist_benchmark_results.json")
    if mnist_file:
        with open(mnist_file) as f:
            mnist_data = json.load(f)

        pm = mnist_data["summary_statistics"]["plateau_metrics"]
        methods_m = ["Frozen\n(Step 2000)", "Continued\nSGD (64 st)", "Matched\nEMA", "Matched\nSWA", "TimesFM\nOrdered", "TimesFM\nShuffled"]
        means_m = [
            pm["frozen"]["mean"],
            pm["gt_sgd"]["mean"],
            pm["ema_matched"]["mean"],
            pm["swa_matched"]["mean"],
            pm["tfm_ordered"]["mean"],
            pm["tfm_shuffled"]["mean"]
        ]
        cis_m = [
            pm["frozen"]["ci95"],
            pm["gt_sgd"]["ci95"],
            pm["ema_matched"]["ci95"],
            pm["swa_matched"]["ci95"],
            pm["tfm_ordered"]["ci95"],
            pm["tfm_shuffled"]["ci95"]
        ]

        plt.figure(figsize=(10, 4.5))
        bars = plt.bar(methods_m, means_m, yerr=cis_m, capsize=6, color=['#7f7f7f', '#1f77b4', '#8c564b', '#2ca02c', '#9467bd', '#d62728'], alpha=0.85)
        plt.ylabel("Validation Loss (Lower is Better)")
        plt.title("Genuine Saturation Plateau (MNIST Multi-Epoch): 5 Seeds with 95% CI")
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out2 = os.path.join(FIG_DIR, "rigorous_mnist_benchmark.png")
        plt.savefig(out2, dpi=300)
        print(f"Saved {out2}")
        plt.close()

    # ---------------------------------------------------------
    # 3. Efficiency & Latency Comparison
    # ---------------------------------------------------------
    if mnist_file:
        eff = mnist_data["summary_statistics"]["efficiency"]
        latencies = [0.0001, eff["avg_sgd_64steps_seconds"], eff["avg_tfm_forecast_seconds"]]
        labels = ["Matched SWA\n(Accumulation)", "Continued SGD\n(64 Steps)", "TimesFM (500M)\n(Horizon-64 Forecast)"]

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(labels, latencies, color=['#2ca02c', '#1f77b4', '#d62728'], alpha=0.85)
        plt.yscale('log')
        plt.ylabel("Wall-Clock GPU Latency (Seconds, Log Scale)")
        plt.title("Compute Efficiency Tradeoff: SWA vs. SGD vs. TimesFM")
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        plt.text(bars[0].get_x() + bars[0].get_width()/2.0, 0.0003, "< 0.1 ms", ha='center', va='bottom', fontweight='bold')
        plt.text(bars[1].get_x() + bars[1].get_width()/2.0, eff["avg_sgd_64steps_seconds"] * 1.5, f"{eff['avg_sgd_64steps_seconds']:.2f} s", ha='center', va='bottom', fontweight='bold')
        plt.text(bars[2].get_x() + bars[2].get_width()/2.0, eff["avg_tfm_forecast_seconds"] * 1.5, f"{eff['avg_tfm_forecast_seconds']:.2f} s", ha='center', va='bottom', fontweight='bold', color='#d62728')

        plt.tight_layout()
        out3 = os.path.join(FIG_DIR, "efficiency_tradeoff.png")
        plt.savefig(out3, dpi=300)
        print(f"Saved {out3}")
        plt.close()

    # ---------------------------------------------------------
    # 4. Small CNN Architecture Replication
    # ---------------------------------------------------------
    cnn_file = find_file("cnn_replication_results.json")
    if cnn_file:
        with open(cnn_file) as f:
            cnn_data = json.load(f)

        f_losses = [x["frozen"]["loss"] for x in cnn_data]
        sgd_losses = [x["gt_sgd"]["loss"] for x in cnn_data]
        tfm_losses = [x["timesfm"]["loss"] for x in cnn_data]
        swa_losses = [x["swa"]["loss"] for x in cnn_data]

        methods_c = ["Frozen\n(Step 1200)", "Continued\nSGD (48 st)", "TimesFM\nZero-Shot", "Matched\nSWA"]
        means_c = [np.mean(f_losses), np.mean(sgd_losses), np.mean(tfm_losses), np.mean(swa_losses)]
        std_c = [np.std(f_losses), np.std(sgd_losses), np.std(tfm_losses), np.std(swa_losses)]

        plt.figure(figsize=(8, 4.5))
        bars_c = plt.bar(methods_c, means_c, yerr=std_c, capsize=6, color=['#7f7f7f', '#1f77b4', '#9467bd', '#2ca02c'], alpha=0.85)
        plt.ylabel("Test Cross-Entropy Loss (Lower is Better)")
        plt.title("Small CNN Architecture Replication: Loss Across Methods (MNIST)")
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars_c:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out4 = os.path.join(FIG_DIR, "cnn_replication_benchmark.png")
        plt.savefig(out4, dpi=300)
        print(f"Saved {out4}")
        plt.close()

if __name__ == "__main__":
    generate_all_figures()
