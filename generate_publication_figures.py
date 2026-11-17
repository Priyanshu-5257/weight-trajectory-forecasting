import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_all_figures():
    # ---------------------------------------------------------
    # 1. Clean California Housing Regression Benchmark
    # ---------------------------------------------------------
    if os.path.exists("rigorous_regression_results.json"):
        with open("rigorous_regression_results.json") as f:
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
        plt.savefig("rigorous_regression_benchmark.png", dpi=300)
        print("Saved rigorous_regression_benchmark.png")

    # ---------------------------------------------------------
    # 2. Rigorous Multi-Epoch MNIST Benchmark (Plateau)
    # ---------------------------------------------------------
    if os.path.exists("rigorous_mnist_benchmark_results.json"):
        with open("rigorous_mnist_benchmark_results.json") as f:
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
        plt.savefig("rigorous_mnist_benchmark.png", dpi=300)
        print("Saved rigorous_mnist_benchmark.png")

    # ---------------------------------------------------------
    # 3. Efficiency & Latency Comparison
    # ---------------------------------------------------------
    if os.path.exists("rigorous_mnist_benchmark_results.json"):
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
        plt.text(bars[2].get_x() + bars[2].get_width()/2.0, eff["avg_tfm_forecast_seconds"] * 1.5, f"{eff['avg_tfm_forecast_seconds']:.2f} s", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("efficiency_tradeoff.png", dpi=300)
        print("Saved efficiency_tradeoff.png")

    # ---------------------------------------------------------
    # 4. Small CNN Architecture Replication
    # ---------------------------------------------------------
    if os.path.exists("cnn_replication_results.json"):
        with open("cnn_replication_results.json") as f:
            cnn_data = json.load(f)

        f_losses = [x["frozen"]["loss"] for x in cnn_data]
        sgd_losses = [x["gt_sgd"]["loss"] for x in cnn_data]
        tfm_losses = [x["timesfm"]["loss"] for x in cnn_data]
        swa_losses = [x["swa"]["loss"] for x in cnn_data]

        methods_c = ["Frozen\n(Step 1200)", "Continued\nSGD (48 st)", "TimesFM\nZero-Shot", "Matched\nSWA"]
        means_c = [np.mean(f_losses), np.mean(sgd_losses), np.mean(tfm_losses), np.mean(swa_losses)]
        std_c = [np.std(f_losses), np.std(sgd_losses), np.std(tfm_losses), np.std(swa_losses)]

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(methods_c, means_c, yerr=std_c, capsize=6, color=['#7f7f7f', '#1f77b4', '#9467bd', '#2ca02c'], alpha=0.85)
        plt.ylabel("Test Cross-Entropy Loss (Lower is Better)")
        plt.title("Convolutional Architecture Replication (Small CNN on MNIST, 9,098 Params)")
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig("cnn_replication_benchmark.png", dpi=300)
        print("Saved cnn_replication_benchmark.png")

if __name__ == "__main__":
    generate_all_figures()
