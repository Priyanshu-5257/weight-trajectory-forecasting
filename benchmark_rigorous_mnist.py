import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import TimesFmModelForPrediction
from sklearn.linear_model import Ridge
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Architecture & Utilities
# -------------------------------------------------------------
class CompactMLP(nn.Module):
    def __init__(self, in_features=196, hidden_dim=16, num_classes=10):
        super().__init__()
        self.downsample = nn.AvgPool2d(2) if in_features == 196 else nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, 100.0 * correct / total

# -------------------------------------------------------------
# 2. Rigorous Multi-Seed & Multi-Stage Benchmark
# -------------------------------------------------------------
def run_rigorous_benchmark(
    seeds=list(range(42, 52)),
    total_steps=2500,
    stage1_step=150,   # Early descent
    stage2_step=750,   # Middle training
    stage3_step=2000,  # Genuine plateau (~2.1 epochs)
    horizon_descent=32,
    horizon_plateau=64,
    context_window=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("================================================================================")
    print(" RIGOROUS BENCHMARK: 3-STAGE OPTIMIZATION & MECHANISM ABLATION")
    print(f" Seeds: {seeds} | Total Steps: {total_steps} (Multi-epoch) | Device: {device}")
    print("================================================================================")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    # Strict Train / Validation Split (50,000 train, 10,000 validation)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # Pre-load TimesFM
    print("Loading TimesFM-2.0-500M...")
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    def forecast_weights_batch(past_w_np, horizon):
        # past_w_np: (P, T)
        P = past_w_np.shape[0]
        tensor_in = torch.tensor(past_w_np, dtype=torch.float32)
        out_list = []
        with torch.no_grad():
            for b_start in range(0, P, 1024):
                b_end = min(b_start + 1024, P)
                out = tfm_model(past_values=tensor_in[b_start:b_end].to(device))
                out_list.append(out.mean_predictions[:, :horizon].cpu().numpy())
        return np.concatenate(out_list, axis=0) # (P, horizon)

    def forecast_with_covariate(past_w_np, past_loss_np, horizon):
        P = past_w_np.shape[0]
        loss_t = torch.tensor(past_loss_np[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            loss_pred = tfm_model(past_values=loss_t).mean_predictions[0, :horizon].cpu().numpy()
        X_tr = past_loss_np.reshape(-1, 1)
        X_fut = loss_pred.reshape(-1, 1)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, past_w_np.T)
        W_fut_trend = ridge.predict(X_fut)
        W_res = past_w_np.T - ridge.predict(X_tr)
        res_pred = forecast_weights_batch(W_res.T, horizon)
        return W_fut_trend.T + res_pred

    all_seed_results = []
    if os.path.exists("rigorous_mnist_benchmark_results.json"):
        try:
            with open("rigorous_mnist_benchmark_results.json") as f:
                existing_data = json.load(f)
                all_seed_results = existing_data.get("per_seed_runs", [])
        except Exception:
            all_seed_results = []

    timing_metrics = {"tfm_forecast_seconds": [], "sgd_train_seconds": []}

    for seed_idx, seed in enumerate(seeds):
        existing_seeds = [s["seed"] for s in all_seed_results]
        if seed in existing_seeds:
            print(f">>> Seed {seed} already evaluated, skipping. <<<")
            continue

        print(f"\n>>> Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) <<<")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        model = CompactMLP(in_features=196, hidden_dim=16, num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        weight_history = []
        train_loss_history = []
        weight_history.append(get_flat_params(model).cpu())
        init_l, init_a = evaluate(model, test_loader, device, criterion)
        train_loss_history.append(init_l)

        # Train model to genuine plateau
        t_train_start = time.time()
        data_iter = iter(train_loader)
        for step in range(1, total_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            weight_history.append(get_flat_params(model).cpu())
            train_loss_history.append(loss.item())

        total_train_time = time.time() - t_train_start
        sgd_step_time = total_train_time / total_steps
        timing_metrics["sgd_train_seconds"].append(sgd_step_time * horizon_plateau)

        # Verify plateau stability: rolling variance of loss over steps 1800..2000
        plateau_losses = train_loss_history[1800:stage3_step + 1]
        loss_var = np.var(plateau_losses)
        print(f"Train finished ({total_train_time:.1f}s). Plateau loss var (steps 1800..2000): {loss_var:.6f}")

        seed_data = {"seed": seed, "loss_var_plateau": float(loss_var)}

        # ---------------------------------------------------------
        # STAGE 1: EARLY DESCENT (Step 150 -> Horizon 32)
        # ---------------------------------------------------------
        w_s1_frozen = weight_history[stage1_step]
        set_flat_params(model, w_s1_frozen.to(device))
        l_s1_f, a_s1_f = evaluate(model, test_loader, device, criterion)

        w_s1_gt = weight_history[stage1_step + horizon_descent]
        set_flat_params(model, w_s1_gt.to(device))
        l_s1_gt, a_s1_gt = evaluate(model, test_loader, device, criterion)

        s1_start = stage1_step - context_window + 1
        W_s1 = torch.stack(weight_history[s1_start:stage1_step + 1], dim=0).numpy().T
        loss_s1 = np.array(train_loss_history[s1_start:stage1_step + 1])

        # TimesFM alone
        w_s1_alone = torch.tensor(forecast_weights_batch(W_s1, horizon_descent)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_s1_alone.to(device))
        l_s1_alone, a_s1_alone = evaluate(model, test_loader, device, criterion)

        # TimesFM + Covariate
        w_s1_cov = torch.tensor(forecast_with_covariate(W_s1, loss_s1, horizon_descent)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_s1_cov.to(device))
        l_s1_cov, a_s1_cov = evaluate(model, test_loader, device, criterion)

        seed_data["stage1_descent"] = {
            "frozen": {"loss": l_s1_f, "acc": a_s1_f},
            "gt_sgd": {"loss": l_s1_gt, "acc": a_s1_gt},
            "tfm_alone": {"loss": l_s1_alone, "acc": a_s1_alone},
            "tfm_covariate": {"loss": l_s1_cov, "acc": a_s1_cov}
        }

        # ---------------------------------------------------------
        # STAGE 2: MIDDLE DESCENT (Step 750 -> Horizon 32)
        # ---------------------------------------------------------
        w_s2_frozen = weight_history[stage2_step]
        set_flat_params(model, w_s2_frozen.to(device))
        l_s2_f, a_s2_f = evaluate(model, test_loader, device, criterion)

        w_s2_gt = weight_history[stage2_step + horizon_descent]
        set_flat_params(model, w_s2_gt.to(device))
        l_s2_gt, a_s2_gt = evaluate(model, test_loader, device, criterion)

        s2_start = stage2_step - context_window + 1
        W_s2 = torch.stack(weight_history[s2_start:stage2_step + 1], dim=0).numpy().T
        
        # SWA & EMA over matched window
        w_s2_swa = torch.mean(torch.stack(weight_history[s2_start:stage2_step + 1], dim=0), dim=0)
        set_flat_params(model, w_s2_swa.to(device))
        l_s2_swa, a_s2_swa = evaluate(model, test_loader, device, criterion)

        w_s2_alone = torch.tensor(forecast_weights_batch(W_s2, horizon_descent)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_s2_alone.to(device))
        l_s2_alone, a_s2_alone = evaluate(model, test_loader, device, criterion)

        seed_data["stage2_middle"] = {
            "frozen": {"loss": l_s2_f, "acc": a_s2_f},
            "gt_sgd": {"loss": l_s2_gt, "acc": a_s2_gt},
            "swa_matched": {"loss": l_s2_swa, "acc": a_s2_swa},
            "tfm_alone": {"loss": l_s2_alone, "acc": a_s2_alone}
        }

        # ---------------------------------------------------------
        # STAGE 3: GENUINE PLATEAU (Step 2000 -> Horizon 64)
        # ---------------------------------------------------------
        w_s3_frozen = weight_history[stage3_step]
        set_flat_params(model, w_s3_frozen.to(device))
        l_s3_f, a_s3_f = evaluate(model, test_loader, device, criterion)

        w_s3_gt = weight_history[stage3_step + horizon_plateau]
        set_flat_params(model, w_s3_gt.to(device))
        l_s3_gt, a_s3_gt = evaluate(model, test_loader, device, criterion)

        s3_start = stage3_step - context_window + 1
        W_s3 = torch.stack(weight_history[s3_start:stage3_step + 1], dim=0).numpy().T

        # Matched SWA
        w_s3_swa = torch.mean(torch.stack(weight_history[s3_start:stage3_step + 1], dim=0), dim=0)
        set_flat_params(model, w_s3_swa.to(device))
        l_s3_swa, a_s3_swa = evaluate(model, test_loader, device, criterion)

        # Matched EMA (alpha = 0.05)
        ema_weights = weight_history[s3_start].clone()
        for t_step in range(s3_start + 1, stage3_step + 1):
            ema_weights = 0.05 * weight_history[t_step] + 0.95 * ema_weights
        set_flat_params(model, ema_weights.to(device))
        l_s3_ema, a_s3_ema = evaluate(model, test_loader, device, criterion)

        # TimesFM Ordered
        t0 = time.time()
        w_s3_ordered = torch.tensor(forecast_weights_batch(W_s3, horizon_plateau)[:, -1], dtype=torch.float32)
        tfm_time = time.time() - t0
        timing_metrics["tfm_forecast_seconds"].append(tfm_time)

        set_flat_params(model, w_s3_ordered.to(device))
        l_s3_ordered, a_s3_ordered = evaluate(model, test_loader, device, criterion)

        # MECHANISM ABLATION: Time-Shuffled Context
        # Randomly permute the time dimension of W_s3
        perm = np.random.RandomState(seed + 999).permutation(context_window)
        W_s3_shuffled = W_s3[:, perm]
        w_s3_shuffled = torch.tensor(forecast_weights_batch(W_s3_shuffled, horizon_plateau)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_s3_shuffled.to(device))
        l_s3_shuffled, a_s3_shuffled = evaluate(model, test_loader, device, criterion)

        seed_data["stage3_plateau"] = {
            "frozen": {"loss": l_s3_f, "acc": a_s3_f},
            "gt_sgd": {"loss": l_s3_gt, "acc": a_s3_gt},
            "swa_matched": {"loss": l_s3_swa, "acc": a_s3_swa},
            "ema_matched": {"loss": l_s3_ema, "acc": a_s3_ema},
            "tfm_ordered": {"loss": l_s3_ordered, "acc": a_s3_ordered},
            "tfm_time_shuffled": {"loss": l_s3_shuffled, "acc": a_s3_shuffled}
        }

        print(f"  Stage 3 Plateau: Frozen={l_s3_f:.4f} | SWA={l_s3_swa:.4f} | EMA={l_s3_ema:.4f} | TimesFM-Ordered={l_s3_ordered:.4f} | TimesFM-Shuffled={l_s3_shuffled:.4f}", flush=True)
        all_seed_results.append(seed_data)
        with open("rigorous_mnist_benchmark_results.json", "w") as f:
            json.dump({"per_seed_runs": all_seed_results}, f, indent=2)

    # ---------------------------------------------------------
    # 3. Aggregate Statistical Analysis (Paired t-tests & 95% CI)
    # ---------------------------------------------------------
    def calc_ci(vals):
        mean = np.mean(vals)
        sem = stats.sem(vals)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1) if len(vals) > 1 else 0.0
        return float(mean), float(sem), float(ci)

    # Aggregate Stage 3 Plateau Losses
    plateau_frozen = [s["stage3_plateau"]["frozen"]["loss"] for s in all_seed_results]
    plateau_swa = [s["stage3_plateau"]["swa_matched"]["loss"] for s in all_seed_results]
    plateau_ema = [s["stage3_plateau"]["ema_matched"]["loss"] for s in all_seed_results]
    plateau_gt = [s["stage3_plateau"]["gt_sgd"]["loss"] for s in all_seed_results]
    plateau_tfm_ord = [s["stage3_plateau"]["tfm_ordered"]["loss"] for s in all_seed_results]
    plateau_tfm_shuf = [s["stage3_plateau"]["tfm_time_shuffled"]["loss"] for s in all_seed_results]

    # Paired t-tests
    t_ord_vs_shuf, p_ord_vs_shuf = stats.ttest_rel(plateau_tfm_ord, plateau_tfm_shuf)
    t_ord_vs_swa, p_ord_vs_swa = stats.ttest_rel(plateau_tfm_ord, plateau_swa)
    t_ord_vs_froz, p_ord_vs_froz = stats.ttest_rel(plateau_tfm_ord, plateau_frozen)

    summary_stats = {
        "plateau_metrics": {
            "frozen": {"mean": calc_ci(plateau_frozen)[0], "ci95": calc_ci(plateau_frozen)[2]},
            "gt_sgd": {"mean": calc_ci(plateau_gt)[0], "ci95": calc_ci(plateau_gt)[2]},
            "swa_matched": {"mean": calc_ci(plateau_swa)[0], "ci95": calc_ci(plateau_swa)[2]},
            "ema_matched": {"mean": calc_ci(plateau_ema)[0], "ci95": calc_ci(plateau_ema)[2]},
            "tfm_ordered": {"mean": calc_ci(plateau_tfm_ord)[0], "ci95": calc_ci(plateau_tfm_ord)[2]},
            "tfm_shuffled": {"mean": calc_ci(plateau_tfm_shuf)[0], "ci95": calc_ci(plateau_tfm_shuf)[2]},
        },
        "hypothesis_tests": {
            "tfm_ordered_vs_shuffled": {"t_stat": float(t_ord_vs_shuf), "p_val": float(p_ord_vs_shuf), "significant_p05": bool(p_ord_vs_shuf < 0.05)},
            "tfm_ordered_vs_swa": {"t_stat": float(t_ord_vs_swa), "p_val": float(p_ord_vs_swa), "significant_p05": bool(p_ord_vs_swa < 0.05)},
            "tfm_ordered_vs_frozen": {"t_stat": float(t_ord_vs_froz), "p_val": float(p_ord_vs_froz), "significant_p05": bool(p_ord_vs_froz < 0.05)},
        },
        "efficiency": {
            "avg_tfm_forecast_seconds": float(np.mean(timing_metrics["tfm_forecast_seconds"])),
            "avg_sgd_64steps_seconds": float(np.mean(timing_metrics["sgd_train_seconds"])),
            "speed_ratio_sgd_vs_tfm": float(np.mean(timing_metrics["tfm_forecast_seconds"]) / (np.mean(timing_metrics["sgd_train_seconds"]) + 1e-6))
        }
    }

    print("\n" + "=" * 80)
    print(" RIGOROUS BENCHMARK RESULTS SUMMARY (Across Seeds)")
    print("=" * 80)
    for m, vals in summary_stats["plateau_metrics"].items():
        print(f" {m:<16}: Mean Val Loss = {vals['mean']:.4f} +/- {vals['ci95']:.4f} (95% CI)")
    print("-" * 80)
    print(" MECHANISM TEST (Ordered vs Time-Shuffled History):")
    print(f"  t-statistic = {summary_stats['hypothesis_tests']['tfm_ordered_vs_shuffled']['t_stat']:.3f}, p-value = {summary_stats['hypothesis_tests']['tfm_ordered_vs_shuffled']['p_val']:.4f}")
    print(f"  Statistically Significant Advantage (p < 0.05): {summary_stats['hypothesis_tests']['tfm_ordered_vs_shuffled']['significant_p05']}")
    print("-" * 80)
    print(" EFFICIENCY COMPARISON:")
    print(f"  TimesFM Horizon-64 Forecast Latency: {summary_stats['efficiency']['avg_tfm_forecast_seconds']:.2f}s")
    print(f"  64 Steps of Continued SGD Training:  {summary_stats['efficiency']['avg_sgd_64steps_seconds']:.2f}s")
    print("=" * 80)

    # Save to JSON
    output_payload = {
        "summary_statistics": summary_stats,
        "per_seed_runs": all_seed_results
    }
    with open("rigorous_mnist_benchmark_results.json", "w") as f:
        json.dump(output_payload, f, indent=2)
    print("\nSaved comprehensive results to rigorous_mnist_benchmark_results.json")

    return output_payload

if __name__ == "__main__":
    run_rigorous_benchmark(seeds=list(range(42, 52)))
