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
import matplotlib.pyplot as plt

from experiment_saturation_forecast import CompactMLP, get_flat_params, set_flat_params, evaluate_model

def run_saturation_covariate_experiment(
    total_steps=300,
    context_k=240,
    horizon_H=64,
    sat_window=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"=== Saturation Regime: TimesFM + Train Loss Covariate Experiment ===")
    print(f"Device: {device}")
    print(f"Context Cut-off: Step {context_k} | Horizon: {horizon_H} steps | Saturation Window: {sat_window} steps")

    # Load MNIST with Clean 3-Way Split (50k train / 10k val / 10k test)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)

    model = CompactMLP(in_features=196, hidden_dim=16, num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    weight_trajectory = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    # Initial state
    w0 = get_flat_params(model).cpu()
    weight_trajectory.append(w0)
    init_val_loss, init_val_acc = evaluate_model(model, test_loader, device, criterion)
    train_loss_history.append(init_val_loss)

    print("\n--- Training Model to Saturation (300 Steps) ---")
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

        w_curr = get_flat_params(model).cpu()
        weight_trajectory.append(w_curr)
        train_loss_history.append(loss.item())

        if step % 30 == 0 or step == context_k:
            vl, va = evaluate_model(model, test_loader, device, criterion)
            print(f"Step {step:3d}: Train Loss={loss.item():.4f} | Val Loss={vl:.4f} | Val Acc={va:.2f}%")

    # Step 240 state
    w_context_k = weight_trajectory[context_k]
    set_flat_params(model, w_context_k.to(device))
    loss_k, acc_k = evaluate_model(model, test_loader, device, criterion)

    # SWA over steps 180..240
    w_swa = torch.mean(torch.stack(weight_trajectory[180:context_k + 1], dim=0), dim=0)
    set_flat_params(model, w_swa.to(device))
    loss_swa, acc_swa = evaluate_model(model, test_loader, device, criterion)
    print(f"\nBaseline Step 240 Frozen: Val Loss={loss_k:.4f}, Val Acc={acc_k:.2f}%")
    print(f"Baseline SWA (180..240):  Val Loss={loss_swa:.4f}, Val Acc={acc_swa:.2f}%")

    # Slice saturated window
    start_idx = context_k - sat_window + 1
    W_sat = torch.stack(weight_trajectory[start_idx:context_k + 1], dim=0).numpy() # (sat_window, P)
    train_loss_sat = np.array(train_loss_history[start_idx:context_k + 1]) # (sat_window,)

    # Load TimesFM
    print("\nLoading TimesFM-2.0-500M...")
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    # -------------------------------------------------------------
    # 1. TimesFM Alone (Pure Weights)
    # -------------------------------------------------------------
    print(f"\n--- Forecasting TimesFM Alone (horizon={horizon_H}) ---")
    past_weights_tensor = torch.tensor(W_sat.T, dtype=torch.float32) # (P, sat_window)
    alone_forecasts = []
    t0 = time.time()
    with torch.no_grad():
        for b_start in range(0, num_params, 512):
            b_end = min(b_start + 512, num_params)
            out = tfm_model(past_values=past_weights_tensor[b_start:b_end].to(device))
            alone_forecasts.append(out.mean_predictions[:, :horizon_H].cpu().numpy())
    alone_forecasts = np.concatenate(alone_forecasts, axis=0) # (P, horizon_H)
    print(f"TimesFM Alone finished in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------
    # 2. TimesFM + Train Loss Covariate (XReg + TimesFM)
    # -------------------------------------------------------------
    print(f"\n--- Forecasting TimesFM + Train Loss Covariate (horizon={horizon_H}) ---")
    # Step A: Forecast train loss
    t0 = time.time()
    loss_tensor = torch.tensor(train_loss_sat[None, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        loss_forecast = tfm_model(past_values=loss_tensor).mean_predictions[0, :horizon_H].cpu().numpy()

    # Step B: Fit Ridge regression w_i(t) = a_i + b_i * L(t) on the plateau
    X_train = train_loss_sat.reshape(-1, 1)
    X_future = loss_forecast.reshape(-1, 1)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, W_sat) # (sat_window, P)
    W_trend_hist = ridge.predict(X_train)
    W_trend_future = ridge.predict(X_future) # (horizon_H, P)
    W_residuals = W_sat - W_trend_hist # (sat_window, P)

    # Step C: Forecast residuals with TimesFM
    res_tensor = torch.tensor(W_residuals.T, dtype=torch.float32)
    res_forecasts = []
    with torch.no_grad():
        for b_start in range(0, num_params, 512):
            b_end = min(b_start + 512, num_params)
            out = tfm_model(past_values=res_tensor[b_start:b_end].to(device))
            res_forecasts.append(out.mean_predictions[:, :horizon_H].cpu().numpy())
    res_forecasts = np.concatenate(res_forecasts, axis=0) # (P, horizon_H)

    # Step D: Reconstruct
    W_cov_forecasts = W_trend_future.T + res_forecasts # (P, horizon_H)
    print(f"TimesFM + Covariate finished in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------
    # 3. Horizon Comparison
    # -------------------------------------------------------------
    eval_horizons = [8, 16, 24, 32, 40, 48, 56, 64]
    results_table = []

    print("\n" + "=" * 115)
    print(f"{'Horizon (h)':<12} | {'TimesFM Alone Loss':<20} | {'TimesFM Alone Acc':<18} | {'TimesFM+Loss Loss':<20} | {'TimesFM+Loss Acc':<18} | {'Dist to SWA (Cov)':<15}")
    print("-" * 115)

    comparison_data = {
        "horizons": eval_horizons,
        "tfm_alone_loss": [],
        "tfm_alone_acc": [],
        "tfm_cov_loss": [],
        "tfm_cov_acc": [],
        "dist_to_swa_alone": [],
        "dist_to_swa_cov": [],
        "ground_truth_sgd_loss": [],
        "ground_truth_sgd_acc": []
    }

    for h in eval_horizons:
        step_num = context_k + h

        # TimesFM Alone
        w_alone = torch.tensor(alone_forecasts[:, h - 1], dtype=torch.float32)
        set_flat_params(model, w_alone.to(device))
        l_alone, a_alone = evaluate_model(model, test_loader, device, criterion)
        d_swa_alone = float(torch.norm(w_alone - w_swa).item())

        # TimesFM + Covariate
        w_cov = torch.tensor(W_cov_forecasts[:, h - 1], dtype=torch.float32)
        set_flat_params(model, w_cov.to(device))
        l_cov, a_cov = evaluate_model(model, test_loader, device, criterion)
        d_swa_cov = float(torch.norm(w_cov - w_swa).item())

        # Ground Truth SGD if within 300 steps
        if step_num <= total_steps:
            w_gt = weight_trajectory[step_num]
            set_flat_params(model, w_gt.to(device))
            l_gt, a_gt = evaluate_model(model, test_loader, device, criterion)
        else:
            l_gt, a_gt = None, None

        comparison_data["tfm_alone_loss"].append(l_alone)
        comparison_data["tfm_alone_acc"].append(a_alone)
        comparison_data["tfm_cov_loss"].append(l_cov)
        comparison_data["tfm_cov_acc"].append(a_cov)
        comparison_data["dist_to_swa_alone"].append(d_swa_alone)
        comparison_data["dist_to_swa_cov"].append(d_swa_cov)
        comparison_data["ground_truth_sgd_loss"].append(l_gt)
        comparison_data["ground_truth_sgd_acc"].append(a_gt)

        print(f"h = {h:<7d} | {l_alone:<20.4f} | {a_alone:<18.2f}% | {l_cov:<20.4f} | {a_cov:<18.2f}% | {d_swa_cov:<15.4f}")

    print("=" * 115)
    print(f"Baseline Checkpoint 240: Loss = {loss_k:.4f} | Acc = {acc_k:.2f}%")
    print(f"Baseline SWA:            Loss = {loss_swa:.4f} | Acc = {acc_swa:.2f}%")

    # Save Plot
    plt.figure(figsize=(13, 5))
    
    # Subplot 1: Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(eval_horizons, comparison_data["tfm_alone_loss"], marker='o', label="TimesFM Alone", color='orange')
    plt.plot(eval_horizons, comparison_data["tfm_cov_loss"], marker='s', label="TimesFM + Train Loss Covariate", color='purple')
    gt_valid = [(h, l) for h, l in zip(eval_horizons, comparison_data["ground_truth_sgd_loss"]) if l is not None]
    if gt_valid:
        plt.plot([x[0] for x in gt_valid], [x[1] for x in gt_valid], marker='x', label="Ground Truth (More SGD)", color='blue', linestyle='--')
    plt.axhline(y=loss_k, color='gray', linestyle='--', label=f'Step 240 Frozen ({loss_k:.4f})')
    plt.axhline(y=loss_swa, color='green', linestyle=':', label=f'SWA Plateau Mean ({loss_swa:.4f})')
    plt.title("Validation Loss: TimesFM Alone vs. TimesFM + Loss Covariate")
    plt.xlabel("Horizon h (Steps ahead of 240)")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(eval_horizons, comparison_data["tfm_alone_acc"], marker='o', label="TimesFM Alone", color='orange')
    plt.plot(eval_horizons, comparison_data["tfm_cov_acc"], marker='s', label="TimesFM + Train Loss Covariate", color='purple')
    gt_acc_valid = [(h, a) for h, a in zip(eval_horizons, comparison_data["ground_truth_sgd_acc"]) if a is not None]
    if gt_acc_valid:
        plt.plot([x[0] for x in gt_acc_valid], [x[1] for x in gt_acc_valid], marker='x', label="Ground Truth (More SGD)", color='blue', linestyle='--')
    plt.axhline(y=acc_k, color='gray', linestyle='--', label=f'Step 240 Frozen ({acc_k:.2f}%)')
    plt.axhline(y=acc_swa, color='green', linestyle=':', label=f'SWA Plateau Mean ({acc_swa:.2f}%)')
    plt.title("Validation Accuracy (%): TimesFM Alone vs. + Covariate")
    plt.xlabel("Horizon h (Steps ahead of 240)")
    plt.ylabel("Val Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir = os.path.join(root_dir, "figures")
    res_dir = os.path.join(root_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    plt.tight_layout()
    out_fig = os.path.join(fig_dir, "saturation_covariate_forecast.png")
    plt.savefig(out_fig, dpi=300)
    print(f"\nSaved plot to {out_fig}")
    plt.close()

    out_res = os.path.join(res_dir, "saturation_covariate_results.json")
    with open(out_res, "w") as f:
        json.dump({
            "step_240_frozen": {"loss": loss_k, "acc": acc_k},
            "swa": {"loss": loss_swa, "acc": acc_swa},
            "comparison": comparison_data
        }, f, indent=2)
    print(f"Saved metrics to {out_res}")

if __name__ == "__main__":
    run_saturation_covariate_experiment()
