import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import TimesFmModelForPrediction
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Compact Model Definition
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

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
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
# 2. Main Saturation Experiment
# -------------------------------------------------------------
def run_saturation_experiment(
    total_steps=300,
    context_k=240,
    horizon_H=32,
    batch_size=64,
    lr=0.03,
    momentum=0.9,
    downsample=True,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"=== Saturated Regime Trajectory Forecasting Experiment ===")
    print(f"Device: {device}")
    print(f"Total Steps: {total_steps} | Cut-off at Saturation: Step {context_k} | Horizon: {horizon_H} steps")

    # Load MNIST
    # Load MNIST with Clean 3-Way Split (50k train / 10k val / 10k test)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    in_features = 196 if downsample else 784
    hidden_dim = 16 if downsample else 32
    torch.manual_seed(42)
    np.random.seed(42)

    model = CompactMLP(in_features=in_features, hidden_dim=hidden_dim, num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Target Architecture: {num_params} total parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    weight_trajectory = []
    loss_history = []
    acc_history = []

    print("\n--- Training Model to Full Saturation (300 Steps) ---")
    data_iter = iter(train_loader)
    
    w0 = get_flat_params(model).cpu()
    weight_trajectory.append(w0)
    init_loss, init_acc = evaluate_model(model, test_loader, device, criterion)
    loss_history.append((0, init_loss))
    acc_history.append((0, init_acc))
    print(f"Step   0: Test Loss = {init_loss:.4f}, Test Acc = {init_acc:.2f}%")

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

        if step % 30 == 0 or step == context_k or step == context_k + horizon_H:
            test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
            loss_history.append((step, test_loss))
            acc_history.append((step, test_acc))
            print(f"Step {step:3d}: Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2f}%")

    target_step = context_k + horizon_H
    w_context_k = weight_trajectory[context_k]
    w_ground_truth = weight_trajectory[target_step]
    delta_actual = w_ground_truth - w_context_k

    set_flat_params(model, w_context_k.to(device))
    loss_k, acc_k = evaluate_model(model, test_loader, device, criterion)

    set_flat_params(model, w_ground_truth.to(device))
    loss_gt, acc_gt = evaluate_model(model, test_loader, device, criterion)

    print("\n" + "=" * 65)
    print(f"Saturated Cut-off: Step {context_k:3d} | Loss: {loss_k:.4f} | Acc: {acc_k:.2f}%")
    print(f"Ground Truth:      Step {target_step:3d} | Loss: {loss_gt:.4f} | Acc: {acc_gt:.2f}%")
    print("=" * 65)

    # ---------------------------------------------------------
    # Baseline: Stochastic Weight Averaging (SWA) over Plateau
    # ---------------------------------------------------------
    # Average weights across steps 180 to context_k (last 60 saturated steps)
    swa_start = max(0, context_k - 60)
    w_plateau_stack = torch.stack(weight_trajectory[swa_start:context_k + 1], dim=0)
    w_swa = torch.mean(w_plateau_stack, dim=0)
    set_flat_params(model, w_swa.to(device))
    loss_swa, acc_swa = evaluate_model(model, test_loader, device, criterion)
    print(f"SWA (avg over steps {swa_start}..{context_k}): Loss = {loss_swa:.4f} | Acc = {acc_swa:.2f}%")

    # Full Historical trajectory matrix
    W_hist = torch.stack(weight_trajectory[:context_k + 1], dim=0).numpy() # (context_k+1, P)
    W_hist_T = W_hist.T # (P, context_k+1)

    # ---------------------------------------------------------
    # Baseline: Linear Momentum Extrapolation (Tangent Vector)
    # ---------------------------------------------------------
    m = 5
    v_k = (W_hist[context_k] - W_hist[context_k - m]) / m
    w_pred_linear_traj = []
    for h in range(1, horizon_H + 1):
        w_pred_linear_traj.append(torch.tensor(W_hist[context_k] + h * v_k, dtype=torch.float32))

    # ---------------------------------------------------------
    # Baseline: Damped Momentum Extrapolation
    # ---------------------------------------------------------
    gamma = 0.95
    w_pred_damped_traj = []
    for h in range(1, horizon_H + 1):
        decay_sum = sum(gamma ** j for j in range(1, h + 1))
        w_pred_damped_traj.append(torch.tensor(W_hist[context_k] + decay_sum * v_k, dtype=torch.float32))

    # ---------------------------------------------------------
    # Google TimesFM on Saturation Window (Last 64 Saturated Steps)
    # ---------------------------------------------------------
    print("\n--- Running Google TimesFM-2.0-500M on Saturated Trajectory ---")
    tfm_device = device
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(tfm_device)
    tfm_model.eval()

    # We evaluate two TimesFM modes:
    # 1. TimesFM-Plateau: fed only the saturated window (last 64 steps: 177..240)
    # 2. TimesFM-Full: fed the full history (steps 0..240, 241 steps)
    sat_window = 64
    past_sat = torch.tensor(W_hist_T[:, -sat_window:], dtype=torch.float32)
    past_full = torch.tensor(W_hist_T, dtype=torch.float32)

    tfm_batch_size = 512

    def run_tfm_batch(past_tensor, name):
        forecasts = []
        t0 = time.time()
        print(f"Forecasting {name} (context length={past_tensor.size(1)})...")
        with torch.no_grad():
            for b_start in range(0, num_params, tfm_batch_size):
                b_end = min(b_start + tfm_batch_size, num_params)
                batch_past = past_tensor[b_start:b_end].to(tfm_device)
                out = tfm_model(past_values=batch_past)
                preds = out.mean_predictions[:, :horizon_H].cpu()
                forecasts.append(preds)
        print(f"Finished {name} in {time.time() - t0:.2f}s.")
        return torch.cat(forecasts, dim=0) # (P, horizon_H)

    tfm_forecasts_sat = run_tfm_batch(past_sat, "TimesFM (Plateau Window)")
    tfm_forecasts_full = run_tfm_batch(past_full, "TimesFM (Full History)")

    w_pred_tfm_sat_traj = [tfm_forecasts_sat[:, h].float() for h in range(horizon_H)]
    w_pred_tfm_full_traj = [tfm_forecasts_full[:, h].float() for h in range(horizon_H)]

    # ---------------------------------------------------------
    # Full Horizon Evaluation Across the Saturation Basin
    # ---------------------------------------------------------
    print("\n--- Evaluating Models across the Saturated Horizon ---")
    eval_checkpoints = [4, 8, 12, 16, 20, 24, 28, 32]
    
    trajectory_eval = {
        "Ground_Truth (SGD)": [],
        "Linear_Momentum": [],
        "Damped_Momentum": [],
        "TimesFM (Plateau Window)": [],
        "TimesFM (Full History)": []
    }

    for h in eval_checkpoints:
        idx = h - 1
        step_num = context_k + h

        # Ground Truth
        gt_w = weight_trajectory[step_num]
        set_flat_params(model, gt_w.to(device))
        l_gt, a_gt = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Ground_Truth (SGD)"].append((step_num, l_gt, a_gt))

        # Linear
        set_flat_params(model, w_pred_linear_traj[idx].to(device))
        l_lin, a_lin = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Linear_Momentum"].append((step_num, l_lin, a_lin))

        # Damped
        set_flat_params(model, w_pred_damped_traj[idx].to(device))
        l_damp, a_damp = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Damped_Momentum"].append((step_num, l_damp, a_damp))

        # TimesFM Plateau
        set_flat_params(model, w_pred_tfm_sat_traj[idx].to(device))
        l_sat, a_sat = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["TimesFM (Plateau Window)"].append((step_num, l_sat, a_sat))

        # TimesFM Full
        set_flat_params(model, w_pred_tfm_full_traj[idx].to(device))
        l_full, a_full = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["TimesFM (Full History)"].append((step_num, l_full, a_full))

    # Evaluate summary at step (context_k + horizon_H)
    def compute_metrics(w_pred, name):
        set_flat_params(model, w_pred.to(device))
        loss, acc = evaluate_model(model, test_loader, device, criterion)
        delta = w_pred - w_context_k
        cos_sim = float(torch.dot(delta, delta_actual) / (torch.norm(delta) * torch.norm(delta_actual) + 1e-8))
        l2_to_gt = float(torch.norm(w_pred - w_ground_truth).item())
        l2_to_swa = float(torch.norm(w_pred - w_swa).item())
        return {
            "name": name,
            "test_loss": loss,
            "test_acc": acc,
            "l2_to_gt": l2_to_gt,
            "l2_to_swa": l2_to_swa,
            "cos_sim_actual_update": cos_sim,
            "loss_change": loss - loss_k,
            "acc_change": acc - acc_k
        }

    summary = []
    summary.append({
        "name": "Step_k_Frozen",
        "test_loss": loss_k,
        "test_acc": acc_k,
        "l2_to_gt": float(torch.norm(delta_actual).item()),
        "l2_to_swa": float(torch.norm(w_context_k - w_swa).item()),
        "cos_sim_actual_update": 0.0,
        "loss_change": 0.0,
        "acc_change": 0.0
    })
    summary.append(compute_metrics(w_swa, "SWA (Plateau Avg)"))
    summary.append(compute_metrics(w_ground_truth, "Ground_Truth (SGD)"))
    summary.append(compute_metrics(w_pred_linear_traj[-1], "Linear_Momentum"))
    summary.append(compute_metrics(w_pred_damped_traj[-1], "Damped_Momentum"))
    summary.append(compute_metrics(w_pred_tfm_sat_traj[-1], "TimesFM (Plateau)"))
    summary.append(compute_metrics(w_pred_tfm_full_traj[-1], "TimesFM (Full)"))

    # Print Formatted Table
    print("\n" + "=" * 115)
    print(f"{'Method':<26} | {'Test Loss':<10} | {'Loss Diff':<10} | {'Test Acc (%)':<12} | {'L2 to GT':<10} | {'L2 to SWA':<10}")
    print("-" * 115)
    for m in summary:
        diff_str = f"{m['loss_change']:+.4f}"
        print(f"{m['name']:<26} | {m['test_loss']:<10.4f} | {diff_str:<10} | {m['test_acc']:<12.2f} | {m['l2_to_gt']:<10.4f} | {m['l2_to_swa']:<10.4f}")
    print("=" * 115 + "\n")

    # Plot results
    plt.figure(figsize=(13, 5))
    
    # Subplot 1: Test Loss
    plt.subplot(1, 2, 1)
    steps_x = [pt[0] for pt in trajectory_eval["Ground_Truth (SGD)"]]
    for method, pts in trajectory_eval.items():
        losses = [pt[1] for pt in pts]
        plt.plot(steps_x, losses, marker='o', label=method)
    plt.axhline(y=loss_k, color='gray', linestyle='--', label=f'Frozen Step {context_k}')
    plt.axhline(y=loss_swa, color='green', linestyle=':', label='SWA (Plateau Avg)')
    plt.title("Test Loss at Saturation Regime")
    plt.xlabel("Step")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Test Accuracy
    plt.subplot(1, 2, 2)
    for method, pts in trajectory_eval.items():
        accs = [pt[2] for pt in pts]
        plt.plot(steps_x, accs, marker='o', label=method)
    plt.axhline(y=acc_k, color='gray', linestyle='--', label=f'Frozen Step {context_k}')
    plt.axhline(y=acc_swa, color='green', linestyle=':', label='SWA (Plateau Avg)')
    plt.title("Test Accuracy (%) at Saturation Regime")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("saturation_trajectory_forecast.png", dpi=300)
    print("Saved plot to saturation_trajectory_forecast.png")

    with open("saturation_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "trajectory_eval": trajectory_eval
        }, f, indent=2)
    print("Saved metrics to saturation_results.json")

    return summary, trajectory_eval

if __name__ == "__main__":
    run_saturation_experiment()
