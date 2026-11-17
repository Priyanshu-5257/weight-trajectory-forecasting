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

# -------------------------------------------------------------
# 2. Evaluation Helper
# -------------------------------------------------------------
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
# 3. Main Experiment Routine
# -------------------------------------------------------------
def run_experiment(
    total_steps=100,
    context_k=64,
    horizon_H=32,
    batch_size=64,
    lr=0.03,
    momentum=0.9,
    downsample=True,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print(f"=== Optimization Trajectory Forecasting Experiment ===")
    print(f"Device: {device}")
    print(f"Context: 0..{context_k} steps | Horizon: next {horizon_H} steps (target: step {context_k + horizon_H})")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

    weight_trajectory = [] # List of 1D tensors
    loss_trajectory = []
    acc_trajectory = []

    print("\n--- Training Model & Recording Checkpoint Trajectory ---")
    data_iter = iter(train_loader)
    
    # Initial state (step 0)
    w0 = get_flat_params(model).cpu()
    weight_trajectory.append(w0)
    init_loss, init_acc = evaluate_model(model, test_loader, device, criterion)
    loss_trajectory.append(init_loss)
    acc_trajectory.append(init_acc)
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

        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        loss_trajectory.append(test_loss)
        acc_trajectory.append(test_acc)

        if step % 16 == 0 or step == context_k or step == context_k + horizon_H:
            print(f"Step {step:3d}: Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2f}%")

    target_step = context_k + horizon_H
    w_context_k = weight_trajectory[context_k]
    w_ground_truth = weight_trajectory[target_step]
    delta_actual = w_ground_truth - w_context_k
    actual_update_norm = float(torch.norm(delta_actual).item())

    loss_k = loss_trajectory[context_k]
    acc_k = acc_trajectory[context_k]
    loss_gt = loss_trajectory[target_step]
    acc_gt = acc_trajectory[target_step]

    print("\n" + "=" * 65)
    print(f"Context Cut-off: Step {context_k:3d} | Loss: {loss_k:.4f} | Acc: {acc_k:.2f}%")
    print(f"Ground Truth:   Step {target_step:3d} | Loss: {loss_gt:.4f} | Acc: {acc_gt:.2f}%")
    print("=" * 65)

    # Historical trajectory matrix: shape (context_k + 1, num_params)
    W_hist = torch.stack(weight_trajectory[:context_k + 1], dim=0).numpy() # (K+1, P)
    W_hist_T = W_hist.T # (num_params, K+1)

    # ---------------------------------------------------------
    # Baseline 1: Linear Momentum Extrapolation (Tangent Vector)
    # ---------------------------------------------------------
    m = 5
    v_k = (W_hist[context_k] - W_hist[context_k - m]) / m
    w_pred_linear_traj = []
    for h in range(1, horizon_H + 1):
        w_pred_linear_traj.append(torch.tensor(W_hist[context_k] + h * v_k, dtype=torch.float32))
    w_pred_linear = w_pred_linear_traj[-1]

    # ---------------------------------------------------------
    # Baseline 2: Damped Momentum Extrapolation (Inertia with Decay)
    # ---------------------------------------------------------
    gamma = 0.95
    w_pred_damped_traj = []
    for h in range(1, horizon_H + 1):
        decay_sum = sum(gamma ** j for j in range(1, h + 1))
        w_pred_damped_traj.append(torch.tensor(W_hist[context_k] + decay_sum * v_k, dtype=torch.float32))
    w_pred_damped = w_pred_damped_traj[-1]

    # ---------------------------------------------------------
    # Baseline 3: Quadratic Polynomial Extrapolation
    # ---------------------------------------------------------
    print("\nFitting Polynomial (Degree 2) extrapolation...")
    t_hist = np.arange(context_k + 1)
    window = min(20, context_k)
    t_fit = t_hist[-window:]
    W_sub = W_hist_T[:, -window:]
    
    poly_coeffs = [np.polyfit(t_fit, W_sub[i], deg=2) for i in range(num_params)]
    w_pred_poly_traj = []
    for h in range(1, horizon_H + 1):
        step_val = context_k + h
        p_vals = np.array([np.polyval(c, step_val) for c in poly_coeffs])
        w_pred_poly_traj.append(torch.tensor(p_vals, dtype=torch.float32))
    w_pred_poly = w_pred_poly_traj[-1]

    # ---------------------------------------------------------
    # Method 4: Google TimesFM (Zero-Shot Foundation Model)
    # ---------------------------------------------------------
    print("\n--- Running Google TimesFM-2.0-500M Extrapolation ---")
    tfm_device = device
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(tfm_device)
    tfm_model.eval()

    # We batch forecast all parameters
    all_past_values = torch.tensor(W_hist_T, dtype=torch.float32) # (P, context_k + 1)
    tfm_batch_size = 512
    tfm_forecasts = []

    print(f"Forecasting {num_params} parameter series using batch size {tfm_batch_size}...")
    t0 = time.time()
    with torch.no_grad():
        for b_start in range(0, num_params, tfm_batch_size):
            b_end = min(b_start + tfm_batch_size, num_params)
            batch_past = all_past_values[b_start:b_end].to(tfm_device)
            out = tfm_model(past_values=batch_past)
            # mean_predictions shape: (B, 128)
            preds = out.mean_predictions[:, :horizon_H].cpu()
            tfm_forecasts.append(preds)
            print(f"  Processed {b_end}/{num_params} parameters...")

    tfm_forecasts = torch.cat(tfm_forecasts, dim=0) # (P, horizon_H)
    tfm_time = time.time() - t0
    print(f"TimesFM forecasting completed in {tfm_time:.2f} seconds.")

    # Build TimesFM forecasted trajectory
    w_pred_timesfm_traj = []
    for h in range(horizon_H):
        w_pred_timesfm_traj.append(tfm_forecasts[:, h].float())
    w_pred_timesfm = w_pred_timesfm_traj[-1]

    # ---------------------------------------------------------
    # Full Horizon Evaluation (Testing the Landscape Navigation)
    # ---------------------------------------------------------
    print("\n--- Evaluating Models on Test Set across the Horizon ---")
    eval_checkpoints = [4, 8, 12, 16, 20, 24, 28, 32]
    
    trajectory_eval = {
        "Ground_Truth": [],
        "Linear_Momentum": [],
        "Damped_Momentum": [],
        "Polynomial_Deg2": [],
        "TimesFM": []
    }

    for h in eval_checkpoints:
        idx = h - 1
        step_num = context_k + h
        # Ground Truth
        gt_w = weight_trajectory[step_num]
        set_flat_params(model, gt_w.to(device))
        l_gt, a_gt = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Ground_Truth"].append((step_num, l_gt, a_gt))

        # Linear
        set_flat_params(model, w_pred_linear_traj[idx].to(device))
        l_lin, a_lin = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Linear_Momentum"].append((step_num, l_lin, a_lin))

        # Damped
        set_flat_params(model, w_pred_damped_traj[idx].to(device))
        l_damp, a_damp = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Damped_Momentum"].append((step_num, l_damp, a_damp))

        # Polynomial
        set_flat_params(model, w_pred_poly_traj[idx].to(device))
        l_poly, a_poly = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["Polynomial_Deg2"].append((step_num, l_poly, a_poly))

        # TimesFM
        set_flat_params(model, w_pred_timesfm_traj[idx].to(device))
        l_tfm, a_tfm = evaluate_model(model, test_loader, device, criterion)
        trajectory_eval["TimesFM"].append((step_num, l_tfm, a_tfm))

    # Evaluate final metrics at step (context_k + horizon_H)
    def compute_metrics(w_pred, name):
        set_flat_params(model, w_pred.to(device))
        loss, acc = evaluate_model(model, test_loader, device, criterion)
        delta = w_pred - w_context_k
        cos_sim = float(torch.dot(delta, delta_actual) / (torch.norm(delta) * torch.norm(delta_actual) + 1e-8))
        l2_dist = float(torch.norm(w_pred - w_ground_truth).item())
        update_norm = float(torch.norm(delta).item())
        return {
            "name": name,
            "test_loss": loss,
            "test_acc": acc,
            "l2_to_gt": l2_dist,
            "cos_sim_actual_update": cos_sim,
            "update_norm": update_norm,
            "loss_change": loss - loss_k, # negative is improvement
            "acc_change": acc - acc_k
        }

    summary = []
    summary.append({
        "name": "Step_k_Frozen",
        "test_loss": loss_k,
        "test_acc": acc_k,
        "l2_to_gt": actual_update_norm,
        "cos_sim_actual_update": 0.0,
        "update_norm": 0.0,
        "loss_change": 0.0,
        "acc_change": 0.0
    })
    summary.append(compute_metrics(w_ground_truth, "Ground_Truth (SGD)"))
    summary.append(compute_metrics(w_pred_linear, "Linear_Momentum"))
    summary.append(compute_metrics(w_pred_damped, "Damped_Momentum"))
    summary.append(compute_metrics(w_pred_poly, "Polynomial_Deg2"))
    summary.append(compute_metrics(w_pred_timesfm, "TimesFM (500M)"))

    # Print Formatted Table
    print("\n" + "=" * 105)
    print(f"{'Method':<20} | {'Test Loss':<10} | {'Loss Diff':<10} | {'Test Acc (%)':<12} | {'L2 to GT':<10} | {'Cosine Sim':<10}")
    print("-" * 105)
    for m in summary:
        diff_str = f"{m['loss_change']:+.4f}"
        print(f"{m['name']:<20} | {m['test_loss']:<10.4f} | {diff_str:<10} | {m['test_acc']:<12.2f} | {m['l2_to_gt']:<10.4f} | {m['cos_sim_actual_update']:<10.4f}")
    print("=" * 105 + "\n")

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Test Loss
    plt.subplot(1, 2, 1)
    steps_x = [pt[0] for pt in trajectory_eval["Ground_Truth"]]
    for method, pts in trajectory_eval.items():
        losses = [pt[1] for pt in pts]
        plt.plot(steps_x, losses, marker='o', label=method)
    plt.axhline(y=loss_k, color='gray', linestyle='--', label=f'Frozen Step {context_k}')
    plt.title("Test Loss across Forecast Horizon")
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
    plt.title("Test Accuracy (%) across Forecast Horizon")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimization_trajectory_forecast.png", dpi=300)
    print("Saved plot to optimization_trajectory_forecast.png")

    with open("results.json", "w") as f:
        json.dump({
            "summary": summary,
            "trajectory_eval": trajectory_eval
        }, f, indent=2)
    print("Saved metrics to results.json")

    return summary, trajectory_eval

if __name__ == "__main__":
    run_experiment()
