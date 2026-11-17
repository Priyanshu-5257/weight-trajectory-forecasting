import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from transformers import TimesFmModelForPrediction
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Regression Model Architecture (8 -> 32 -> 16 -> 1)
# -------------------------------------------------------------
class RegressionMLP(nn.Module):
    def __init__(self, in_features=8, hidden1=32, hidden2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def evaluate_val_mse(model, x_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x_val)
        loss = criterion(preds, y_val).item()
    return loss

# -------------------------------------------------------------
# 2. Main Regression Trajectory Experiment
# -------------------------------------------------------------
def run_regression_experiment(
    total_steps=250,
    descent_context=60,
    descent_horizon=32,
    sat_context=180,
    sat_horizon=48,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=== California Housing Regression Trajectory Experiment ===")
    print(f"Device: {device} | Optimizer: Adam (lr=0.005) | Total Steps: {total_steps}")

    # Prepare Data with Strict No-Leakage Protocol
    X_raw, y_raw = fetch_california_housing(return_X_y=True)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    # Fit scalers strictly on training data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

    X_test_scaled = scaler_x.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    X_val = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    torch.manual_seed(42)
    np.random.seed(42)

    model = RegressionMLP().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Regression MLP Parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    weight_trajectory = []
    train_loss_history = []
    val_loss_history = []

    # Initial state
    w0 = get_flat_params(model).cpu()
    weight_trajectory.append(w0)
    init_val_mse = evaluate_val_mse(model, X_val, y_val, criterion)
    val_loss_history.append((0, init_val_mse))
    train_loss_history.append(init_val_mse)
    print(f"Step   0: Val MSE = {init_val_mse:.4f}")

    data_iter = iter(train_loader)
    for step in range(1, total_steps + 1):
        try:
            bx, by = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            bx, by = next(data_iter)

        bx, by = bx.to(device), by.to(device)
        model.train()
        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()

        w_curr = get_flat_params(model).cpu()
        weight_trajectory.append(w_curr)
        train_loss_history.append(loss.item())

        if step % 30 == 0 or step == descent_context or step == sat_context or step == sat_context + sat_horizon:
            vm = evaluate_val_mse(model, X_val, y_val, criterion)
            val_loss_history.append((step, vm))
            print(f"Step {step:3d}: Train Loss={loss.item():.4f} | Val MSE={vm:.4f}")

    # Load TimesFM
    print("\nLoading TimesFM-2.0-500M...")
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    def run_tfm(past_weights_np, horizon):
        # past_weights_np: (P, T)
        tensor_in = torch.tensor(past_weights_np, dtype=torch.float32)
        out_list = []
        with torch.no_grad():
            for b_start in range(0, num_params, 512):
                b_end = min(b_start + 512, num_params)
                out = tfm_model(past_values=tensor_in[b_start:b_end].to(device))
                out_list.append(out.mean_predictions[:, :horizon].cpu().numpy())
        return np.concatenate(out_list, axis=0) # (P, horizon)

    def run_tfm_with_covariate(past_weights_np, past_losses_np, horizon):
        # 1. Forecast future loss
        loss_t = torch.tensor(past_losses_np[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            loss_pred = tfm_model(past_values=loss_t).mean_predictions[0, :horizon].cpu().numpy()

        # 2. Fit Ridge
        X_tr = past_losses_np.reshape(-1, 1)
        X_fut = loss_pred.reshape(-1, 1)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, past_weights_np.T) # (T, P)
        W_tr_fut = ridge.predict(X_fut) # (horizon, P)
        W_res = past_weights_np.T - ridge.predict(X_tr) # (T, P)

        # 3. Forecast residuals with TimesFM
        res_pred = run_tfm(W_res.T, horizon) # (P, horizon)

        # 4. Combine
        return W_tr_fut.T + res_pred # (P, horizon)

    # -------------------------------------------------------------
    # PHASE 1: Active Descent Phase (Step 60 -> Horizon 32)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 1: ACTIVE DESCENT REGIME (Cut-off Step 60, Horizon h=32)")
    print("=" * 65)
    
    W_hist_descent = torch.stack(weight_trajectory[:descent_context + 1], dim=0).numpy().T # (P, 61)
    loss_hist_descent = np.array(train_loss_history[:descent_context + 1])

    w_desc_k = weight_trajectory[descent_context]
    set_flat_params(model, w_desc_k.to(device))
    mse_desc_frozen = evaluate_val_mse(model, X_val, y_val, criterion)

    w_desc_gt = weight_trajectory[descent_context + descent_horizon]
    set_flat_params(model, w_desc_gt.to(device))
    mse_desc_gt = evaluate_val_mse(model, X_val, y_val, criterion)

    # Linear baseline
    v_desc = (weight_trajectory[descent_context] - weight_trajectory[descent_context - 5]).numpy() / 5.0
    w_desc_linear = torch.tensor(weight_trajectory[descent_context].numpy() + descent_horizon * v_desc, dtype=torch.float32)
    set_flat_params(model, w_desc_linear.to(device))
    mse_desc_linear = evaluate_val_mse(model, X_val, y_val, criterion)

    # TimesFM Alone
    print("Forecasting Descent: TimesFM Alone...")
    desc_alone = run_tfm(W_hist_descent, descent_horizon)
    w_desc_alone = torch.tensor(desc_alone[:, -1], dtype=torch.float32)
    set_flat_params(model, w_desc_alone.to(device))
    mse_desc_alone = evaluate_val_mse(model, X_val, y_val, criterion)

    # TimesFM + Covariate
    print("Forecasting Descent: TimesFM + Train Loss Covariate...")
    desc_cov = run_tfm_with_covariate(W_hist_descent, loss_hist_descent, descent_horizon)
    w_desc_cov = torch.tensor(desc_cov[:, -1], dtype=torch.float32)
    set_flat_params(model, w_desc_cov.to(device))
    mse_desc_cov = evaluate_val_mse(model, X_val, y_val, criterion)

    print("\n--- Phase 1 (Descent) Results at Step 92 ---")
    print(f"Step 60 Frozen:          Val MSE = {mse_desc_frozen:.4f}")
    print(f"Ground Truth (More Adam):Val MSE = {mse_desc_gt:.4f}")
    print(f"Linear Momentum:         Val MSE = {mse_desc_linear:.4f}")
    print(f"TimesFM Alone:           Val MSE = {mse_desc_alone:.4f}")
    print(f"TimesFM + Loss Covariate:Val MSE = {mse_desc_cov:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: Saturation Plateau Phase (Step 180 -> Horizon 48)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 2: SATURATION PLATEAU REGIME (Cut-off Step 180, Horizon h=48)")
    print("=" * 65)

    sat_window = 60
    start_s = sat_context - sat_window + 1
    W_hist_sat = torch.stack(weight_trajectory[start_s:sat_context + 1], dim=0).numpy().T # (P, 60)
    loss_hist_sat = np.array(train_loss_history[start_s:sat_context + 1])

    w_sat_k = weight_trajectory[sat_context]
    set_flat_params(model, w_sat_k.to(device))
    mse_sat_frozen = evaluate_val_mse(model, X_val, y_val, criterion)

    w_sat_gt = weight_trajectory[sat_context + sat_horizon]
    set_flat_params(model, w_sat_gt.to(device))
    mse_sat_gt = evaluate_val_mse(model, X_val, y_val, criterion)

    # SWA over plateau (steps 120..180)
    w_sat_swa = torch.mean(torch.stack(weight_trajectory[120:sat_context + 1], dim=0), dim=0)
    set_flat_params(model, w_sat_swa.to(device))
    mse_sat_swa = evaluate_val_mse(model, X_val, y_val, criterion)

    # TimesFM Alone
    print("Forecasting Saturation: TimesFM Alone...")
    sat_alone = run_tfm(W_hist_sat, sat_horizon)
    w_sat_alone = torch.tensor(sat_alone[:, -1], dtype=torch.float32)
    set_flat_params(model, w_sat_alone.to(device))
    mse_sat_alone = evaluate_val_mse(model, X_val, y_val, criterion)

    # TimesFM + Covariate
    print("Forecasting Saturation: TimesFM + Train Loss Covariate...")
    sat_cov = run_tfm_with_covariate(W_hist_sat, loss_hist_sat, sat_horizon)
    w_sat_cov = torch.tensor(sat_cov[:, -1], dtype=torch.float32)
    set_flat_params(model, w_sat_cov.to(device))
    mse_sat_cov = evaluate_val_mse(model, X_val, y_val, criterion)

    print("\n--- Phase 2 (Saturation) Results at Step 228 ---")
    print(f"Step 180 Frozen:         Val MSE = {mse_sat_frozen:.4f}")
    print(f"Ground Truth (More Adam):Val MSE = {mse_sat_gt:.4f}")
    print(f"SWA (Plateau Average):   Val MSE = {mse_sat_swa:.4f}")
    print(f"TimesFM Alone:           Val MSE = {mse_sat_alone:.4f}")
    print(f"TimesFM + Loss Covariate:Val MSE = {mse_sat_cov:.4f}")

    # Save summary
    results = {
        "phase1_descent": {
            "step_frozen": mse_desc_frozen,
            "ground_truth_adam": mse_desc_gt,
            "linear_momentum": mse_desc_linear,
            "timesfm_alone": mse_desc_alone,
            "timesfm_covariate": mse_desc_cov
        },
        "phase2_saturation": {
            "step_frozen": mse_sat_frozen,
            "ground_truth_adam": mse_sat_gt,
            "swa_plateau": mse_sat_swa,
            "timesfm_alone": mse_sat_alone,
            "timesfm_covariate": mse_sat_cov
        }
    }

    with open("regression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to regression_results.json")

    return results

if __name__ == "__main__":
    run_regression_experiment()
