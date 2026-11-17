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
from scipy import stats

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

def evaluate_mse(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        loss = criterion(preds, y_tensor).item()
    return loss

def run_clean_regression_benchmark(
    seeds=[42, 43, 44, 45, 46],
    total_steps=300,
    sat_step=200,
    horizon=48,
    context_len=60,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("================================================================================")
    print(" RIGOROUS REGRESSION BENCHMARK (LEAKAGE-FREE SPLIT)")
    print(f" Seeds: {seeds} | Total Steps: {total_steps} | Horizon: {horizon}")
    print("================================================================================")

    # 1. Load Raw Data
    X_raw, y_raw = fetch_california_housing(return_X_y=True)

    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    criterion = nn.MSELoss()
    all_seed_results = []

    def forecast_weights(past_w_np, h):
        P = past_w_np.shape[0]
        tensor_in = torch.tensor(past_w_np, dtype=torch.float32)
        out_list = []
        with torch.no_grad():
            for b_start in range(0, P, 1024):
                b_end = min(b_start + 1024, P)
                out = tfm_model(past_values=tensor_in[b_start:b_end].to(device))
                out_list.append(out.mean_predictions[:, :h].cpu().numpy())
        return np.concatenate(out_list, axis=0)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Running Regression Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 2. Strict Leakage-Free Split (70% Train, 15% Val, 15% Test)
        X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
            X_raw, y_raw, test_size=0.3, random_state=seed
        )
        X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed
        )

        # 3. Fit Scalers ONLY on Train Split!
        scaler_x = StandardScaler().fit(X_train_raw)
        scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))

        X_train = scaler_x.transform(X_train_raw)
        y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()

        X_val = torch.tensor(scaler_x.transform(X_val_raw), dtype=torch.float32).to(device)
        y_val = torch.tensor(scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten(), dtype=torch.float32).to(device)

        X_test = torch.tensor(scaler_x.transform(X_test_raw), dtype=torch.float32).to(device)
        y_test = torch.tensor(scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten(), dtype=torch.float32).to(device)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = RegressionMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

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

            weight_history.append(get_flat_params(model).cpu())

        # Checkpoint evaluation at Saturation (Step 200)
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        test_mse_frozen = evaluate_mse(model, X_test, y_test, criterion)

        # Ground Truth Adam continuation (Step 248)
        w_gt = weight_history[sat_step + horizon]
        set_flat_params(model, w_gt.to(device))
        test_mse_gt = evaluate_mse(model, X_test, y_test, criterion)

        # Matched SWA (over steps 141..200)
        s_start = sat_step - context_len + 1
        W_context = torch.stack(weight_history[s_start:sat_step + 1], dim=0).numpy().T # (P, 60)
        w_swa = torch.mean(torch.stack(weight_history[s_start:sat_step + 1], dim=0), dim=0)
        set_flat_params(model, w_swa.to(device))
        test_mse_swa = evaluate_mse(model, X_test, y_test, criterion)

        # TimesFM Ordered Forecast
        w_tfm_ord = torch.tensor(forecast_weights(W_context, horizon)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_tfm_ord.to(device))
        test_mse_tfm = evaluate_mse(model, X_test, y_test, criterion)

        # Mechanism Ablation: TimesFM Shuffled
        perm = np.random.RandomState(seed + 777).permutation(context_len)
        W_shuffled = W_context[:, perm]
        w_tfm_shuf = torch.tensor(forecast_weights(W_shuffled, horizon)[:, -1], dtype=torch.float32)
        set_flat_params(model, w_tfm_shuf.to(device))
        test_mse_shuf = evaluate_mse(model, X_test, y_test, criterion)

        print(f"  Seed {seed} Test MSE: Frozen={test_mse_frozen:.4f} | SWA={test_mse_swa:.4f} | TimesFM={test_mse_tfm:.4f} | Shuffled={test_mse_shuf:.4f} | More Adam={test_mse_gt:.4f}")
        all_seed_results.append({
            "seed": seed,
            "test_mse_frozen": test_mse_frozen,
            "test_mse_gt_adam": test_mse_gt,
            "test_mse_swa": test_mse_swa,
            "test_mse_timesfm": test_mse_tfm,
            "test_mse_shuffled": test_mse_shuf
        })

    def calc_ci(vals):
        return float(np.mean(vals)), float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1))

    f_m, f_ci = calc_ci([r["test_mse_frozen"] for r in all_seed_results])
    s_m, s_ci = calc_ci([r["test_mse_swa"] for r in all_seed_results])
    t_m, t_ci = calc_ci([r["test_mse_timesfm"] for r in all_seed_results])
    sh_m, sh_ci = calc_ci([r["test_mse_shuffled"] for r in all_seed_results])
    gt_m, gt_ci = calc_ci([r["test_mse_gt_adam"] for r in all_seed_results])

    print("\n" + "=" * 80)
    print(" CLEAN REGRESSION BENCHMARK SUMMARY (HELD-OUT TEST SET)")
    print("=" * 80)
    print(f" Frozen Checkpoint:   Test MSE = {f_m:.4f} +/- {f_ci:.4f}")
    print(f" Continued Adam:       Test MSE = {gt_m:.4f} +/- {gt_ci:.4f}")
    print(f" Matched SWA:          Test MSE = {s_m:.4f} +/- {s_ci:.4f}")
    print(f" TimesFM Ordered:      Test MSE = {t_m:.4f} +/- {t_ci:.4f}")
    print(f" TimesFM Time-Shuffled:Test MSE = {sh_m:.4f} +/- {sh_ci:.4f}")
    print("=" * 80)

    with open("rigorous_regression_results.json", "w") as f:
        json.dump({
            "summary": {
                "frozen": {"mean": f_m, "ci95": f_ci},
                "gt_adam": {"mean": gt_m, "ci95": gt_ci},
                "swa": {"mean": s_m, "ci95": s_ci},
                "timesfm_ordered": {"mean": t_m, "ci95": t_ci},
                "timesfm_shuffled": {"mean": sh_m, "ci95": sh_ci}
            },
            "per_seed": all_seed_results
        }, f, indent=2)

    return all_seed_results

if __name__ == "__main__":
    run_clean_regression_benchmark(seeds=[42, 43, 44, 45, 46])
