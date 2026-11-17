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
from torchvision import datasets, transforms
from transformers import TimesFmModelForPrediction
from scipy import stats

# ==============================================================================
# 1. Model Architectures
# ==============================================================================

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

class CompactMNISTMLP(nn.Module):
    def __init__(self, in_features=196, hidden_dim=16, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = nn.functional.adaptive_avg_pool2d(x, (14, 14)).view(x.size(0), -1)
        return self.net(x)

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def eval_regression_mse(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        return criterion(preds, y_tensor).item()

def eval_mnist(model, dataloader, device, criterion):
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

# ==============================================================================
# 2. Low-Rank Trajectory Decomposition & Forecasting Engine
# ==============================================================================

class LowRankTrajectoryForecaster:
    def __init__(self, weight_window, ref_checkpoint=None):
        """
        weight_window: (P, T) numpy array of checkpoints
        ref_checkpoint: (P,) numpy array of reference checkpoint W_ref.
                        Defaults to the final checkpoint W_T (frozen).
        """
        self.P, self.T = weight_window.shape
        if ref_checkpoint is None:
            self.w_ref = weight_window[:, -1].copy()
        else:
            self.w_ref = ref_checkpoint.copy()

        # Displacement matrix D = [W_1 - W_ref, ..., W_T - W_ref]
        t0 = time.time()
        self.D = weight_window - self.w_ref[:, None]
        
        # Truncated SVD
        self.U, self.S, self.Vt = np.linalg.svd(self.D, full_matrices=False)
        self.svd_time = time.time() - t0

        # Variance metrics
        s_sq = self.S ** 2
        total_var = np.sum(s_sq)
        self.explained_variance_ratio = s_sq / (total_var + 1e-12)

    def get_rank_diagnostics(self, rank):
        r = min(rank, self.T, self.P)
        exp_var = float(np.sum(self.explained_variance_ratio[:r]))
        Cr = np.diag(self.S[:r]) @ self.Vt[:r, :] # (r, T)
        D_rec = self.U[:, :r] @ Cr
        norm_D = np.linalg.norm(self.D)
        rec_error = float(np.linalg.norm(self.D - D_rec) / (norm_D + 1e-12))
        return {
            "rank": r,
            "explained_variance": exp_var,
            "reconstruction_error": rec_error
        }

    def forecast_linear(self, rank, horizon, slope_window=16):
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = np.diag(self.S[:r]) @ self.Vt[:r, :] # (r, T)
        k = min(slope_window, self.T - 1)
        slopes = (Cr[:, -1] - Cr[:, -1 - k]) / float(k)
        c_future = Cr[:, -1] + slopes * horizon
        w_rec = self.w_ref + Ur @ c_future
        return torch.tensor(w_rec, dtype=torch.float32)

    def forecast_damped(self, rank, horizon, gamma=0.9, slope_window=16):
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = np.diag(self.S[:r]) @ self.Vt[:r, :]
        k = min(slope_window, self.T - 1)
        slopes = (Cr[:, -1] - Cr[:, -1 - k]) / float(k)
        damping = (1.0 - (gamma ** horizon)) / (1.0 - gamma)
        c_future = Cr[:, -1] + slopes * damping
        w_rec = self.w_ref + Ur @ c_future
        return torch.tensor(w_rec, dtype=torch.float32)

    def forecast_timesfm(self, rank, horizon, tfm_model, device, shuffle=False, seed=42):
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = (np.diag(self.S[:r]) @ self.Vt[:r, :]).copy() # (r, T)
        
        if shuffle:
            rng = np.random.RandomState(seed + 999)
            perm = rng.permutation(self.T)
            Cr = Cr[:, perm]

        t0 = time.time()
        tensor_in = torch.tensor(Cr, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = tfm_model(past_values=tensor_in)
            c_future = out.mean_predictions[:, horizon - 1].cpu().numpy()
        tfm_latency = time.time() - t0

        w_rec = self.w_ref + Ur @ c_future
        return torch.tensor(w_rec, dtype=torch.float32), tfm_latency

# ==============================================================================
# 3. Experiment Part A: California Housing Regression Benchmark (10 Seeds)
# ==============================================================================

def run_low_rank_regression(
    seeds=list(range(42, 52)),
    ranks=[1, 4, 8, 16],
    total_steps=250,
    sat_step=200,
    horizon=48,
    context_len=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" LOW-RANK WEIGHT TRAJECTORY FORECASTING: CALIFORNIA HOUSING REGRESSION")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizon: {horizon} | Cut-off: Step {sat_step}")
    print("=" * 80)

    # 1. Clean Data Splitting (Zero Leakage)
    X_raw, y_raw = fetch_california_housing(return_X_y=True)
    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.50, random_state=42
    )
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

    X_val_scaled = scaler_x.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()

    X_test_scaled = scaler_x.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    criterion = nn.MSELoss()

    # Pre-load TimesFM
    print("Loading TimesFM-2.0-500M...")
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    all_seed_results = []
    explained_vars = {r: [] for r in ranks}
    rec_errors = {r: [] for r in ranks}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        model = RegressionMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

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

        # Baseline: Frozen Checkpoint (Step 200)
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        val_frozen = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_frozen = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Baseline: Continued Adam (48 Steps)
        w_gt = weight_history[sat_step + horizon]
        set_flat_params(model, w_gt.to(device))
        val_gt = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_gt = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Baseline: Matched SWA & EMA over last 64 steps
        start_idx = sat_step - context_len + 1
        w_window = torch.stack(weight_history[start_idx:sat_step + 1], dim=0) # (64, P)
        w_swa = torch.mean(w_window, dim=0)
        set_flat_params(model, w_swa.to(device))
        val_swa = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_swa = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Matched EMA (alpha = 0.05)
        w_ema = w_window[0].clone()
        for t in range(1, context_len):
            w_ema = 0.05 * w_window[t] + 0.95 * w_ema
        set_flat_params(model, w_ema.to(device))
        val_ema = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_ema = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Low-Rank Trajectory Decomposition
        W_win_np = w_window.numpy().T # (P, 64)
        forecaster = LowRankTrajectoryForecaster(W_win_np, ref_checkpoint=w_frozen.numpy())

        seed_data = {
            "seed": seed,
            "baselines": {
                "frozen": {"val_mse": val_frozen, "test_mse": test_frozen},
                "continued_adam": {"val_mse": val_gt, "test_mse": test_gt},
                "swa": {"val_mse": val_swa, "test_mse": test_swa},
                "ema": {"val_mse": val_ema, "test_mse": test_ema}
            },
            "low_rank": {}
        }

        print(f"  Baselines: Frozen={test_frozen:.4f} | Continued Adam={test_gt:.4f} | SWA={test_swa:.4f} | EMA={test_ema:.4f}", flush=True)

        for r in ranks:
            diag = forecaster.get_rank_diagnostics(r)
            explained_vars[r].append(diag["explained_variance"])
            rec_errors[r].append(diag["reconstruction_error"])

            # 1. Linear Extrapolation
            w_lin = forecaster.forecast_linear(r, horizon)
            set_flat_params(model, w_lin.to(device))
            val_lin = eval_regression_mse(model, x_val_t, y_val_t, criterion)
            test_lin = eval_regression_mse(model, x_test_t, y_test_t, criterion)

            # 2. Damped Linear Extrapolation
            w_damped = forecaster.forecast_damped(r, horizon, gamma=0.9)
            set_flat_params(model, w_damped.to(device))
            val_damped = eval_regression_mse(model, x_val_t, y_val_t, criterion)
            test_damped = eval_regression_mse(model, x_test_t, y_test_t, criterion)

            # 3. Low-Rank TimesFM
            w_tfm, tfm_time = forecaster.forecast_timesfm(r, horizon, tfm_model, device, shuffle=False)
            set_flat_params(model, w_tfm.to(device))
            val_tfm = eval_regression_mse(model, x_val_t, y_val_t, criterion)
            test_tfm = eval_regression_mse(model, x_test_t, y_test_t, criterion)

            # 4. Low-Rank TimesFM (Shuffled Control)
            w_shuf, _ = forecaster.forecast_timesfm(r, horizon, tfm_model, device, shuffle=True, seed=seed)
            set_flat_params(model, w_shuf.to(device))
            val_shuf = eval_regression_mse(model, x_val_t, y_val_t, criterion)
            test_shuf = eval_regression_mse(model, x_test_t, y_test_t, criterion)

            seed_data["low_rank"][f"rank_{r}"] = {
                "explained_variance": diag["explained_variance"],
                "reconstruction_error": diag["reconstruction_error"],
                "linear": {"val_mse": val_lin, "test_mse": test_lin},
                "damped": {"val_mse": val_damped, "test_mse": test_damped},
                "timesfm": {"val_mse": val_tfm, "test_mse": test_tfm, "latency_s": tfm_time},
                "timesfm_shuffled": {"val_mse": val_shuf, "test_mse": test_shuf}
            }

            print(f"  Rank {r:2d} (ExpVar={diag['explained_variance']*100:.1f}%): Lin={test_lin:.4f} | Damped={test_damped:.4f} | TFM={test_tfm:.4f} | Shuf={test_shuf:.4f} ({tfm_time*1000:.1f}ms)", flush=True)

        all_seed_results.append(seed_data)

    # Statistical Aggregation
    def calc_ci(vals):
        mean = float(np.mean(vals))
        ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)) if len(vals) > 1 else 0.0
        return mean, ci

    summary = {
        "dataset": "California Housing",
        "num_seeds": len(seeds),
        "horizon": horizon,
        "explained_variance_by_rank": {
            r: {"mean": float(np.mean(explained_vars[r])), "ci95": calc_ci(explained_vars[r])[1]} for r in ranks
        },
        "reconstruction_error_by_rank": {
            r: {"mean": float(np.mean(rec_errors[r])), "ci95": calc_ci(rec_errors[r])[1]} for r in ranks
        },
        "baselines": {
            "frozen": {"mean": calc_ci([s["baselines"]["frozen"]["test_mse"] for s in all_seed_results])[0],
                       "ci95": calc_ci([s["baselines"]["frozen"]["test_mse"] for s in all_seed_results])[1]},
            "continued_adam": {"mean": calc_ci([s["baselines"]["continued_adam"]["test_mse"] for s in all_seed_results])[0],
                               "ci95": calc_ci([s["baselines"]["continued_adam"]["test_mse"] for s in all_seed_results])[1]},
            "swa": {"mean": calc_ci([s["baselines"]["swa"]["test_mse"] for s in all_seed_results])[0],
                    "ci95": calc_ci([s["baselines"]["swa"]["test_mse"] for s in all_seed_results])[1]},
            "ema": {"mean": calc_ci([s["baselines"]["ema"]["test_mse"] for s in all_seed_results])[0],
                    "ci95": calc_ci([s["baselines"]["ema"]["test_mse"] for s in all_seed_results])[1]}
        },
        "low_rank_methods": {}
    }

    for r in ranks:
        summary["low_rank_methods"][f"rank_{r}"] = {
            "linear": {"mean": calc_ci([s["low_rank"][f"rank_{r}"]["linear"]["test_mse"] for s in all_seed_results])[0],
                       "ci95": calc_ci([s["low_rank"][f"rank_{r}"]["linear"]["test_mse"] for s in all_seed_results])[1]},
            "damped": {"mean": calc_ci([s["low_rank"][f"rank_{r}"]["damped"]["test_mse"] for s in all_seed_results])[0],
                       "ci95": calc_ci([s["low_rank"][f"rank_{r}"]["damped"]["test_mse"] for s in all_seed_results])[1]},
            "timesfm": {"mean": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm"]["test_mse"] for s in all_seed_results])[0],
                        "ci95": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm"]["test_mse"] for s in all_seed_results])[1]},
            "timesfm_shuffled": {"mean": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm_shuffled"]["test_mse"] for s in all_seed_results])[0],
                                 "ci95": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm_shuffled"]["test_mse"] for s in all_seed_results])[1]}
        }

    output_payload = {
        "summary": summary,
        "per_seed_runs": all_seed_results
    }
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "low_rank_regression_results.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved {out_file}", flush=True)

    return output_payload

# ==============================================================================
# 4. Experiment Part B: MNIST Multi-Stage Benchmark (10 Seeds)
# ==============================================================================

def run_low_rank_mnist(
    seeds=list(range(42, 52)),
    ranks=[1, 4, 8, 16],
    total_steps=2500,
    sat_step=2000,
    horizon=64,
    context_len=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" LOW-RANK WEIGHT TRAJECTORY FORECASTING: MNIST CLASSIFICATION")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizon: {horizon} | Cut-off: Step {sat_step}")
    print("=" * 80)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    print("Loading TimesFM-2.0-500M...")
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    all_seed_results = []
    explained_vars = {r: [] for r in ranks}
    rec_errors = {r: [] for r in ranks}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Running MNIST Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        model = CompactMNISTMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

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

        # Baseline: Frozen Checkpoint (Step 2000)
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        val_l_f, val_a_f = eval_mnist(model, val_loader, device, criterion)
        test_l_f, test_a_f = eval_mnist(model, test_loader, device, criterion)

        # Baseline: Continued SGD (64 steps)
        w_gt = weight_history[sat_step + horizon]
        set_flat_params(model, w_gt.to(device))
        val_l_gt, val_a_gt = eval_mnist(model, val_loader, device, criterion)
        test_l_gt, test_a_gt = eval_mnist(model, test_loader, device, criterion)

        # Baseline: Matched SWA & EMA over last 64 steps
        start_idx = sat_step - context_len + 1
        w_window = torch.stack(weight_history[start_idx:sat_step + 1], dim=0) # (64, P)
        w_swa = torch.mean(w_window, dim=0)
        set_flat_params(model, w_swa.to(device))
        val_l_swa, val_a_swa = eval_mnist(model, val_loader, device, criterion)
        test_l_swa, test_a_swa = eval_mnist(model, test_loader, device, criterion)

        # Matched EMA (alpha = 0.05)
        w_ema = w_window[0].clone()
        for t in range(1, context_len):
            w_ema = 0.05 * w_window[t] + 0.95 * w_ema
        set_flat_params(model, w_ema.to(device))
        val_l_ema, val_a_ema = eval_mnist(model, val_loader, device, criterion)
        test_l_ema, test_a_ema = eval_mnist(model, test_loader, device, criterion)

        # Low-Rank Trajectory Decomposition
        W_win_np = w_window.numpy().T # (P, 64)
        forecaster = LowRankTrajectoryForecaster(W_win_np, ref_checkpoint=w_frozen.numpy())

        seed_data = {
            "seed": seed,
            "baselines": {
                "frozen": {"val_loss": val_l_f, "test_loss": test_l_f, "test_acc": test_a_f},
                "continued_sgd": {"val_loss": val_l_gt, "test_loss": test_l_gt, "test_acc": test_a_gt},
                "swa": {"val_loss": val_l_swa, "test_loss": test_l_swa, "test_acc": test_a_swa},
                "ema": {"val_loss": val_l_ema, "test_loss": test_l_ema, "test_acc": test_a_ema}
            },
            "low_rank": {}
        }

        print(f"  Baselines: Frozen={test_l_f:.4f} ({test_a_f:.2f}%) | SGD={test_l_gt:.4f} ({test_a_gt:.2f}%) | SWA={test_l_swa:.4f} ({test_a_swa:.2f}%)", flush=True)

        for r in ranks:
            diag = forecaster.get_rank_diagnostics(r)
            explained_vars[r].append(diag["explained_variance"])
            rec_errors[r].append(diag["reconstruction_error"])

            # 1. Linear Extrapolation
            w_lin = forecaster.forecast_linear(r, horizon)
            set_flat_params(model, w_lin.to(device))
            val_l_lin, val_a_lin = eval_mnist(model, val_loader, device, criterion)
            test_l_lin, test_a_lin = eval_mnist(model, test_loader, device, criterion)

            # 2. Damped Linear
            w_damped = forecaster.forecast_damped(r, horizon, gamma=0.9)
            set_flat_params(model, w_damped.to(device))
            val_l_damped, val_a_damped = eval_mnist(model, val_loader, device, criterion)
            test_l_damped, test_a_damped = eval_mnist(model, test_loader, device, criterion)

            # 3. Low-Rank TimesFM
            w_tfm, tfm_time = forecaster.forecast_timesfm(r, horizon, tfm_model, device, shuffle=False)
            set_flat_params(model, w_tfm.to(device))
            val_l_tfm, val_a_tfm = eval_mnist(model, val_loader, device, criterion)
            test_l_tfm, test_a_tfm = eval_mnist(model, test_loader, device, criterion)

            # 4. Low-Rank TimesFM (Shuffled Control)
            w_shuf, _ = forecaster.forecast_timesfm(r, horizon, tfm_model, device, shuffle=True, seed=seed)
            set_flat_params(model, w_shuf.to(device))
            val_l_shuf, val_a_shuf = eval_mnist(model, val_loader, device, criterion)
            test_l_shuf, test_a_shuf = eval_mnist(model, test_loader, device, criterion)

            seed_data["low_rank"][f"rank_{r}"] = {
                "explained_variance": diag["explained_variance"],
                "reconstruction_error": diag["reconstruction_error"],
                "linear": {"val_loss": val_l_lin, "test_loss": test_l_lin, "test_acc": test_a_lin},
                "damped": {"val_loss": val_l_damped, "test_loss": test_l_damped, "test_acc": test_a_damped},
                "timesfm": {"val_loss": val_l_tfm, "test_loss": test_l_tfm, "test_acc": test_a_tfm, "latency_s": tfm_time},
                "timesfm_shuffled": {"val_loss": val_l_shuf, "test_loss": test_l_shuf, "test_acc": test_a_shuf}
            }

            print(f"  Rank {r:2d} (ExpVar={diag['explained_variance']*100:.1f}%): Lin={test_l_lin:.4f} | Damped={test_l_damped:.4f} | TFM={test_l_tfm:.4f} | Shuf={test_l_shuf:.4f} ({tfm_time*1000:.1f}ms)", flush=True)

        all_seed_results.append(seed_data)

    def calc_ci(vals):
        mean = float(np.mean(vals))
        ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)) if len(vals) > 1 else 0.0
        return mean, ci

    summary = {
        "dataset": "MNIST",
        "num_seeds": len(seeds),
        "horizon": horizon,
        "explained_variance_by_rank": {
            r: {"mean": float(np.mean(explained_vars[r])), "ci95": calc_ci(explained_vars[r])[1]} for r in ranks
        },
        "reconstruction_error_by_rank": {
            r: {"mean": float(np.mean(rec_errors[r])), "ci95": calc_ci(rec_errors[r])[1]} for r in ranks
        },
        "baselines": {
            "frozen": {"loss_mean": calc_ci([s["baselines"]["frozen"]["test_loss"] for s in all_seed_results])[0],
                       "loss_ci95": calc_ci([s["baselines"]["frozen"]["test_loss"] for s in all_seed_results])[1]},
            "continued_sgd": {"loss_mean": calc_ci([s["baselines"]["continued_sgd"]["test_loss"] for s in all_seed_results])[0],
                             "loss_ci95": calc_ci([s["baselines"]["continued_sgd"]["test_loss"] for s in all_seed_results])[1]},
            "swa": {"loss_mean": calc_ci([s["baselines"]["swa"]["test_loss"] for s in all_seed_results])[0],
                    "loss_ci95": calc_ci([s["baselines"]["swa"]["test_loss"] for s in all_seed_results])[1]},
            "ema": {"loss_mean": calc_ci([s["baselines"]["ema"]["test_loss"] for s in all_seed_results])[0],
                    "loss_ci95": calc_ci([s["baselines"]["ema"]["test_loss"] for s in all_seed_results])[1]}
        },
        "low_rank_methods": {}
    }

    for r in ranks:
        summary["low_rank_methods"][f"rank_{r}"] = {
            "linear": {"loss_mean": calc_ci([s["low_rank"][f"rank_{r}"]["linear"]["test_loss"] for s in all_seed_results])[0],
                       "loss_ci95": calc_ci([s["low_rank"][f"rank_{r}"]["linear"]["test_loss"] for s in all_seed_results])[1]},
            "damped": {"loss_mean": calc_ci([s["low_rank"][f"rank_{r}"]["damped"]["test_loss"] for s in all_seed_results])[0],
                       "loss_ci95": calc_ci([s["low_rank"][f"rank_{r}"]["damped"]["test_loss"] for s in all_seed_results])[1]},
            "timesfm": {"loss_mean": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm"]["test_loss"] for s in all_seed_results])[0],
                        "loss_ci95": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm"]["test_loss"] for s in all_seed_results])[1]},
            "timesfm_shuffled": {"loss_mean": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm_shuffled"]["test_loss"] for s in all_seed_results])[0],
                                 "loss_ci95": calc_ci([s["low_rank"][f"rank_{r}"]["timesfm_shuffled"]["test_loss"] for s in all_seed_results])[1]}
        }

    output_payload = {
        "summary": summary,
        "per_seed_runs": all_seed_results
    }
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "low_rank_mnist_results.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved {out_file}", flush=True)

    return output_payload

if __name__ == "__main__":
    run_low_rank_regression(seeds=list(range(42, 52)))
    run_low_rank_mnist(seeds=list(range(42, 52)))
