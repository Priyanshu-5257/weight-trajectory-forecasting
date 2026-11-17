import os
import sys
import json
import time
import csv
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
# 1. Models & Helper Functions
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

def eval_mnist_tensors(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor)
        loss = criterion(out, y_tensor).item()
        preds = out.argmax(dim=1)
        acc = 100.0 * (preds == y_tensor).sum().item() / y_tensor.size(0)
    return loss, acc

def calc_stats(vals):
    vals = np.array(vals, dtype=np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = float(stats.sem(vals) * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}

# ==============================================================================
# 2. Low-Rank Trajectory Decomposition & Mechanism Probes
# ==============================================================================

class LowRankTrajectoryEngine:
    def __init__(self, weight_window, ref_checkpoint=None):
        """
        weight_window: (P, T) numpy array of checkpoints
        ref_checkpoint: (P,) numpy array of reference checkpoint W_ref (frozen W_T).
        """
        self.P, self.T = weight_window.shape
        if ref_checkpoint is None:
            self.w_ref = weight_window[:, -1].copy()
        else:
            self.w_ref = ref_checkpoint.copy()

        # Displacement matrix D = [W_1 - W_ref, ..., W_T - W_ref]
        t0 = time.perf_counter()
        self.D = weight_window - self.w_ref[:, None]
        
        # Truncated SVD
        self.U, self.S, self.Vt = np.linalg.svd(self.D, full_matrices=False)
        self.svd_time = time.perf_counter() - t0

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

    def get_latent_coefficients(self, rank):
        r = min(rank, self.T, self.P)
        return np.diag(self.S[:r]) @ self.Vt[:r, :] # (r, T)

    def forecast_linear(self, rank, horizon):
        t0 = time.perf_counter()
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = self.get_latent_coefficients(r) # (r, T)
        
        # Linear slope over recent history
        c_now = Cr[:, -1]
        c_prev = Cr[:, -min(self.T, 16)]
        slope = (c_now - c_prev) / min(self.T - 1, 15)
        c_pred = c_now + slope * horizon # (r,)

        # Reconstruct weights
        delta_w = Ur @ c_pred
        w_pred = self.w_ref + delta_w
        elapsed = time.perf_counter() - t0
        return torch.from_numpy(w_pred).float(), c_pred, elapsed

    def forecast_damped(self, rank, horizon, gamma=0.9):
        t0 = time.perf_counter()
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = self.get_latent_coefficients(r) # (r, T)

        c_now = Cr[:, -1]
        c_prev = Cr[:, -min(self.T, 16)]
        slope = (c_now - c_prev) / min(self.T - 1, 15)
        # Sum of geometric series: gamma + gamma^2 + ... + gamma^h
        damping = gamma * (1.0 - (gamma ** horizon)) / (1.0 - gamma)
        c_pred = c_now + slope * damping

        delta_w = Ur @ c_pred
        w_pred = self.w_ref + delta_w
        elapsed = time.perf_counter() - t0
        return torch.from_numpy(w_pred).float(), c_pred, elapsed

    def forecast_timesfm(self, rank, horizon, tfm_model, device, shuffle=False, seed=42):
        t0 = time.perf_counter()
        r = min(rank, self.T, self.P)
        Ur = self.U[:, :r]
        Cr = self.get_latent_coefficients(r) # (r, T)

        if shuffle:
            rng = np.random.RandomState(seed + rank * 100)
            perm = rng.permutation(self.T)
            Cr_input = Cr[:, perm]
        else:
            Cr_input = Cr

        # Normalize per coordinate
        means = np.mean(Cr_input, axis=1, keepdims=True)
        stds = np.std(Cr_input, axis=1, keepdims=True) + 1e-8
        Cr_norm = (Cr_input - means) / stds

        inputs_t = torch.tensor(Cr_norm, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = tfm_model(inputs_t, horizon_len=horizon)
            mean_preds = outputs.mean_predictions.cpu().numpy() # (r, horizon)

        c_pred_norm = mean_preds[:, -1] # (r,)
        c_pred = c_pred_norm * stds.squeeze(-1) + means.squeeze(-1)

        t_forecast = time.perf_counter() - t0

        # Reconstruction
        t_rec_start = time.perf_counter()
        delta_w = Ur @ c_pred
        w_pred = self.w_ref + delta_w
        t_reconstruct = time.perf_counter() - t_rec_start

        return torch.from_numpy(w_pred).float(), c_pred, t_forecast, t_reconstruct

    def compute_mechanism_metrics(self, r, c_pred, w_pred, w_swa, w_actual_continuation):
        """
        Compute mechanistic indicators:
        1. Historical Mean Deviation: ||c_pred - mean(Cr)|| / ||mean(Cr)||
        2. SWA Alignment: L2 distance and Cosine Similarity to SWA
        3. Continuation Alignment: Cosine Similarity and Relative Error to actual continuation
        """
        Cr = self.get_latent_coefficients(r)
        c_mean = np.mean(Cr, axis=1)
        mean_dev = float(np.linalg.norm(c_pred - c_mean) / (np.linalg.norm(c_mean) + 1e-8))

        w_pred_np = w_pred.numpy() if isinstance(w_pred, torch.Tensor) else w_pred
        w_swa_np = w_swa.numpy() if isinstance(w_swa, torch.Tensor) else w_swa

        # SWA L2 distance & Cosine similarity
        l2_dist_swa = float(np.linalg.norm(w_pred_np - w_swa_np))
        rel_l2_swa = float(l2_dist_swa / (np.linalg.norm(w_swa_np) + 1e-8))

        delta_pred = w_pred_np - self.w_ref
        delta_swa = w_swa_np - self.w_ref
        norm_pred = np.linalg.norm(delta_pred)
        norm_swa = np.linalg.norm(delta_swa)
        if norm_pred > 1e-8 and norm_swa > 1e-8:
            cos_sim_swa = float(np.dot(delta_pred, delta_swa) / (norm_pred * norm_swa))
        else:
            cos_sim_swa = 1.0

        # Actual continuation comparison
        w_act_np = w_actual_continuation.numpy() if isinstance(w_actual_continuation, torch.Tensor) else w_actual_continuation
        delta_act = w_act_np - self.w_ref
        c_act = self.U[:, :r].T @ delta_act # actual latent coordinates

        norm_c_pred = np.linalg.norm(c_pred)
        norm_c_act = np.linalg.norm(c_act)
        if norm_c_pred > 1e-8 and norm_c_act > 1e-8:
            cos_sim_continuation = float(np.dot(c_pred, c_act) / (norm_c_pred * norm_c_act))
        else:
            cos_sim_continuation = 0.0

        rel_error_continuation = float(np.linalg.norm(c_pred - c_act) / (norm_c_act + 1e-8))

        return {
            "mean_deviation": mean_dev,
            "l2_dist_to_swa": l2_dist_swa,
            "rel_l2_to_swa": rel_l2_swa,
            "cos_sim_to_swa": cos_sim_swa,
            "cos_sim_to_continuation": cos_sim_continuation,
            "rel_error_to_continuation": rel_error_continuation
        }

# ==============================================================================
# 3. High-Precision Timing Benchmark Suite
# ==============================================================================

def benchmark_compute_costs(tfm_model, device, P=3322, T_context=64, ranks=[1, 4, 8, 16], horizons=[8, 16, 32]):
    """
    Measure separately:
    - Checkpoint storage overhead (bytes in RAM / simulated disk)
    - SVD decomposition time (median, p95)
    - Coefficient forecasting time per (rank, horizon) (median, p95)
    - Weight reconstruction time (median, p95)
    - Peak GPU memory
    """
    print("\n--- Running Precision Computational Cost Benchmarks ---", flush=True)
    bench_results = {}

    # 1. Storage Overhead
    dummy_weight = np.random.randn(P).astype(np.float32)
    bytes_per_checkpoint = dummy_weight.nbytes
    bench_results["storage"] = {
        "bytes_per_checkpoint": bytes_per_checkpoint,
        "kb_per_checkpoint": bytes_per_checkpoint / 1024.0,
        "context_window_64_kb": (bytes_per_checkpoint * T_context) / 1024.0
    }

    # 2. SVD Decomposition
    dummy_history = np.random.randn(P, T_context).astype(np.float32)
    svd_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        _U, _S, _Vt = np.linalg.svd(dummy_history, full_matrices=False)
        svd_times.append((time.perf_counter() - t0) * 1000.0) # in ms
    bench_results["svd_ms"] = {
        "median": float(np.median(svd_times[5:])),
        "p95": float(np.percentile(svd_times[5:], 95))
    }

    # 3. Weight Reconstruction Time (P x r) @ r
    recon_times = {r: [] for r in ranks}
    for r in ranks:
        Ur = np.random.randn(P, r).astype(np.float32)
        c = np.random.randn(r).astype(np.float32)
        w_ref = np.random.randn(P).astype(np.float32)
        for _ in range(25):
            t0 = time.perf_counter()
            _w = w_ref + Ur @ c
            recon_times[r].append((time.perf_counter() - t0) * 1000.0)
        bench_results[f"reconstruction_ms_r{r}"] = {
            "median": float(np.median(recon_times[r][5:])),
            "p95": float(np.percentile(recon_times[r][5:], 95))
        }

    # 4. TimesFM Latent Forecasting Time per (rank, horizon)
    bench_results["timesfm_forecast_ms"] = {}
    for r in ranks:
        bench_results["timesfm_forecast_ms"][f"rank_{r}"] = {}
        for h in horizons:
            dummy_coords = torch.randn(r, T_context, dtype=torch.float32, device=device)
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = tfm_model(dummy_coords, horizon_len=h)
            torch.cuda.synchronize()

            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = tfm_model(dummy_coords, horizon_len=h)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            
            bench_results["timesfm_forecast_ms"][f"rank_{r}"][f"h_{h}"] = {
                "median": float(np.median(times)),
                "p95": float(np.percentile(times, 95))
            }

    # 5. Peak GPU Memory
    torch.cuda.reset_peak_memory_stats()
    dummy_coords = torch.randn(16, T_context, dtype=torch.float32, device=device)
    with torch.no_grad():
        _ = tfm_model(dummy_coords, horizon_len=32)
    torch.cuda.synchronize()
    bench_results["peak_gpu_memory_mib"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    return bench_results

# ==============================================================================
# 4. California Housing Benchmark Runner
# ==============================================================================

def run_california_feasibility_benchmark(
    seeds=list(range(42, 52)),
    ranks=[1, 4, 8, 16],
    horizons=[8, 16, 32],
    sat_step=200,
    context_len=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("=" * 80)
    print(" CALIFORNIA HOUSING FEASIBILITY BENCHMARK (v1.2)")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizons: {horizons} | Cutoff: Step {sat_step}")
    print("=" * 80, flush=True)

    # 1. Dataset with strict 70/15/15 separation
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    indices = np.arange(len(X))
    idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X[idx_train])
    y_train_scaled = scaler_y.fit_transform(y[idx_train].reshape(-1, 1)).flatten()

    X_val_scaled = scaler_x.transform(X[idx_val])
    y_val_scaled = scaler_y.transform(y[idx_val].reshape(-1, 1)).flatten()

    X_test_scaled = scaler_x.transform(X[idx_test])
    y_test_scaled = scaler_y.transform(y[idx_test].reshape(-1, 1)).flatten()

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
    t_load_start = time.perf_counter()
    print("Loading TimesFM-2.0-500M...", flush=True)
    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()
    t_load = time.perf_counter() - t_load_start
    print(f"TimesFM loaded in {t_load:.2f} s", flush=True)

    max_horizon = max(horizons)
    total_steps = sat_step + max_horizon

    all_seed_data = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Regression] Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        model = RegressionMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

        data_iter = iter(train_loader)
        step_times = []
        for step in range(1, total_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            t0_step = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0_step)

            weight_history.append(get_flat_params(model).cpu())

        # Frozen Checkpoint at sat_step (Step 200)
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        val_frozen = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_frozen = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Baselines: SWA & EMA over context window
        start_idx = sat_step - context_len + 1
        w_window = torch.stack(weight_history[start_idx:sat_step + 1], dim=0) # (64, P)
        w_swa = torch.mean(w_window, dim=0)
        set_flat_params(model, w_swa.to(device))
        val_swa = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_swa = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        w_ema = w_window[0].clone()
        for t in range(1, context_len):
            w_ema = 0.05 * w_window[t] + 0.95 * w_ema
        set_flat_params(model, w_ema.to(device))
        val_ema = eval_regression_mse(model, x_val_t, y_val_t, criterion)
        test_ema = eval_regression_mse(model, x_test_t, y_test_t, criterion)

        # Continued Adam for each horizon
        continued_adam = {}
        for h in horizons:
            w_cont = weight_history[sat_step + h]
            set_flat_params(model, w_cont.to(device))
            val_c = eval_regression_mse(model, x_val_t, y_val_t, criterion)
            test_c = eval_regression_mse(model, x_test_t, y_test_t, criterion)
            cont_time = float(np.sum(step_times[sat_step:sat_step + h]))
            continued_adam[f"h_{h}"] = {
                "val_mse": val_c,
                "test_mse": test_c,
                "train_time_s": cont_time
            }

        # SVD on historical context window
        W_win_np = w_window.numpy().T # (P, 64)
        engine = LowRankTrajectoryEngine(W_win_np, ref_checkpoint=w_frozen.numpy())

        seed_entry = {
            "seed": seed,
            "baselines": {
                "frozen": {"val_mse": val_frozen, "test_mse": test_frozen},
                "swa": {"val_mse": val_swa, "test_mse": test_swa},
                "ema": {"val_mse": val_ema, "test_mse": test_ema},
                "continued_adam": continued_adam
            },
            "svd_time_s": engine.svd_time,
            "rank_diagnostics": {r: engine.get_rank_diagnostics(r) for r in ranks},
            "evaluations": {}
        }

        # Grid of ranks and horizons
        for r in ranks:
            seed_entry["evaluations"][f"rank_{r}"] = {}
            for h in horizons:
                w_act = weight_history[sat_step + h]

                # 1. Linear
                w_lin, c_lin, t_lin = engine.forecast_linear(r, h)
                set_flat_params(model, w_lin.to(device))
                val_lin = eval_regression_mse(model, x_val_t, y_val_t, criterion)
                test_lin = eval_regression_mse(model, x_test_t, y_test_t, criterion)
                mech_lin = engine.compute_mechanism_metrics(r, c_lin, w_lin, w_swa, w_act)

                # 2. Damped Linear
                w_dmp, c_dmp, t_dmp = engine.forecast_damped(r, h, gamma=0.9)
                set_flat_params(model, w_dmp.to(device))
                val_dmp = eval_regression_mse(model, x_val_t, y_val_t, criterion)
                test_dmp = eval_regression_mse(model, x_test_t, y_test_t, criterion)
                mech_dmp = engine.compute_mechanism_metrics(r, c_dmp, w_dmp, w_swa, w_act)

                # 3. TimesFM (Ordered)
                w_tfm, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=False)
                set_flat_params(model, w_tfm.to(device))
                val_tfm = eval_regression_mse(model, x_val_t, y_val_t, criterion)
                test_tfm = eval_regression_mse(model, x_test_t, y_test_t, criterion)
                mech_tfm = engine.compute_mechanism_metrics(r, c_tfm, w_tfm, w_swa, w_act)

                # 4. TimesFM (Shuffled Control)
                w_shf, c_shf, t_shf_f, t_shf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                set_flat_params(model, w_shf.to(device))
                val_shf = eval_regression_mse(model, x_val_t, y_val_t, criterion)
                test_shf = eval_regression_mse(model, x_test_t, y_test_t, criterion)
                mech_shf = engine.compute_mechanism_metrics(r, c_shf, w_shf, w_swa, w_act)

                total_tfm_time = engine.svd_time + t_tfm_f + t_tfm_r

                seed_entry["evaluations"][f"rank_{r}"][f"h_{h}"] = {
                    "linear": {"val_mse": val_lin, "test_mse": test_lin, "latency_s": t_lin, "mech": mech_lin},
                    "damped": {"val_mse": val_dmp, "test_mse": test_dmp, "latency_s": t_dmp, "mech": mech_dmp},
                    "timesfm": {"val_mse": val_tfm, "test_mse": test_tfm, "latency_s": total_tfm_time,
                                "forecast_s": t_tfm_f, "reconstruct_s": t_tfm_r, "mech": mech_tfm},
                    "timesfm_shuffled": {"val_mse": val_shf, "test_mse": test_shf, "latency_s": t_shf_f + t_shf_r, "mech": mech_shf}
                }

        # Select Best Hyperparameters Strictly on Validation Set
        best_val_configs = {}
        for method in ["linear", "damped", "timesfm", "timesfm_shuffled"]:
            best_val = float("inf")
            best_cfg = None
            for r in ranks:
                for h in horizons:
                    val_metric = seed_entry["evaluations"][f"rank_{r}"][f"h_{h}"][method]["val_mse"]
                    if val_metric < best_val:
                        best_val = val_metric
                        best_cfg = (r, h)
            best_r, best_h = best_cfg
            chosen_eval = seed_entry["evaluations"][f"rank_{best_r}"][f"h_{best_h}"][method]
            best_val_configs[method] = {
                "selected_rank": best_r,
                "selected_horizon": best_h,
                "val_mse": chosen_eval["val_mse"],
                "test_mse": chosen_eval["test_mse"],
                "latency_s": chosen_eval["latency_s"],
                "mech": chosen_eval["mech"]
            }

        seed_entry["validation_selected"] = best_val_configs
        all_seed_data.append(seed_entry)

        print(f"  Frozen: Val={val_frozen:.4f}, Test={test_frozen:.4f} | SWA: Val={val_swa:.4f}, Test={test_swa:.4f}", flush=True)
        print(f"  Selected TimesFM (r={best_val_configs['timesfm']['selected_rank']}, h={best_val_configs['timesfm']['selected_horizon']}): Val={best_val_configs['timesfm']['val_mse']:.4f}, Test={best_val_configs['timesfm']['test_mse']:.4f}", flush=True)

    return all_seed_data, tfm_model

# ==============================================================================
# 5. MNIST Feasibility Benchmark Runner
# ==============================================================================

def run_mnist_feasibility_benchmark(
    tfm_model,
    seeds=list(range(42, 52)),
    ranks=[1, 4, 8, 16],
    horizons=[8, 16, 32],
    sat_step=2000,
    context_len=64,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("\n" + "=" * 80)
    print(" MNIST IMAGE CLASSIFICATION FEASIBILITY BENCHMARK (v1.2)")
    print(f" Seeds: {seeds} | Ranks: {ranks} | Horizons: {horizons} | Cutoff: Step {sat_step}")
    print("=" * 80, flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    val_loader_init = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
    for x_v, y_v in val_loader_init:
        x_val_t = x_v.to(device)
        y_val_t = y_v.to(device)
        break

    test_loader_init = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    for x_t, y_t in test_loader_init:
        x_test_t = x_t.to(device)
        y_test_t = y_t.to(device)
        break

    criterion = nn.CrossEntropyLoss()

    max_horizon = max(horizons)
    total_steps = sat_step + max_horizon

    all_seed_data = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [MNIST] Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        model = CompactMNISTMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

        data_iter = iter(train_loader)
        step_times = []
        for step in range(1, total_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            t0_step = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0_step)

            weight_history.append(get_flat_params(model).cpu())

        # Frozen Checkpoint at Step sat_step (Step 2000)
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        val_frozen_loss, val_frozen_acc = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
        test_frozen_loss, test_frozen_acc = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)

        # Baselines: SWA & EMA
        start_idx = sat_step - context_len + 1
        w_window = torch.stack(weight_history[start_idx:sat_step + 1], dim=0) # (64, P)
        w_swa = torch.mean(w_window, dim=0)
        set_flat_params(model, w_swa.to(device))
        val_swa_loss, val_swa_acc = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
        test_swa_loss, test_swa_acc = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)

        w_ema = w_window[0].clone()
        for t in range(1, context_len):
            w_ema = 0.05 * w_window[t] + 0.95 * w_ema
        set_flat_params(model, w_ema.to(device))
        val_ema_loss, val_ema_acc = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
        test_ema_loss, test_ema_acc = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)

        # Continued SGD for each horizon
        continued_sgd = {}
        for h in horizons:
            w_cont = weight_history[sat_step + h]
            set_flat_params(model, w_cont.to(device))
            val_c_loss, val_c_acc = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
            test_c_loss, test_c_acc = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)
            cont_time = float(np.sum(step_times[sat_step:sat_step + h]))
            continued_sgd[f"h_{h}"] = {
                "val_loss": val_c_loss, "val_acc": val_c_acc,
                "test_loss": test_c_loss, "test_acc": test_c_acc,
                "train_time_s": cont_time
            }

        # SVD on historical context window
        W_win_np = w_window.numpy().T
        engine = LowRankTrajectoryEngine(W_win_np, ref_checkpoint=w_frozen.numpy())

        seed_entry = {
            "seed": seed,
            "baselines": {
                "frozen": {"val_loss": val_frozen_loss, "val_acc": val_frozen_acc,
                           "test_loss": test_frozen_loss, "test_acc": test_frozen_acc},
                "swa": {"val_loss": val_swa_loss, "val_acc": val_swa_acc,
                        "test_loss": test_swa_loss, "test_acc": test_swa_acc},
                "ema": {"val_loss": val_ema_loss, "val_acc": val_ema_acc,
                        "test_loss": test_ema_loss, "test_acc": test_ema_acc},
                "continued_sgd": continued_sgd
            },
            "svd_time_s": engine.svd_time,
            "rank_diagnostics": {r: engine.get_rank_diagnostics(r) for r in ranks},
            "evaluations": {}
        }

        for r in ranks:
            seed_entry["evaluations"][f"rank_{r}"] = {}
            for h in horizons:
                w_act = weight_history[sat_step + h]

                # 1. Linear
                w_lin, c_lin, t_lin = engine.forecast_linear(r, h)
                set_flat_params(model, w_lin.to(device))
                val_lin_l, val_lin_a = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
                test_lin_l, test_lin_a = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)
                mech_lin = engine.compute_mechanism_metrics(r, c_lin, w_lin, w_swa, w_act)

                # 2. Damped Linear
                w_dmp, c_dmp, t_dmp = engine.forecast_damped(r, h, gamma=0.9)
                set_flat_params(model, w_dmp.to(device))
                val_dmp_l, val_dmp_a = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
                test_dmp_l, test_dmp_a = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)
                mech_dmp = engine.compute_mechanism_metrics(r, c_dmp, w_dmp, w_swa, w_act)

                # 3. TimesFM (Ordered)
                w_tfm, c_tfm, t_tfm_f, t_tfm_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=False)
                set_flat_params(model, w_tfm.to(device))
                val_tfm_l, val_tfm_a = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
                test_tfm_l, test_tfm_a = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)
                mech_tfm = engine.compute_mechanism_metrics(r, c_tfm, w_tfm, w_swa, w_act)

                # 4. TimesFM (Shuffled Control)
                w_shf, c_shf, t_shf_f, t_shf_r = engine.forecast_timesfm(r, h, tfm_model, device, shuffle=True, seed=seed)
                set_flat_params(model, w_shf.to(device))
                val_shf_l, val_shf_a = eval_mnist_tensors(model, x_val_t, y_val_t, criterion)
                test_shf_l, test_shf_a = eval_mnist_tensors(model, x_test_t, y_test_t, criterion)
                mech_shf = engine.compute_mechanism_metrics(r, c_shf, w_shf, w_swa, w_act)

                total_tfm_time = engine.svd_time + t_tfm_f + t_tfm_r

                seed_entry["evaluations"][f"rank_{r}"][f"h_{h}"] = {
                    "linear": {"val_loss": val_lin_l, "val_acc": val_lin_a, "test_loss": test_lin_l, "test_acc": test_lin_a, "latency_s": t_lin, "mech": mech_lin},
                    "damped": {"val_loss": val_dmp_l, "val_acc": val_dmp_a, "test_loss": test_dmp_l, "test_acc": test_dmp_a, "latency_s": t_dmp, "mech": mech_dmp},
                    "timesfm": {"val_loss": val_tfm_l, "val_acc": val_tfm_a, "test_loss": test_tfm_l, "test_acc": test_tfm_a, "latency_s": total_tfm_time,
                                "forecast_s": t_tfm_f, "reconstruct_s": t_tfm_r, "mech": mech_tfm},
                    "timesfm_shuffled": {"val_loss": val_shf_l, "val_acc": val_shf_a, "test_loss": test_shf_l, "test_acc": test_shf_a, "latency_s": t_shf_f + t_shf_r, "mech": mech_shf}
                }

        # Select Best Hyperparameters Strictly on Validation Set
        best_val_configs = {}
        for method in ["linear", "damped", "timesfm", "timesfm_shuffled"]:
            best_val = float("inf")
            best_cfg = None
            for r in ranks:
                for h in horizons:
                    val_metric = seed_entry["evaluations"][f"rank_{r}"][f"h_{h}"][method]["val_loss"]
                    if val_metric < best_val:
                        best_val = val_metric
                        best_cfg = (r, h)
            best_r, best_h = best_cfg
            chosen_eval = seed_entry["evaluations"][f"rank_{best_r}"][f"h_{best_h}"][method]
            best_val_configs[method] = {
                "selected_rank": best_r,
                "selected_horizon": best_h,
                "val_loss": chosen_eval["val_loss"],
                "val_acc": chosen_eval["val_acc"],
                "test_loss": chosen_eval["test_loss"],
                "test_acc": chosen_eval["test_acc"],
                "latency_s": chosen_eval["latency_s"],
                "mech": chosen_eval["mech"]
            }

        seed_entry["validation_selected"] = best_val_configs
        all_seed_data.append(seed_entry)

        print(f"  Frozen: Val={val_frozen_loss:.4f}, Test={test_frozen_loss:.4f} | SWA: Val={val_swa_loss:.4f}, Test={test_swa_loss:.4f}", flush=True)
        print(f"  Selected TimesFM (r={best_val_configs['timesfm']['selected_rank']}, h={best_val_configs['timesfm']['selected_horizon']}): Val={best_val_configs['timesfm']['val_loss']:.4f}, Test={best_val_configs['timesfm']['test_loss']:.4f}", flush=True)

    return all_seed_data

# ==============================================================================
# 6. Aggregation & Export Suite
# ==============================================================================

def aggregate_and_export_results(reg_data, mnist_data, compute_bench, ranks, horizons, filename_prefix="feasibility_benchmark"):
    print("\n--- Aggregating Benchmark Results & Statistical CIs ---", flush=True)
    summary = {
        "metadata": {
            "version": "1.2",
            "num_seeds": len(reg_data),
            "ranks": ranks,
            "horizons": horizons,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "compute_benchmarks": compute_bench,
        "california_housing": {},
        "mnist": {}
    }

    # Helper for stats
    def summarize_dataset(dataset_runs, is_classification=False):
        metric_key = "loss" if is_classification else "mse"
        res = {
            "baselines": {
                "frozen": {
                    f"val_{metric_key}": calc_stats([s["baselines"]["frozen"][f"val_{metric_key}"] for s in dataset_runs]),
                    f"test_{metric_key}": calc_stats([s["baselines"]["frozen"][f"test_{metric_key}"] for s in dataset_runs]),
                },
                "swa": {
                    f"val_{metric_key}": calc_stats([s["baselines"]["swa"][f"val_{metric_key}"] for s in dataset_runs]),
                    f"test_{metric_key}": calc_stats([s["baselines"]["swa"][f"test_{metric_key}"] for s in dataset_runs]),
                },
                "ema": {
                    f"val_{metric_key}": calc_stats([s["baselines"]["ema"][f"val_{metric_key}"] for s in dataset_runs]),
                    f"test_{metric_key}": calc_stats([s["baselines"]["ema"][f"test_{metric_key}"] for s in dataset_runs]),
                },
                "continued": {}
            },
            "validation_selected_methods": {},
            "grid_evaluations": {},
            "mechanistic_averages": {},
            "explained_variance_by_rank": {
                r: calc_stats([s["rank_diagnostics"][r]["explained_variance"] for s in dataset_runs]) for r in ranks
            },
            "reconstruction_error_by_rank": {
                r: calc_stats([s["rank_diagnostics"][r]["reconstruction_error"] for s in dataset_runs]) for r in ranks
            }
        }

        cont_key = "continued_sgd" if is_classification else "continued_adam"
        for h in horizons:
            res["baselines"]["continued"][f"h_{h}"] = {
                f"val_{metric_key}": calc_stats([s["baselines"][cont_key][f"h_{h}"][f"val_{metric_key}"] for s in dataset_runs]),
                f"test_{metric_key}": calc_stats([s["baselines"][cont_key][f"h_{h}"][f"test_{metric_key}"] for s in dataset_runs]),
                "train_time_s": calc_stats([s["baselines"][cont_key][f"h_{h}"]["train_time_s"] for s in dataset_runs])
            }

        # Validation-Selected Methods
        for method in ["linear", "damped", "timesfm", "timesfm_shuffled"]:
            res["validation_selected_methods"][method] = {
                f"val_{metric_key}": calc_stats([s["validation_selected"][method][f"val_{metric_key}"] for s in dataset_runs]),
                f"test_{metric_key}": calc_stats([s["validation_selected"][method][f"test_{metric_key}"] for s in dataset_runs]),
                "latency_s": calc_stats([s["validation_selected"][method]["latency_s"] for s in dataset_runs]),
                "rank_counts": {r: sum(1 for s in dataset_runs if s["validation_selected"][method]["selected_rank"] == r) for r in ranks},
                "horizon_counts": {h: sum(1 for s in dataset_runs if s["validation_selected"][method]["selected_horizon"] == h) for h in horizons},
                "mean_deviation": calc_stats([s["validation_selected"][method]["mech"]["mean_deviation"] for s in dataset_runs]),
                "cos_sim_to_swa": calc_stats([s["validation_selected"][method]["mech"]["cos_sim_to_swa"] for s in dataset_runs]),
                "cos_sim_to_continuation": calc_stats([s["validation_selected"][method]["mech"]["cos_sim_to_continuation"] for s in dataset_runs]),
                "rel_error_to_continuation": calc_stats([s["validation_selected"][method]["mech"]["rel_error_to_continuation"] for s in dataset_runs])
            }

        # Grid averages
        for r in ranks:
            res["grid_evaluations"][f"rank_{r}"] = {}
            for h in horizons:
                res["grid_evaluations"][f"rank_{r}"][f"h_{h}"] = {}
                for method in ["linear", "damped", "timesfm", "timesfm_shuffled"]:
                    res["grid_evaluations"][f"rank_{r}"][f"h_{h}"][method] = {
                        f"val_{metric_key}": calc_stats([s["evaluations"][f"rank_{r}"][f"h_{h}"][method][f"val_{metric_key}"] for s in dataset_runs]),
                        f"test_{metric_key}": calc_stats([s["evaluations"][f"rank_{r}"][f"h_{h}"][method][f"test_{metric_key}"] for s in dataset_runs]),
                        "latency_s": calc_stats([s["evaluations"][f"rank_{r}"][f"h_{h}"][method]["latency_s"] for s in dataset_runs]),
                        "cos_sim_to_swa": calc_stats([s["evaluations"][f"rank_{r}"][f"h_{h}"][method]["mech"]["cos_sim_to_swa"] for s in dataset_runs]),
                        "cos_sim_to_continuation": calc_stats([s["evaluations"][f"rank_{r}"][f"h_{h}"][method]["mech"]["cos_sim_to_continuation"] for s in dataset_runs])
                    }

        return res

    summary["california_housing"] = summarize_dataset(reg_data, is_classification=False)
    summary["mnist"] = summarize_dataset(mnist_data, is_classification=True)

    # Export full JSON
    full_payload = {
        "summary": summary,
        "california_raw_runs": reg_data,
        "mnist_raw_runs": mnist_data
    }

    with open(f"{filename_prefix}_results.json", "w") as f:
        json.dump(full_payload, f, indent=2)
    print(f"Saved {filename_prefix}_results.json")

    with open(f"{filename_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {filename_prefix}_summary.json")

    # Export CSV for tabular audit
    csv_rows = []
    for s in reg_data:
        seed = s["seed"]
        row = {
            "dataset": "California_Housing",
            "seed": seed,
            "metric_type": "MSE",
            "frozen_val": s["baselines"]["frozen"]["val_mse"],
            "frozen_test": s["baselines"]["frozen"]["test_mse"],
            "swa_val": s["baselines"]["swa"]["val_mse"],
            "swa_test": s["baselines"]["swa"]["test_mse"],
            "ema_val": s["baselines"]["ema"]["val_mse"],
            "ema_test": s["baselines"]["ema"]["test_mse"],
            "cont_h32_val": s["baselines"]["continued_adam"]["h_32"]["val_mse"],
            "cont_h32_test": s["baselines"]["continued_adam"]["h_32"]["test_mse"],
            "cont_h32_time_s": s["baselines"]["continued_adam"]["h_32"]["train_time_s"],
            "tfm_selected_rank": s["validation_selected"]["timesfm"]["selected_rank"],
            "tfm_selected_h": s["validation_selected"]["timesfm"]["selected_horizon"],
            "tfm_val": s["validation_selected"]["timesfm"]["val_mse"],
            "tfm_test": s["validation_selected"]["timesfm"]["test_mse"],
            "tfm_latency_s": s["validation_selected"]["timesfm"]["latency_s"],
            "tfm_cos_swa": s["validation_selected"]["timesfm"]["mech"]["cos_sim_to_swa"],
            "tfm_cos_cont": s["validation_selected"]["timesfm"]["mech"]["cos_sim_to_continuation"],
            "tfm_rel_err_cont": s["validation_selected"]["timesfm"]["mech"]["rel_error_to_continuation"],
            "tfm_shuf_test": s["validation_selected"]["timesfm_shuffled"]["test_mse"],
            "linear_test": s["validation_selected"]["linear"]["test_mse"],
            "damped_test": s["validation_selected"]["damped"]["test_mse"],
        }
        csv_rows.append(row)

    for s in mnist_data:
        seed = s["seed"]
        row = {
            "dataset": "MNIST",
            "seed": seed,
            "metric_type": "CrossEntropyLoss",
            "frozen_val": s["baselines"]["frozen"]["val_loss"],
            "frozen_test": s["baselines"]["frozen"]["test_loss"],
            "swa_val": s["baselines"]["swa"]["val_loss"],
            "swa_test": s["baselines"]["swa"]["test_loss"],
            "ema_val": s["baselines"]["ema"]["val_loss"],
            "ema_test": s["baselines"]["ema"]["test_loss"],
            "cont_h32_val": s["baselines"]["continued_sgd"]["h_32"]["val_loss"],
            "cont_h32_test": s["baselines"]["continued_sgd"]["h_32"]["test_loss"],
            "cont_h32_time_s": s["baselines"]["continued_sgd"]["h_32"]["train_time_s"],
            "tfm_selected_rank": s["validation_selected"]["timesfm"]["selected_rank"],
            "tfm_selected_h": s["validation_selected"]["timesfm"]["selected_horizon"],
            "tfm_val": s["validation_selected"]["timesfm"]["val_loss"],
            "tfm_test": s["validation_selected"]["timesfm"]["test_loss"],
            "tfm_latency_s": s["validation_selected"]["timesfm"]["latency_s"],
            "tfm_cos_swa": s["validation_selected"]["timesfm"]["mech"]["cos_sim_to_swa"],
            "tfm_cos_cont": s["validation_selected"]["timesfm"]["mech"]["cos_sim_to_continuation"],
            "tfm_rel_err_cont": s["validation_selected"]["timesfm"]["mech"]["rel_error_to_continuation"],
            "tfm_shuf_test": s["validation_selected"]["timesfm_shuffled"]["test_loss"],
            "linear_test": s["validation_selected"]["linear"]["test_loss"],
            "damped_test": s["validation_selected"]["damped"]["test_loss"],
        }
        csv_rows.append(row)

    fieldnames = list(csv_rows[0].keys())
    with open(f"{filename_prefix}_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"Saved {filename_prefix}_per_seed.csv")

    return summary

# ==============================================================================
# 7. Main Entry Point
# ==============================================================================

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(42, 52))
    ranks = [1, 4, 8, 16]
    horizons = [8, 16, 32]

    print("Starting Low-Rank Forecasting Feasibility Benchmark (v1.2)...", flush=True)

    # 1. Run California Housing
    reg_runs, tfm_model = run_california_feasibility_benchmark(
        seeds=seeds, ranks=ranks, horizons=horizons, sat_step=200, context_len=64, device_str=device_str
    )

    # 2. Run MNIST
    mnist_runs = run_mnist_feasibility_benchmark(
        tfm_model=tfm_model, seeds=seeds, ranks=ranks, horizons=horizons, sat_step=2000, context_len=64, device_str=device_str
    )

    # 3. Precision Computational Cost Benchmarking
    compute_bench = benchmark_compute_costs(
        tfm_model=tfm_model, device=torch.device(device_str), P=3322, T_context=64, ranks=ranks, horizons=horizons
    )

    # 4. Aggregation and Export
    aggregate_and_export_results(reg_runs, mnist_runs, compute_bench, ranks=ranks, horizons=horizons)
    print("\nBenchmark v1.2 execution complete!", flush=True)

if __name__ == "__main__":
    main()
