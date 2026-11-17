import time
import numpy as np
import torch

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

        t_rec_start = time.perf_counter()
        delta_w = Ur @ c_pred
        w_pred = self.w_ref + delta_w
        t_reconstruct = time.perf_counter() - t_rec_start

        return torch.from_numpy(w_pred).float(), c_pred, t_forecast, t_reconstruct

    def compute_mechanism_metrics(self, r, c_pred, w_pred, w_swa, w_actual_continuation):
        Cr = self.get_latent_coefficients(r)
        c_mean = np.mean(Cr, axis=1)
        mean_dev = float(np.linalg.norm(c_pred - c_mean) / (np.linalg.norm(c_mean) + 1e-8))

        w_pred_np = w_pred.numpy() if isinstance(w_pred, torch.Tensor) else w_pred
        w_swa_np = w_swa.numpy() if isinstance(w_swa, torch.Tensor) else w_swa

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

        w_act_np = w_actual_continuation.numpy() if isinstance(w_actual_continuation, torch.Tensor) else w_actual_continuation
        delta_act = w_act_np - self.w_ref
        c_act = self.U[:, :r].T @ delta_act

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

# Alias for backwards compatibility
LowRankTrajectoryForecaster = LowRankTrajectoryEngine
