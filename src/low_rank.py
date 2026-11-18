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
        t0_prep = time.perf_counter()
        self.D = weight_window - self.w_ref[:, None]
        self.prep_time = time.perf_counter() - t0_prep
        
        # Truncated SVD
        t0_svd = time.perf_counter()
        self.U, self.S, self.Vt = np.linalg.svd(self.D, full_matrices=False)
        self.svd_time = time.perf_counter() - t0_svd

        s_sq = self.S ** 2
        total_var = np.sum(s_sq)
        self.explained_variance_ratio = s_sq / (total_var + 1e-12)

        # Detect effective numerical rank (filter singular values below floating-point machine precision)
        eps = float(np.finfo(np.float32).eps)
        tol = float(self.S[0] * max(self.D.shape) * eps) if len(self.S) > 0 else 1e-12
        self.effective_rank = int(np.sum(self.S > tol))

    def get_rank_diagnostics(self, rank):
        r = min(rank, self.T, self.P)
        exp_var = float(np.sum(self.explained_variance_ratio[:r]))
        Cr = np.diag(self.S[:r]) @ self.Vt[:r, :] # (r, T)
        D_rec = self.U[:, :r] @ Cr
        norm_D = np.linalg.norm(self.D)
        rec_error = float(np.linalg.norm(self.D - D_rec) / (norm_D + 1e-12))
        return {
            "rank": r,
            "effective_rank": self.effective_rank,
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
            outputs = tfm_model(inputs_t)
            mean_preds = outputs.mean_predictions.cpu().numpy() # (r, 128)

        # Index at requested horizon (1-indexed horizon h -> 0-indexed column h-1)
        h_idx = min(max(0, horizon - 1), mean_preds.shape[1] - 1)
        c_pred_norm = mean_preds[:, h_idx] # (r,)
        c_pred = c_pred_norm * stds.squeeze(-1) + means.squeeze(-1)

        t_forecast = time.perf_counter() - t0

        t_rec_start = time.perf_counter()
        delta_w = Ur @ c_pred
        w_pred = self.w_ref + delta_w
        t_reconstruct = time.perf_counter() - t_rec_start

        return torch.from_numpy(w_pred).float(), c_pred, t_forecast, t_reconstruct

    def forecast_weight_decay(self, horizon, eta=1e-3, weight_decay=1.0, interval=25):
        t0 = time.perf_counter()
        total_steps = horizon * interval
        decay_factor = (1.0 - eta * weight_decay) ** total_steps
        w_pred = self.w_ref * decay_factor
        delta_w = w_pred - self.w_ref
        c_pred = self.U.T @ delta_w
        elapsed = time.perf_counter() - t0
        return torch.from_numpy(w_pred).float(), c_pred, elapsed

    def compute_mechanism_metrics(self, r, c_pred, w_pred, w_swa, w_actual_continuation):
        Cr = self.get_latent_coefficients(r)
        c_mean = np.mean(Cr, axis=1)
        mean_dev = float(np.linalg.norm(c_pred - c_mean) / (np.linalg.norm(c_mean) + 1e-8))

        w_pred_np = w_pred.numpy() if isinstance(w_pred, torch.Tensor) else w_pred
        w_swa_np = w_swa.numpy() if isinstance(w_swa, torch.Tensor) else w_swa
        w_act_np = w_actual_continuation.numpy() if isinstance(w_actual_continuation, torch.Tensor) else w_actual_continuation

        # SWA L2 distance & Cosine similarity (Full Parameter Space)
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

        # FULL-PARAMETER SPACE ALIGNMENT TO ACTUAL CONTINUATION
        delta_act = w_act_np - self.w_ref
        norm_act = np.linalg.norm(delta_act)
        if norm_pred > 1e-8 and norm_act > 1e-8:
            cos_sim_full = float(np.dot(delta_pred, delta_act) / (norm_pred * norm_act))
        else:
            cos_sim_full = 0.0

        rel_l2_full = float(np.linalg.norm(w_pred_np - w_act_np) / (norm_act + 1e-8))

        # LATENT SUBSPACE ALIGNMENT
        c_act = self.U[:, :r].T @ delta_act
        norm_c_pred = np.linalg.norm(c_pred)
        norm_c_act = np.linalg.norm(c_act)
        if norm_c_pred > 1e-8 and norm_c_act > 1e-8:
            cos_sim_latent = float(np.dot(c_pred, c_act) / (norm_c_pred * norm_c_act))
        else:
            cos_sim_latent = 0.0

        rel_l2_latent = float(np.linalg.norm(c_pred - c_act) / (norm_c_act + 1e-8))

        # RADIAL WEIGHT DECAY ALIGNMENT
        v_radial = -self.w_ref / (np.linalg.norm(self.w_ref) + 1e-8)
        radial_alignment_act = float(np.dot(delta_act, v_radial) / (norm_act + 1e-8))
        radial_alignment_pred = float(np.dot(delta_pred, v_radial) / (norm_pred + 1e-8))

        return {
            "mean_deviation": mean_dev,
            "l2_dist_to_swa": l2_dist_swa,
            "rel_l2_to_swa": rel_l2_swa,
            "cos_sim_to_swa": cos_sim_swa,
            "cos_sim_to_continuation": cos_sim_full, # FULL PARAMETER (replaces coordinate-only cos)
            "rel_error_to_continuation": rel_l2_full, # FULL PARAMETER
            "cos_sim_latent": cos_sim_latent,
            "rel_l2_latent": rel_l2_latent,
            "radial_alignment_act": radial_alignment_act,
            "radial_alignment_pred": radial_alignment_pred
        }

# Alias for backwards compatibility
LowRankTrajectoryForecaster = LowRankTrajectoryEngine
