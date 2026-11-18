import torch
import numpy as np

def get_layer_parameter_indices(model):
    """
    Partitions model parameters into disjoint layer groups:
    'embed': token embeddings and positional embeddings
    'attn': multi-head self-attention projections (Q, K, V, Out)
    'mlp': feedforward layers (mlp_in, mlp_out)
    'unembed': unembedding head
    Returns:
        dict[str, list[int]]: dictionary mapping group name to flat parameter index list.
    """
    layer_indices = {
        "embed": [],
        "attn": [],
        "mlp": [],
        "unembed": []
    }
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        indices = list(range(offset, offset + numel))
        if "unembed" in name:
            layer_indices["unembed"].extend(indices)
        elif "embed" in name or "pos_embed" in name:
            layer_indices["embed"].extend(indices)
        elif any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
            layer_indices["attn"].extend(indices)
        elif "mlp" in name:
            layer_indices["mlp"].extend(indices)
        else:
            raise ValueError(f"Unrecognized parameter: {name}")
        offset += numel

    total_params = sum(len(v) for v in layer_indices.values())
    expected_total = sum(p.numel() for p in model.parameters())
    assert total_params == expected_total, f"Parameter count mismatch: {total_params} vs {expected_total}"
    return layer_indices

def compute_svd_basis(
    D: torch.Tensor,
    rank: int = None,
    return_effective_rank: bool = False,
    return_diagnostics: bool = False,
    tol: float = None
):
    """
    Computes thin SVD of displacement matrix D: (P, T)
    D = U S V^T in float64 for high numerical stability.
    Automatically filters out zero-displacement columns (e.g. W_ref - W_ref = 0).

    Args:
        D: (P, T) displacement matrix
        rank: requested subspace dimension (if None, returns full valid rank)
        return_effective_rank: whether to return effective rank (for backward compatibility)
        return_diagnostics: whether to return comprehensive rank and spectrum diagnostics
        tol: optional custom tolerance for numerical rank

    Returns:
        U: (P, actual_rank) orthonormal basis in float32
        S: singular values (T_valid,) in float64
        cum_var: cumulative explained variance ratio (T_valid,) in float64
        diagnostics / r_eff: (optional)
    """
    col_norms = torch.norm(D.double(), dim=0)
    valid_cols = col_norms > 1e-12
    if torch.sum(valid_cols) > 0:
        D_valid = D[:, valid_cols].double()
    else:
        D_valid = D.double()

    # Float64 SVD for maximum numerical precision
    U_64, S_64, V_64 = torch.linalg.svd(D_valid, full_matrices=False)

    P_dim, T_valid = D_valid.shape
    max_possible_rank = min(P_dim, T_valid)

    # Condition number
    cond_num = float((S_64[0] / (S_64[-1] + 1e-15)).item()) if len(S_64) > 0 else 1.0

    # Numerical ranks under explicitly stated tolerances
    eps_64 = torch.finfo(torch.float64).eps
    eps_32 = torch.finfo(torch.float32).eps
    tol_m64 = (S_64[0] * max(D_valid.shape) * eps_64).item() if len(S_64) > 0 else 1e-15
    tol_m32 = (S_64[0] * max(D_valid.shape) * eps_32).item() if len(S_64) > 0 else 1e-7
    tol_r1e3 = (S_64[0] * 1e-3).item() if len(S_64) > 0 else 1e-3
    tol_a1e6 = 1e-6

    if tol is not None:
        r_eff_custom = int(torch.sum(S_64 > tol).item())
    else:
        r_eff_custom = int(torch.sum(S_64 > tol_m64).item())

    r_mach_64 = int(torch.sum(S_64 > tol_m64).item())
    r_mach_32 = int(torch.sum(S_64 > tol_m32).item())
    r_rel_1e3 = int(torch.sum(S_64 > tol_r1e3).item())
    r_abs_1e6 = int(torch.sum(S_64 > tol_a1e6).item())

    var = S_64**2
    total_var = torch.sum(var) + 1e-15
    cum_var = torch.cumsum(var, dim=0) / total_var

    # Energy ranks at 99%, 99.9%, 99.99% explained variance
    r_energy_99 = int((cum_var >= 0.990).nonzero()[0].item()) + 1 if torch.any(cum_var >= 0.990) else len(S_64)
    r_energy_999 = int((cum_var >= 0.999).nonzero()[0].item()) + 1 if torch.any(cum_var >= 0.999) else len(S_64)
    r_energy_9999 = int((cum_var >= 0.9999).nonzero()[0].item()) + 1 if torch.any(cum_var >= 0.9999) else len(S_64)

    # Basis selection: never silently truncate requested rank to a low energy threshold
    # The actual basis dimension is bounded only by the physical number of valid columns
    if rank is not None:
        actual_rank = min(rank, max_possible_rank)
    else:
        actual_rank = max_possible_rank

    U = U_64[:, :actual_rank].float()

    diagnostics = {
        "max_rank": max_possible_rank,
        "actual_rank": actual_rank,
        "requested_rank": rank if rank is not None else max_possible_rank,
        "condition_number": cond_num,
        "r_mach_64": r_mach_64,
        "r_mach_32": r_mach_32,
        "r_rel_1e3": r_rel_1e3,
        "r_abs_1e6": r_abs_1e6,
        "r_energy_99": r_energy_99,
        "r_energy_999": r_energy_999,
        "r_energy_9999": r_energy_9999,
        "singular_values": S_64.tolist(),
        "cum_var": cum_var.tolist()
    }

    if return_diagnostics:
        return U, S_64, cum_var, diagnostics
    if return_effective_rank:
        return U, S_64, cum_var, r_eff_custom
    return U, S_64, cum_var

def qr_project_displacement(D: torch.Tensor, dw: torch.Tensor):
    """
    Independent float64 QR decomposition and projection.
    Projects dw onto the column span of D using QR factorization:
    Q, R = qr(D_valid)
    dw_parallel = Q (Q^T dw)
    dw_perp = dw - dw_parallel
    """
    col_norms = torch.norm(D.double(), dim=0)
    valid_cols = col_norms > 1e-12
    if torch.sum(valid_cols) > 0:
        D_valid = D[:, valid_cols].double()
    else:
        D_valid = D.double()

    Q, R = torch.linalg.qr(D_valid)
    dw_64 = dw.double()
    coeffs = torch.matmul(Q.T, dw_64)
    dw_par_64 = torch.matmul(Q, coeffs)
    dw_perp_64 = dw_64 - dw_par_64

    norm_tot_sq = torch.sum(dw_64**2) + 1e-15
    norm_par_sq = torch.sum(dw_par_64**2)
    cov = (norm_par_sq / norm_tot_sq).item()
    return dw_par_64.float(), dw_perp_64.float(), float(min(max(cov, 0.0), 1.0))

def compute_radial_decay_alignment(U, w_ref):
    """
    Computes cosine alignment between the first SVD direction u1 and the radial decay direction.
    v_decay = -w_ref / ||w_ref||
    alignment = |<u1, v_decay>|
    """
    norm_w = torch.norm(w_ref.float()) + 1e-12
    v_decay = -w_ref.float() / norm_w
    u1 = U[:, 0].float()
    u1_norm = torch.norm(u1) + 1e-12
    cos_sim = torch.abs(torch.dot(u1 / u1_norm, v_decay)).item()
    return float(cos_sim)

def project_displacement(dw, U_r):
    """
    Projects displacement vector dw (P,) onto subspace spanned by orthonormal columns U_r (P, r).
    dw_parallel = U_r (U_r^T dw)
    dw_perp = dw - dw_parallel
    Returns:
        dw_parallel, dw_perp, coverage_ratio
    """
    coeffs = torch.matmul(U_r.T, dw.float()) # (r,)
    dw_parallel = torch.matmul(U_r, coeffs) # (P,)
    dw_perp = dw.float() - dw_parallel # (P,)

    norm_total_sq = torch.sum(dw.float()**2) + 1e-12
    norm_parallel_sq = torch.sum(dw_parallel**2)
    coverage = (norm_parallel_sq / norm_total_sq).item()
    return dw_parallel, dw_perp, float(min(max(coverage, 0.0), 1.0))
