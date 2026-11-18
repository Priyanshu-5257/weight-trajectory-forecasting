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

def compute_svd_basis(D, rank=None, return_effective_rank=False, tol=None):
    """
    Computes thin SVD of displacement matrix D: (P, T)
    D = U S V^T
    Automatically filters out zero-displacement columns (e.g. W_ref - W_ref = 0)
    and drops singular vectors below numerical tolerance to prevent nullspace leakage.
    Returns:
        U: (P, min(rank, r_eff)) orthonormal basis
        S: singular values (T_valid,)
        cum_var: cumulative explained variance ratio
        r_eff: (optional) effective numerical rank
    """
    col_norms = torch.norm(D.float(), dim=0)
    valid_cols = col_norms > 1e-12
    if torch.sum(valid_cols) > 0:
        D_valid = D[:, valid_cols].float()
    else:
        D_valid = D.float()

    U, S, V = torch.linalg.svd(D_valid, full_matrices=False)

    # Numerical tolerance for effective rank
    if tol is None:
        eps = torch.finfo(torch.float32).eps
        tol = (S[0] * max(D_valid.shape) * eps).item() if len(S) > 0 else 1e-12
    r_eff = int(torch.sum(S > tol).item())

    var = S**2
    total_var = torch.sum(var) + 1e-12
    cum_var = torch.cumsum(var, dim=0) / total_var

    max_valid_rank = min(r_eff, U.shape[1])
    if rank is not None:
        r = min(rank, max_valid_rank)
        U = U[:, :r]
    else:
        U = U[:, :max_valid_rank]

    if return_effective_rank:
        return U, S, cum_var, r_eff
    return U, S, cum_var

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
