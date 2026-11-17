import pytest
import torch
import numpy as np
from src.models import GrokkingTransformer
from src.subspace_analysis import (
    get_layer_parameter_indices,
    compute_svd_basis,
    compute_radial_decay_alignment,
    project_displacement
)

class TestSubspaceAnalysis:
    def test_layer_parameter_indices(self):
        model = GrokkingTransformer(p=97)
        layer_indices = get_layer_parameter_indices(model)

        # Check all 4 expected groups are present
        assert set(layer_indices.keys()) == {"embed", "attn", "mlp", "unembed"}

        # Check total parameter count
        total_indexed = sum(len(v) for v in layer_indices.values())
        expected_total = sum(p.numel() for p in model.parameters())
        assert total_indexed == expected_total == 223104

        # Check disjointness
        all_indices = []
        for v in layer_indices.values():
            all_indices.extend(v)
        assert len(all_indices) == len(set(all_indices))
        assert set(all_indices) == set(range(expected_total))

        # Check exact group counts
        assert len(layer_indices["embed"]) == 12928
        assert len(layer_indices["attn"]) == 66048
        assert len(layer_indices["mlp"]) == 131712
        assert len(layer_indices["unembed"]) == 12416

    def test_svd_basis_and_orthonormality(self):
        torch.manual_seed(42)
        P, T = 1000, 32
        D = torch.randn(P, T)

        U, S, cum_var = compute_svd_basis(D, rank=8)
        assert U.shape == (P, 8)
        assert len(S) == T
        assert len(cum_var) == T
        assert cum_var[-1] == pytest.approx(1.0, abs=1e-5)

        # Orthonormality check: U^T U = I_8
        eye = torch.matmul(U.T, U)
        assert torch.allclose(eye, torch.eye(8), atol=1e-5)

    def test_radial_decay_alignment(self):
        torch.manual_seed(42)
        P = 500
        w_ref = torch.randn(P)

        # 1. Perfectly aligned direction: u1 = -w_ref
        u1_aligned = (-w_ref / torch.norm(w_ref)).unsqueeze(1)
        align_perfect = compute_radial_decay_alignment(u1_aligned, w_ref)
        assert align_perfect == pytest.approx(1.0, abs=1e-5)

        # 2. Orthogonal direction
        u1_orth = torch.randn(P)
        u1_orth = u1_orth - (torch.dot(u1_orth, w_ref) / torch.norm(w_ref)**2) * w_ref
        u1_orth = (u1_orth / torch.norm(u1_orth)).unsqueeze(1)
        align_orth = compute_radial_decay_alignment(u1_orth, w_ref)
        assert align_orth == pytest.approx(0.0, abs=1e-5)

    def test_project_displacement_orthogonality_and_energy(self):
        torch.manual_seed(42)
        P, T, r = 1000, 32, 4
        D = torch.randn(P, T)
        U, S, _ = compute_svd_basis(D, rank=r)

        dw = torch.randn(P)
        dw_par, dw_perp, cov = project_displacement(dw, U)

        # 1. Par + Perp = dw
        assert torch.allclose(dw_par + dw_perp, dw, atol=1e-5)

        # 2. Orthogonality: <dw_par, dw_perp> = 0
        dot = torch.dot(dw_par, dw_perp).item()
        assert abs(dot) < 1e-4

        # 3. Pythagorean theorem: ||dw_par||^2 + ||dw_perp||^2 = ||dw||^2
        norm_par_sq = torch.sum(dw_par**2).item()
        norm_perp_sq = torch.sum(dw_perp**2).item()
        norm_tot_sq = torch.sum(dw**2).item()
        assert (norm_par_sq + norm_perp_sq) == pytest.approx(norm_tot_sq, rel=1e-4)

        # 4. Coverage in [0, 1]
        assert 0.0 <= cov <= 1.0
        assert cov == pytest.approx(norm_par_sq / norm_tot_sq, rel=1e-4)
