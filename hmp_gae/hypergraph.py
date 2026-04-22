# hmp_gae/hypergraph.py
# k-NN hypergraph construction for HMP-GAE.
#
# For each node i we create one hyperedge epsilon_i = {i} U top-k nearest
# neighbors of i (by cosine similarity in the eta feature space). Setting
# M = N (one hyperedge per node centered at that node) makes the incidence
# matrix square and keeps the decoder dimension stable across rounds.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def knn_hypergraph(
    eta: torch.Tensor,
    k: int,
    include_self: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a k-NN hypergraph from node features.

    Args:
        eta: (N, d) node feature matrix (dense).
        k:   number of neighbors per hyperedge (excluding self). Effective k
             is clipped to [1, N-1]. When include_self=True, each hyperedge
             contains k+1 nodes (self + k neighbors).
        include_self: if True, node i is always in hyperedge epsilon_i.
        eps: numerical guard for degree inversion.

    Returns:
        H     : (N, N) incidence matrix in {0, 1}. H[i, e] = 1 means node i
                belongs to hyperedge e. Column e is centered on node e.
        D_V_inv : (N,) 1/degree(node i). Used as diag(D_V^{-1}) (kept as a
                  vector to avoid constructing a dense diag matrix).
        D_E_inv : (N,) 1/degree(hyperedge e). Same.
    """
    if eta.dim() != 2:
        raise ValueError(f"knn_hypergraph expects 2D eta, got {eta.shape}")
    N, _ = eta.shape
    if N == 0:
        raise ValueError("knn_hypergraph received empty eta (N=0)")

    # Effective neighborhood size.
    k_eff = max(1, min(int(k), N - 1))

    # Pairwise cosine similarity matrix (N, N). Diagonal becomes 1.
    eta_n = F.normalize(eta, p=2, dim=1, eps=eps)
    sim = eta_n @ eta_n.t()

    # Mask out self-similarities so they don't dominate the top-k selection.
    sim_for_knn = sim.clone()
    sim_for_knn.fill_diagonal_(float("-inf"))

    # top-k neighbors per node (as columns of the hyperedge centered at node i).
    _, nbrs = torch.topk(sim_for_knn, k=k_eff, dim=1)  # (N, k_eff)

    H = torch.zeros(N, N, device=eta.device, dtype=eta.dtype)
    rows = torch.arange(N, device=eta.device).view(-1, 1).expand_as(nbrs)
    # H[ nbrs[i, j], i ] = 1 : neighbor membership in hyperedge i
    # i.e. column i is filled with the neighbors of center node i.
    H[nbrs, rows] = 1.0
    if include_self:
        # Also include the center node itself in its hyperedge.
        diag_idx = torch.arange(N, device=eta.device)
        H[diag_idx, diag_idx] = 1.0

    # Degree vectors (node degree = #hyperedges node belongs to; hyperedge
    # degree = #nodes in that hyperedge).
    d_v = H.sum(dim=1)  # (N,)
    d_e = H.sum(dim=0)  # (N,) -- equal to k_eff + (1 if include_self else 0)

    D_V_inv = 1.0 / d_v.clamp(min=eps)
    D_E_inv = 1.0 / d_e.clamp(min=eps)

    return H, D_V_inv, D_E_inv


def apply_diag_inv(D_inv_vec: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficient equivalent of diag(D_inv_vec) @ X via broadcasting.
    """
    if D_inv_vec.dim() != 1:
        raise ValueError(f"Expected 1D D_inv_vec, got {D_inv_vec.shape}")
    return D_inv_vec.unsqueeze(1) * X
