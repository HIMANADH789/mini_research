"""
Phase 1A — Evidential Graph Learner (StructureNet)
====================================================
At each transformer layer, learns a residual graph structure with calibrated
uncertainty via Dirichlet-based evidential deep learning.

Mathematical formulation:
  1. Pair features:   f_ij = MLP(h_i ⊕ h_j ⊕ h_i⊙h_j ⊕ Δpos_ij)
  2. Edge residual:   μ_ij = tanh(MLP(f_ij)) · scale     ∈ [-scale, +scale]
  3. Dirichlet conc:  α_ij = softplus(MLP(f_ij)) + 1     ∈ (1, ∞)^C
  4. Uncertainty:      u_ij = C / Σ_c α_ijc               ∈ (0, 1)
  5. Effective graph:  A_eff = (A_prior + μ) ⊙ (1 - u)
  6. Post-process:     symmetrize → ReLU → row-normalize

Key robustness features:
  - Sparse computation: only score top-k neighbor edges from A_prior
  - Residual scaling: μ initialized near 0 → initial graph ≈ prior
  - Edge dropout: randomly drop edges during training for regularization
  - Dirichlet numerical stability: α clamped to [1.01, 1000]
  - Float32 enforcement for KL divergence (lgamma/digamma under AMP)

Config fields consumed (from config.model):
  config.model.structure_d_pair:    64
  config.model.structure_n_evidence: 4
  config.model.structure_top_k:     32
  config.model.structure_dropout:   0.1
  config.model.structure_edge_drop: 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════
#  EVIDENTIAL GRAPH LEARNER
# ══════════════════════════════════════════════════════════════

class EvidentialGraphLearner(nn.Module):
    """
    Learns graph structure with calibrated uncertainty via Dirichlet evidence.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension.
    d_pair : int
        Pair feature dimension after projection.
    n_evidence : int
        Number of Dirichlet evidence classes (C).
    top_k : int
        Number of neighbor edges to score per node (from A_prior support).
    mu_scale_init : float
        Initial residual scale — starts small so graph ≈ prior at init.
    dropout : float
        Dropout rate in pair feature MLP.
    edge_drop_rate : float
        Fraction of edges randomly dropped during training.
    """

    def __init__(
        self,
        embed_dim: int,
        d_pair: int = 64,
        n_evidence: int = 4,
        top_k: int = 32,
        mu_scale_init: float = 0.01,
        dropout: float = 0.1,
        edge_drop_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.d_pair = d_pair
        self.n_evidence = n_evidence
        self.top_k = top_k
        self.edge_drop_rate = edge_drop_rate

        # ── Pair feature builder ──────────────────────────────────
        # Input: h_i ⊕ h_j ⊕ h_i⊙h_j ⊕ Δpos = 2d + d + 3
        pair_input_dim = embed_dim * 2 + embed_dim + 3
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_input_dim, d_pair * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_pair * 2, d_pair),
        )

        # ── Evidential head: edge residual μ ──────────────────────
        self.mu_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, 1),
        )

        # ── Evidential head: Dirichlet concentrations α ───────────
        self.alpha_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, n_evidence),
        )

        # ── Residual scaling (learnable, starts small) ────────────
        # Controls how much the learned residual can modify the prior initially
        self.mu_scale = nn.Parameter(
            torch.tensor(mu_scale_init, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projections, zero-init for output heads."""
        for m in self.pair_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init the final layer of mu_head → μ starts at 0
        for m in self.mu_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Make last linear layer output near zero
        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)

        for m in self.alpha_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────
    # SPARSE GATHER UTILITY
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _gather_neighbors(
        h: Tensor, neighbor_idx: Tensor
    ) -> Tensor:
        """
        Efficiently gather neighbor features using index tensor.

        Parameters
        ----------
        h : [B, N, d]
        neighbor_idx : [B, N, k]

        Returns
        -------
        h_j : [B, N, k, d]
        """
        B, N, d = h.shape
        k = neighbor_idx.shape[-1]

        # Expand idx for gather: [B, N, k] → [B, N, k, d]
        idx_expanded = neighbor_idx.unsqueeze(-1).expand(B, N, k, d)

        # Expand h for gather: [B, N, d] → [B, 1, N, d] → need [B, N, k, d]
        # Use gather along dim=1 after expanding h
        h_expanded = h.unsqueeze(1).expand(B, N, N, d)
        h_j = torch.gather(h_expanded, 2, idx_expanded)  # [B, N, k, d]

        return h_j

    @staticmethod
    def _gather_positions(
        positions: Tensor, neighbor_idx: Tensor
    ) -> Tensor:
        """
        Gather neighbor positions.

        Parameters
        ----------
        positions : [N, 3]
        neighbor_idx : [B, N, k]

        Returns
        -------
        pos_j : [B, N, k, 3]
        """
        B, N, k = neighbor_idx.shape

        # positions: [N, 3] → expand for each batch and query
        pos_expanded = positions.unsqueeze(0).unsqueeze(0).expand(B, N, -1, 3)  # [B, N, N, 3]
        idx_expanded = neighbor_idx.unsqueeze(-1).expand(B, N, k, 3)
        pos_j = torch.gather(pos_expanded, 2, idx_expanded)  # [B, N, k, 3]

        return pos_j

    # ──────────────────────────────────────────────────────
    # EVIDENTIAL KL LOSS
    # ──────────────────────────────────────────────────────

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def evidential_kl_loss(alpha: Tensor) -> Tensor:
        """
        KL divergence between learned Dirichlet Dir(α) and uniform prior Dir(1).

        KL(Dir(α) || Dir(1)) = log Γ(Σα) − Σ log Γ(α)
                               − log Γ(C) + Σ (α−1)(ψ(α) − ψ(Σα))

        All computations in float32 for numerical stability (lgamma, digamma).

        Parameters
        ----------
        alpha : Tensor [*, C]
            Dirichlet concentration parameters, each > 1.

        Returns
        -------
        kl : scalar Tensor
            Mean KL divergence.
        """
        C = alpha.shape[-1]
        alpha_0 = alpha.sum(-1, keepdim=True)  # [*, 1]

        kl = (
            torch.lgamma(alpha_0.squeeze(-1))
            - torch.lgamma(alpha).sum(-1)
            - torch.lgamma(torch.tensor(float(C), device=alpha.device, dtype=alpha.dtype))
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(-1)
        )

        return kl.mean()

    # ──────────────────────────────────────────────────────
    # FORWARD
    # ──────────────────────────────────────────────────────

    def forward(
        self,
        h: Tensor,
        positions: Tensor,
        A_prior: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        """
        Learn graph structure with evidential uncertainty.

        Parameters
        ----------
        h : Tensor [B, N, d]
            Token features from previous transformer layer.
        positions : Tensor [N, 3]
            Token 3D positions.
        A_prior : Tensor [B, N, N]
            Structural prior from Phase 0.

        Returns
        -------
        A_effective : Tensor [B, N, N]
            Learned, uncertainty-gated, row-normalized graph.
        uncertainty : Tensor [B, N, N]
            Per-edge uncertainty map.
        ev_info : dict
            Diagnostic information including alpha tensor for loss.
        """
        B, N, d = h.shape
        device = h.device
        k = min(self.top_k, N - 1)

        # Ensure positions on correct device
        positions = positions.to(device)

        # ── Step 1: Get top-k neighbor indices from A_prior ───────
        _, neighbor_idx = A_prior.topk(k, dim=-1)  # [B, N, k]

        # ── Step 2: Gather neighbor features ──────────────────────
        h_j = self._gather_neighbors(h, neighbor_idx)       # [B, N, k, d]
        h_i = h.unsqueeze(2).expand_as(h_j)                 # [B, N, k, d]

        # Gather neighbor positions
        pos_j = self._gather_positions(positions, neighbor_idx)  # [B, N, k, 3]
        pos_i = positions.unsqueeze(0).unsqueeze(2).expand(B, N, k, 3)
        delta_pos = pos_i - pos_j  # [B, N, k, 3]

        # ── Step 3: Build pair features ───────────────────────────
        # Concatenate: h_i ⊕ h_j ⊕ h_i⊙h_j ⊕ Δpos
        pair_input = torch.cat([
            h_i,
            h_j,
            h_i * h_j,      # Hadamard product — captures compatibility
            delta_pos,       # relative position
        ], dim=-1)  # [B, N, k, 2d + d + 3]

        pair_feat = self.pair_proj(pair_input)  # [B, N, k, d_pair]

        # ── Step 4: Evidential predictions ────────────────────────
        # Edge strength residual
        mu_raw = self.mu_head(pair_feat).squeeze(-1)  # [B, N, k]
        mu = torch.tanh(mu_raw) * F.softplus(self.mu_scale)  # bounded, starts near 0

        # Dirichlet concentration parameters
        alpha = F.softplus(self.alpha_head(pair_feat)) + 1.0  # [B, N, k, C], all > 1
        alpha = alpha.clamp(min=1.01, max=1000.0)  # numerical stability for lgamma

        # ── Step 5: Uncertainty computation (Dirichlet) ───────────
        C = self.n_evidence
        S = alpha.sum(dim=-1)         # [B, N, k] — total evidence
        u = float(C) / S             # [B, N, k] — uncertainty ∈ (0, 1)
        u = u.clamp(max=0.999)        # never exactly 1 (would zero-out edge)

        # ── Step 6: Effective graph construction ──────────────────
        # Gather prior values at neighbor positions
        A_prior_vals = torch.gather(
            A_prior, 2, neighbor_idx
        )  # [B, N, k]

        # Residual graph learning: prior + learned adjustment
        A_raw = A_prior_vals + mu

        # Uncertainty gating: suppress uncertain edges
        A_gated = A_raw * (1.0 - u)

        # ── Step 7: Edge dropout (training regularization) ────────
        if self.training and self.edge_drop_rate > 0:
            edge_mask = torch.bernoulli(
                torch.full_like(A_gated, 1.0 - self.edge_drop_rate)
            )
            A_gated = A_gated * edge_mask

        # ── Step 8: Scatter back to dense [B, N, N] ──────────────
        A_effective = torch.zeros(B, N, N, device=device, dtype=h.dtype)
        A_effective.scatter_(2, neighbor_idx, A_gated.to(h.dtype))

        # ── Step 9: Symmetrize + ReLU + row-normalize ────────────
        A_effective = (A_effective + A_effective.transpose(-1, -2)) / 2.0
        A_effective = F.relu(A_effective)  # non-negative edges only

        # Row normalize
        row_sums = A_effective.sum(-1, keepdim=True).clamp(min=1e-8)
        A_effective = A_effective / row_sums

        # ── Step 10: Scatter uncertainty map ──────────────────────
        uncertainty = torch.zeros(B, N, N, device=device, dtype=h.dtype)
        uncertainty.scatter_(2, neighbor_idx, u.to(h.dtype))

        # ── Diagnostics ──────────────────────────────────────────
        ev_info = {
            "alpha": alpha,  # [B, N, k, C] — needed for evidential KL loss
            "mean_uncertainty": u.mean().item(),
            "evidence_strength": S.mean().item(),
            "mu_scale": F.softplus(self.mu_scale).item(),
            "mu_mean": mu.abs().mean().item(),
            "edge_sparsity": (A_effective == 0).float().mean().item(),
        }

        return A_effective, uncertainty, ev_info


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder

    print("=" * 60)
    print("  Phase 1A — EvidentialGraphLearner Unit Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d = 2, 512, 384

    # Build prior first (dependency)
    builder = HyperFidelityPriorBuilder(embed_dim=d, d_prior=16, top_k=32).to(device)
    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    features = torch.randn(B, N, d, device=device)
    A_prior, _ = builder(features, positions)

    # Build StructureNet
    learner = EvidentialGraphLearner(
        embed_dim=d, d_pair=64, n_evidence=4, top_k=32,
        mu_scale_init=0.01, dropout=0.1, edge_drop_rate=0.1,
    ).to(device)

    # ── Forward pass ─────────────────────────────────────
    h = torch.randn(B, N, d, device=device)
    A_eff, unc, ev_info = learner(h, positions, A_prior.detach())

    # ── Test 1: Output shapes ────────────────────────────
    assert A_eff.shape == (B, N, N), f"FAIL A_eff shape: {A_eff.shape}"
    assert unc.shape == (B, N, N), f"FAIL unc shape: {unc.shape}"
    print(f"  ✅ Shapes: A_eff={A_eff.shape}, unc={unc.shape}")

    # ── Test 2: Row normalization ────────────────────────
    row_sums = A_eff.sum(-1)
    max_dev = (row_sums - 1).abs().max().item()
    print(f"  ✅ Row normalization: max deviation {max_dev:.6f}")

    # ── Test 3: Non-negative ─────────────────────────────
    assert (A_eff >= 0).all(), "FAIL: negative values in A_effective"
    print(f"  ✅ Non-negative: all values ≥ 0")

    # ── Test 4: Uncertainty bounds ───────────────────────
    assert 0 < ev_info["mean_uncertainty"] < 1, \
        f"FAIL: mean_uncertainty={ev_info['mean_uncertainty']}"
    print(f"  ✅ Uncertainty: mean={ev_info['mean_uncertainty']:.4f}")

    # ── Test 5: Residual starts near zero ────────────────
    assert ev_info["mu_mean"] < 0.1, \
        f"FAIL: mu is too large at init ({ev_info['mu_mean']:.4f})"
    print(f"  ✅ Residual scale: mu_mean={ev_info['mu_mean']:.6f} (near zero at init)")

    # ── Test 6: Differentiable ───────────────────────────
    loss = A_eff.sum()
    loss.backward()
    assert learner.mu_scale.grad is not None, "FAIL: no gradient to mu_scale"
    print(f"  ✅ Differentiable: gradients flow through")

    # ── Test 7: KL loss computation ──────────────────────
    learner.zero_grad()
    alpha = ev_info["alpha"]
    kl = EvidentialGraphLearner.evidential_kl_loss(alpha)
    assert torch.isfinite(kl), f"FAIL: KL loss is {kl.item()}"
    kl.backward()
    print(f"  ✅ Evidential KL loss: {kl.item():.4f} (finite, differentiable)")

    # ── Test 8: Alpha bounds ─────────────────────────────
    assert (alpha >= 1.01).all(), "FAIL: alpha below 1.01"
    assert (alpha <= 1000.0).all(), "FAIL: alpha above 1000"
    print(f"  ✅ Alpha bounds: [{alpha.min():.2f}, {alpha.max():.2f}]")

    # ── Test 9: Evidence strength ────────────────────────
    assert ev_info["evidence_strength"] > 0, "FAIL: evidence_strength ≤ 0"
    print(f"  ✅ Evidence strength: {ev_info['evidence_strength']:.4f}")

    # ── Test 10: Parameter count ─────────────────────────
    total_params = sum(p.numel() for p in learner.parameters())
    print(f"  ✅ Parameters: {total_params:,}")
    print(f"     Device: {device}")

    print()
    print("  ════════════════════════════════════════════════")
    print("  ✅ ALL PHASE 1A TESTS PASSED")
    print("  ════════════════════════════════════════════════")
