"""
Phase 1C — SABiT Transformer Block
====================================
Assembles one complete SABiT block:
  EvidentialGraphLearner → GraphBiasedAttention → FFN

Each block:
  1. Learns graph structure from current features (StructureNet)
  2. Applies structure-biased attention (GraphBiasedAttention)
  3. Applies feed-forward network
  4. Uses layer scale (CaiT/DeiT-III) for deep training stability
  5. Uses DropPath (stochastic depth) for regularization

Architecture:
  x → EvidentialGraphLearner(x, pos, A_prior) → A_effective, uncertainty, ev_info
    → LayerNorm → GraphBiasedAttention(·, A_eff, A_prior) → ×layer_scale₁ → DropPath → +Residual
    → LayerNorm → FFN → ×layer_scale₂ → DropPath → +Residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple

from src.boilerplates.models.swinunetr import DropPath
from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner
from src.boilerplates.models.sabit.graph_attention import GraphBiasedAttention


class SABiTBlock(nn.Module):
    """
    One SABiT transformer block: StructureNet + GraphBiasedAttention + FFN.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    d_pair : int
        Pair feature dimension for StructureNet.
    n_evidence : int
        Number of Dirichlet evidence classes.
    top_k : int
        Number of neighbor edges to score.
    mlp_ratio : float
        FFN hidden dimension multiplier.
    drop : float
        Dropout rate in FFN.
    attn_drop : float
        Dropout rate on attention weights.
    drop_path : float
        Stochastic depth rate.
    layer_scale_init : float
        Initial value for layer scale parameters (CaiT-style).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_pair: int = 64,
        n_evidence: int = 4,
        top_k: int = 32,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()

        # ── Graph structure learner (Phase 1A) ────────────────
        self.graph_learner = EvidentialGraphLearner(
            embed_dim=dim,
            d_pair=d_pair,
            n_evidence=n_evidence,
            top_k=top_k,
            dropout=drop,
        )

        # ── Graph-biased attention (Phase 1B) ─────────────────
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphBiasedAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # ── Feed-forward network ──────────────────────────────
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # ── Regularization ────────────────────────────────────
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ── Layer scale (CaiT/DeiT-III) ──────────────────────
        # Multiplies residual by a small learnable scalar (init 1e-4)
        # Prevents early training instability in deep stacks
        self.layer_scale_1 = nn.Parameter(
            torch.full((dim,), layer_scale_init)
        )
        self.layer_scale_2 = nn.Parameter(
            torch.full((dim,), layer_scale_init)
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        A_prior: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """
        Forward pass through one SABiT block.

        Parameters
        ----------
        x : Tensor [B, N, d]
            Token features.
        positions : Tensor [N, 3]
            Token 3D positions.
        A_prior : Tensor [B, N, N]
            Structural prior from Phase 0.

        Returns
        -------
        x : Tensor [B, N, d]
            Updated token features.
        A_effective : Tensor [B, N, N]
            Learned graph from this block's StructureNet.
        uncertainty : Tensor [B, N, N]
            Per-edge uncertainty map.
        ev_info : dict
            Evidential diagnostics (alpha, mean_uncertainty, etc.).
        """
        # ── Graph structure learning ──────────────────────────
        A_effective, uncertainty, ev_info = self.graph_learner(
            x, positions, A_prior
        )

        # ── Attention with structure bias ─────────────────────
        # Pre-norm → attention → scale → drop_path → residual
        x = x + self.drop_path(
            self.layer_scale_1 * self.attn(
                self.norm1(x), A_effective, A_prior
            )
        )

        # ── Feed-forward network ──────────────────────────────
        # Pre-norm → FFN → scale → drop_path → residual
        x = x + self.drop_path(
            self.layer_scale_2 * self.ffn(self.norm2(x))
        )

        return x, A_effective, uncertainty, ev_info


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder

    print("=" * 60)
    print("  Phase 1C - SABiTBlock Unit Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d, heads = 2, 512, 384, 12

    # Build prior
    builder = HyperFidelityPriorBuilder(embed_dim=d, top_k=32).to(device)
    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    features = torch.randn(B, N, d, device=device)
    A_prior, _ = builder(features, positions)

    # Build block
    block = SABiTBlock(dim=d, num_heads=heads, d_pair=64, n_evidence=4, top_k=32).to(device)
    x = torch.randn(B, N, d, device=device, requires_grad=True)

    # Forward
    out, A_eff, unc, ev_info = block(x, positions, A_prior.detach())

    assert out.shape == (B, N, d), f"Shape: {out.shape}"
    assert A_eff.shape == (B, N, N)
    assert unc.shape == (B, N, N)
    assert torch.isfinite(out).all()
    print(f"  [OK] Shape: {out.shape}")
    print(f"  [OK] Finite output")

    # Backward
    out.sum().backward()
    assert x.grad is not None
    assert block.layer_scale_1.grad is not None
    print(f"  [OK] Differentiable (layer_scale grads exist)")

    params = sum(p.numel() for p in block.parameters())
    print(f"  [OK] Parameters: {params:,}")
    print(f"  [OK] ALL PHASE 1C TESTS PASSED")
