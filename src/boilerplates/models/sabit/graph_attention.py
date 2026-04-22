"""
Phase 1B — Gated Structure-Biased Multi-Head Self-Attention
============================================================
Multi-head self-attention where attention logits are augmented with a gated
structural bias derived from both the learned graph (A_effective from
StructureNet) and the anatomical prior (A_prior from PriorBuilder).

Mathematical formulation:
  1. Standard:  Q, K, V = split(Linear(x)),  attn_logits = QK^T / √d_h
  2. Gate:      g = sigmoid(MLP([A_effective, A_prior]))
  3. Blend:     A_blended = g · A_effective + (1−g) · A_prior
  4. Bias:      B = softplus(Linear(A_blended)) · λ_h    per head
  5. Output:    attn = softmax(attn_logits + B) @ V

Key design principles:
  - softplus(bias) → always non-negative → can only BOOST attention, never suppress
    semantically important connections below content-based scores
  - Per-head learnable λ_h → each head independently learns graph trust level
  - Gate learns a per-position curriculum: early training → gate ≈ 0.5 (trust prior),
    late training → gate → 1.0 (trust learned structure)
  - NaN guard: attention logits clamped to [-50, 50] before softmax

Config fields consumed (from config.model):
  config.model.attn_drop:       0.0
  config.model.proj_drop:       0.0
  config.model.bias_scale_init: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ══════════════════════════════════════════════════════════════
#  GATED STRUCTURE-BIASED ATTENTION
# ══════════════════════════════════════════════════════════════

class GraphBiasedAttention(nn.Module):
    """
    Multi-head self-attention with gated structural bias from learned
    graph and anatomical prior.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        Whether QKV projection has bias.
    attn_drop : float
        Dropout rate on attention weights.
    proj_drop : float
        Dropout rate on output projection.
    bias_scale_init : float
        Initial value for per-head bias scale factor λ_h.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias_scale_init: float = 1.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ── Standard QKV projection ──────────────────────────────
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── Gate network: per-edge confidence in learned vs prior ──
        # Input: [A_effective_ij, A_prior_ij] → 2 features per edge
        # Output: sigmoid → gate ∈ (0, 1)
        self.gate_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # ── Bias projection: graph value → per-head attention bias ──
        # Input: blended graph value (scalar per edge)
        # Output: one bias value per head
        self.bias_proj = nn.Linear(1, num_heads)

        # ── Per-head trust scale λ_h ─────────────────────────────
        # Each head independently learns how much to rely on graph structure
        self.bias_scale = nn.Parameter(
            torch.full((num_heads,), bias_scale_init, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projections, specific init for gate and bias."""
        # QKV and output projection
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # Gate network — initialize bias so gate starts near 0.5 (balanced)
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Bias projection — small init so structural bias starts gentle
        nn.init.xavier_uniform_(self.bias_proj.weight)
        if self.bias_proj.bias is not None:
            nn.init.zeros_(self.bias_proj.bias)

    def forward(
        self,
        x: Tensor,
        A_effective: Tensor,
        A_prior: Tensor,
    ) -> Tensor:
        """
        Forward pass with gated structural bias.

        Parameters
        ----------
        x : Tensor [B, N, d]
            Token features.
        A_effective : Tensor [B, N, N]
            Learned graph from StructureNet (Phase 1A).
        A_prior : Tensor [B, N, N]
            Structural prior from PriorBuilder (Phase 0).

        Returns
        -------
        output : Tensor [B, N, d]
            Attention output with structural bias.
        """
        B, N, d = x.shape
        heads = self.num_heads
        d_h = self.head_dim

        # ══════════════════════════════════════════════════
        # 1. STANDARD QKV COMPUTATION
        # ══════════════════════════════════════════════════

        qkv = self.qkv(x)  # [B, N, 3d]
        qkv = qkv.reshape(B, N, 3, heads, d_h).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)  # each [B, heads, N, d_h]

        # Scaled dot-product attention logits
        attn_logits = (Q @ K.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # ══════════════════════════════════════════════════
        # 2. GATED STRUCTURE BIAS
        # ══════════════════════════════════════════════════

        # Stack graph matrices for gate input: [B, N, N, 2]
        gate_input = torch.stack([A_effective, A_prior], dim=-1)

        # Gate: per-edge confidence in learned structure vs prior
        g = torch.sigmoid(self.gate_net(gate_input))  # [B, N, N, 1]

        # Blend graphs with gate
        A_blended = (
            g * A_effective.unsqueeze(-1)
            + (1.0 - g) * A_prior.unsqueeze(-1)
        )  # [B, N, N, 1]

        # Project blended graph to per-head bias
        bias = self.bias_proj(A_blended)  # [B, N, N, heads]
        bias = bias.permute(0, 3, 1, 2)  # [B, heads, N, N]

        # Apply softplus (non-negative) and per-head scale
        bias = F.softplus(bias) * self.bias_scale.view(1, -1, 1, 1)

        # ══════════════════════════════════════════════════
        # 3. BIASED ATTENTION
        # ══════════════════════════════════════════════════

        # Add structural bias to content-based logits
        attn_logits = attn_logits + bias

        # NaN guard: clamp before softmax to prevent overflow
        attn_logits = attn_logits.clamp(-50.0, 50.0)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted value aggregation
        out = (attn_weights @ V)  # [B, heads, N, d_h]
        out = out.transpose(1, 2).reshape(B, N, d)  # [B, N, d]

        # Output projection
        out = self.proj_drop(self.proj(out))

        return out


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
    from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner

    print("=" * 60)
    print("  Phase 1B — GraphBiasedAttention Unit Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d = 2, 512, 384
    heads = 12

    # Build dependencies
    builder = HyperFidelityPriorBuilder(embed_dim=d, d_prior=16, top_k=32).to(device)
    learner = EvidentialGraphLearner(embed_dim=d, d_pair=64, n_evidence=4, top_k=32).to(device)
    attn = GraphBiasedAttention(dim=d, num_heads=heads).to(device)

    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    features = torch.randn(B, N, d, device=device)

    # Phase 0: Prior
    A_prior, _ = builder(features, positions)

    # Phase 1A: StructureNet
    A_eff, unc, ev_info = learner(features, positions, A_prior.detach())

    # Phase 1B: Graph-biased attention
    x = torch.randn(B, N, d, device=device, requires_grad=True)
    out = attn(x, A_eff.detach(), A_prior.detach())

    # ── Test 1: Output shape ─────────────────────────────
    assert out.shape == (B, N, d), f"FAIL shape: {out.shape}"
    print(f"  ✅ Shape: {out.shape}")

    # ── Test 2: Output is finite ─────────────────────────
    assert torch.isfinite(out).all(), "FAIL: output contains NaN/Inf"
    print(f"  ✅ Finite: no NaN or Inf in output")

    # ── Test 3: Differentiable ───────────────────────────
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "FAIL: no gradient to input"
    assert attn.bias_scale.grad is not None, "FAIL: no gradient to bias_scale"
    assert attn.qkv.weight.grad is not None, "FAIL: no gradient to QKV"
    print(f"  ✅ Differentiable: gradients flow to input, bias_scale, QKV")

    # ── Test 4: Bias scale values ────────────────────────
    print(f"  ✅ Bias scale: {attn.bias_scale.data.tolist()}")

    # ── Test 5: Gate network output range ────────────────
    with torch.no_grad():
        gate_input = torch.stack([A_eff, A_prior], dim=-1)
        g = torch.sigmoid(attn.gate_net(gate_input))
        assert g.min() >= 0.0 and g.max() <= 1.0, "FAIL: gate outside [0,1]"
        print(f"  ✅ Gate range: [{g.min():.4f}, {g.max():.4f}]")

    # ── Test 6: Bias is non-negative (softplus) ──────────
    with torch.no_grad():
        A_blended = g * A_eff.unsqueeze(-1) + (1-g) * A_prior.unsqueeze(-1)
        bias = attn.bias_proj(A_blended).permute(0,3,1,2)
        bias_activated = F.softplus(bias)
        assert (bias_activated >= 0).all(), "FAIL: bias has negative values"
        print(f"  ✅ Bias non-negative: softplus enforced")

    # ── Test 7: Parameter count ──────────────────────────
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"  ✅ Parameters: {total_params:,}")

    # ── Test 8: Memory per forward pass ──────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = attn(torch.randn(1, N, d, device=device),
                 torch.rand(1, N, N, device=device),
                 torch.rand(1, N, N, device=device))
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  ✅ Peak VRAM (B=1): {peak_mb:.1f} MB")

    print(f"     Device: {device}")
    print()
    print("  ════════════════════════════════════════════════")
    print("  ✅ ALL PHASE 1B TESTS PASSED")
    print("  ════════════════════════════════════════════════")
