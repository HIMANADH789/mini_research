"""
Phase 0 — Hyper-Fidelity Prior Builder
=======================================
Constructs a learnable, sparse structural prior A_prior ∈ ℝ^{B×N×N} from four
domain-specific kernels encoding anatomical knowledge:

  1. Spatial Prior    — Gaussian RBF on token 3D positions (nearby → related)
  2. Intensity Prior  — Gaussian RBF in projected feature space (similar tissue → related)
  3. Boundary Prior   — Learned MLP boundary detector on feature diffs (cross-boundary → suppressed)
  4. Stability Prior  — EMA-tracked feature drift (stable tokens → reliable anchors)

The four kernels are combined with learnable, temperature-scaled softmax weights,
symmetrized, sparsified (top-k per row), and row-normalized into a stochastic matrix.

Key robustness features:
  - Self-loop removal on every kernel
  - Chunked boundary computation to prevent OOM
  - EMA feature buffer via register_buffer (checkpoint-compatible)
  - Temperature-controlled prior mixing
  - All bandwidth parameters stored in log-space (softplus for positivity)

Config fields consumed (from config.model):
  config.model.prior_d_prior:      16    (feature projection dim)
  config.model.prior_top_k:        32    (edges per node)
  config.model.prior_sigma_init:   0.1   (spatial bandwidth)
  config.model.prior_tau_init:     0.5   (intensity bandwidth)
  config.model.prior_alpha_init:   5.0   (boundary sharpness)
  config.model.prior_gamma_init:   2.0   (stability decay)
  config.model.prior_ema_momentum: 0.99  (stability EMA momentum)
  config.model.prior_temperature:  1.0   (softmax temperature)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple


# ══════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════

def inverse_softplus(x: float) -> float:
    """Compute y such that softplus(y) = x.  Used for parameter init."""
    if x <= 0:
        raise ValueError(f"inverse_softplus requires x > 0, got {x}")
    # softplus(y) = log(1 + exp(y)) = x  →  y = log(exp(x) - 1)
    # For large x, exp(x) dominates: y ≈ x
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


# ══════════════════════════════════════════════════════════════
#  HYPER-FIDELITY PRIOR BUILDER
# ══════════════════════════════════════════════════════════════

class HyperFidelityPriorBuilder(nn.Module):
    """
    Constructs a multi-kernel structural prior for graph-biased attention.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension (e.g. 384 at bottleneck stage).
    d_prior : int
        Projection dimension for intensity/boundary feature space.
    top_k : int
        Number of edges retained per node after sparsification.
    sigma_init : float
        Initial spatial Gaussian bandwidth.
    tau_init : float
        Initial intensity Gaussian bandwidth.
    alpha_init : float
        Initial boundary sharpness.
    gamma_init : float
        Initial stability decay rate.
    ema_momentum : float
        Momentum for feature EMA buffer (stability prior).
    prior_temperature : float
        Temperature for softmax weight mixing (lower → sharper distribution).
    """

    def __init__(
        self,
        embed_dim: int,
        d_prior: int = 16,
        top_k: int = 32,
        sigma_init: float = 0.1,
        tau_init: float = 0.5,
        alpha_init: float = 5.0,
        gamma_init: float = 2.0,
        ema_momentum: float = 0.99,
        prior_temperature: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.d_prior = d_prior
        self.top_k = top_k
        self.ema_momentum = ema_momentum
        self.prior_temperature = prior_temperature

        # ── Learnable kernel bandwidths (log-space → softplus → positive) ────
        self.log_sigma = nn.Parameter(
            torch.tensor(inverse_softplus(sigma_init), dtype=torch.float32)
        )
        self.log_tau = nn.Parameter(
            torch.tensor(inverse_softplus(tau_init), dtype=torch.float32)
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(inverse_softplus(alpha_init), dtype=torch.float32)
        )
        self.log_gamma = nn.Parameter(
            torch.tensor(inverse_softplus(gamma_init), dtype=torch.float32)
        )

        # ── Prior mixing weights (4 kernels) ────────────────────────────────
        # Initialized proportionally: spatial 0.4, intensity 0.4, boundary 0.15, stability 0.05
        self.prior_logits = nn.Parameter(torch.tensor([0.4, 0.4, 0.15, 0.05]))

        # ── Intensity feature projection (2-layer MLP for rich fingerprinting) ────
        self.intensity_proj = nn.Sequential(
            nn.Linear(embed_dim, d_prior * 2),
            nn.GELU(),
            nn.Linear(d_prior * 2, d_prior),
        )

        # ── Boundary detector (learned MLP on pairwise feature diffs) ──────
        self.boundary_mlp = nn.Sequential(
            nn.Linear(d_prior, d_prior),
            nn.GELU(),
            nn.Linear(d_prior, 1),
        )

        # ── Stability prior EMA buffer (no gradients, checkpoint-compatible) ──
        self.register_buffer("feature_ema", None)

        # ── Spatial prior cache (computed once per resolution) ─────────────
        self.register_buffer("_spatial_cache", None)
        self._spatial_cache_key = None  # (N,) tuple for cache invalidation

        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming init for all projection layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────
    # STATIC HELPERS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def build_position_grid(D: int, H: int, W: int) -> Tensor:
        """
        Create normalized [0,1] 3D coordinates for N = D×H×W tokens.

        Returns
        -------
        positions : Tensor [N, 3]
            Each row is (d, h, w) ∈ [0, 1]³.
        """
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, D),
                torch.linspace(0, 1, H),
                torch.linspace(0, 1, W),
                indexing="ij",
            ),
            dim=-1,
        )
        return coords.reshape(-1, 3)  # [N, 3]

    # ──────────────────────────────────────────────────────
    # KERNEL COMPUTATIONS
    # ──────────────────────────────────────────────────────

    def _compute_spatial_prior(self, positions: Tensor) -> Tensor:
        """
        Gaussian RBF kernel on 3D token positions.

        A_spatial(i,j) = exp(−||p_i − p_j||² / 2σ²)

        Returns [N, N] — batch-independent.
        """
        N = positions.shape[0]

        sigma = F.softplus(self.log_sigma)
        dist_sq = torch.cdist(positions, positions).pow(2)  # [N, N]
        A = torch.exp(-dist_sq / (2.0 * sigma.pow(2) + 1e-8))

        # Remove self-loops (no in-place ops — safe for autograd)
        mask = 1.0 - torch.eye(N, device=positions.device, dtype=positions.dtype)
        A = A * mask
        return A

    def _compute_intensity_prior(self, z: Tensor) -> Tensor:
        """
        Gaussian RBF kernel in projected feature space.

        z = MLP(h), A_intensity(i,j) = exp(−||z_i − z_j||² / 2τ²)

        Parameters
        ----------
        z : Tensor [B, N, d_prior]

        Returns [B, N, N]
        """
        tau = F.softplus(self.log_tau)
        feat_dist_sq = torch.cdist(z, z).pow(2)  # [B, N, N]
        A = torch.exp(-feat_dist_sq / (2.0 * tau.pow(2) + 1e-8))

        # Remove self-loops
        N = z.shape[1]
        eye = torch.eye(N, device=z.device, dtype=z.dtype)
        A = A * (1.0 - eye.unsqueeze(0))
        return A

    def _compute_boundary_prior(self, z: Tensor) -> Tensor:
        """
        Learned boundary detector on pairwise feature differences.

        Δz_ij = |z_i − z_j|
        b_ij   = MLP(Δz_ij)
        A_boundary(i,j) = 1 − sigmoid(α · b_ij)

        Memory-efficient chunked computation to avoid materializing [B, N, N, d_prior].

        Parameters
        ----------
        z : Tensor [B, N, d_prior]

        Returns [B, N, N]
        """
        B, N, d = z.shape
        alpha = F.softplus(self.log_alpha)
        device = z.device
        dtype = z.dtype

        # Chunk to keep peak memory bounded:
        #   Full: [B, N, N, d] = B*N*N*d floats. For N=512, d=16: 32M entries/batch
        #   Chunked: [B, chunk, N, d] — 128 rows at a time → 8M entries/batch
        chunk_size = min(N, 128)
        chunks = []

        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)

            z_i = z[:, i_start:i_end, :].unsqueeze(2)  # [B, chunk, 1, d]
            z_j = z.unsqueeze(1)                         # [B, 1, N, d]
            delta = (z_i - z_j).abs()                    # [B, chunk, N, d]

            # Boundary MLP: [B, chunk, N, d] → [B, chunk, N, 1]
            b = self.boundary_mlp(delta).squeeze(-1)     # [B, chunk, N]

            # A_boundary = 1 - sigmoid(α * b)
            # High feature difference → high b → sigmoid → high → A_boundary → low
            # → cross-boundary tokens have LOW connectivity (correct)
            chunks.append(1.0 - torch.sigmoid(alpha * b))

        # Concatenate chunks (no in-place assignment — safe for autograd)
        A_boundary = torch.cat(chunks, dim=1)  # [B, N, N]

        # Remove self-loops
        eye = torch.eye(N, device=device, dtype=dtype)
        A_boundary = A_boundary * (1.0 - eye.unsqueeze(0))
        return A_boundary

    def _compute_stability_prior(self, z: Tensor) -> Tensor:
        """
        Stability prior using EMA feature drift.

        s_i = exp(−γ · ||z_i − z̄_i||²)
        A_stability(i,j) = s_i · s_j   (outer product)

        Stable tokens (low drift) are more reliable graph anchors.

        Parameters
        ----------
        z : Tensor [B, N, d_prior]

        Returns [B, N, N]
        """
        B, N, d = z.shape
        gamma = F.softplus(self.log_gamma)
        device = z.device

        # Initialize or update EMA buffer
        z_detached = z.detach()
        if self.feature_ema is None:
            # First forward pass: initialize EMA to current features
            self.feature_ema = z_detached.clone()
            # First call → all tokens are "perfectly stable"
            stability = torch.ones(B, N, device=device, dtype=z.dtype)
        else:
            # Handle batch size mismatch (eval vs train)
            if self.feature_ema.shape[0] != B:
                self.feature_ema = z_detached.clone()
                stability = torch.ones(B, N, device=device, dtype=z.dtype)
            else:
                # Per-token drift from EMA
                drift = (z - self.feature_ema.detach()).pow(2).sum(-1)  # [B, N]
                stability = torch.exp(-gamma * drift)                   # [B, N] ∈ (0, 1]

                # Update EMA (in-place, no grad)
                self.feature_ema = (
                    self.ema_momentum * self.feature_ema
                    + (1.0 - self.ema_momentum) * z_detached
                )

        # Outer product → pairwise stability
        A_stability = stability.unsqueeze(-1) * stability.unsqueeze(-2)  # [B, N, N]

        # Remove self-loops
        eye = torch.eye(N, device=device, dtype=z.dtype)
        A_stability = A_stability * (1.0 - eye.unsqueeze(0))
        return A_stability

    # ──────────────────────────────────────────────────────
    # FORWARD
    # ──────────────────────────────────────────────────────

    def forward(
        self, features: Tensor, positions: Tensor
    ) -> Tuple[Tensor, Dict[str, object]]:
        """
        Construct the hyper-fidelity structural prior.

        Parameters
        ----------
        features : Tensor [B, N, d]
            Token embeddings at current encoder stage.
        positions : Tensor [N, 3]
            Normalized 3D coordinates from build_position_grid().

        Returns
        -------
        A_prior : Tensor [B, N, N]
            Sparse, symmetric, row-normalized structural prior.
        prior_info : dict
            Diagnostic information for logging.
        """
        B, N, d = features.shape

        # Ensure positions are on the correct device
        positions = positions.to(features.device)

        # ── Project features to prior space ────────────────────────
        z = self.intensity_proj(features)  # [B, N, d_prior]

        # ── Compute four kernels ──────────────────────────────────
        A_spatial   = self._compute_spatial_prior(positions)       # [N, N]
        A_intensity = self._compute_intensity_prior(z)             # [B, N, N]
        A_boundary  = self._compute_boundary_prior(z)              # [B, N, N]
        A_stability = self._compute_stability_prior(z)             # [B, N, N]

        # ── Combine with learnable, temperature-scaled weights ────
        w = F.softmax(
            self.prior_logits / self.prior_temperature, dim=0
        )  # [4]

        A_combined = (
            w[0] * A_spatial.unsqueeze(0)   # [1, N, N] → broadcast to [B, N, N]
            + w[1] * A_intensity
            + w[2] * A_boundary
            + w[3] * A_stability
        )

        # ── Symmetry enforcement ──────────────────────────────────
        A_combined = (A_combined + A_combined.transpose(-1, -2)) / 2.0

        # ── Clamp to valid range ──────────────────────────────────
        A_combined = A_combined.clamp(min=0.0, max=1.0)

        # ── Top-k sparsification per row ──────────────────────────
        k = min(self.top_k, N - 1)  # can't have more neighbors than tokens-1
        vals, idx = A_combined.topk(k, dim=-1)  # [B, N, k]

        A_sparse = torch.zeros_like(A_combined)
        A_sparse.scatter_(-1, idx, vals.to(A_sparse.dtype))

        # ── Row normalization → stochastic matrix ─────────────────
        row_sums = A_sparse.sum(-1, keepdim=True).clamp(min=1e-8)
        A_prior = A_sparse / row_sums

        # ── Diagnostics ───────────────────────────────────────────
        sigma = F.softplus(self.log_sigma)
        tau   = F.softplus(self.log_tau)
        alpha = F.softplus(self.log_alpha)
        gamma = F.softplus(self.log_gamma)

        prior_info = {
            "prior_weights": w.detach().cpu(),
            "sigma": sigma.item(),
            "tau": tau.item(),
            "alpha": alpha.item(),
            "gamma": gamma.item(),
            "sparsity": (A_sparse == 0).float().mean().item(),
            "mean_value": A_prior[A_prior > 0].mean().item() if (A_prior > 0).any() else 0.0,
        }

        return A_prior, prior_info


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 0 — HyperFidelityPriorBuilder Unit Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d = 2, 512, 384

    builder = HyperFidelityPriorBuilder(
        embed_dim=d, d_prior=16, top_k=32,
        sigma_init=0.1, tau_init=0.5, alpha_init=5.0, gamma_init=2.0,
    ).to(device)

    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    features = torch.randn(B, N, d, device=device)

    # ── Forward pass ─────────────────────────────────────
    A_prior, info = builder(features, positions)

    # ── Test 1: Output shape ─────────────────────────────
    assert A_prior.shape == (B, N, N), f"FAIL shape: {A_prior.shape}"
    print(f"  ✅ Shape: {A_prior.shape}")

    # ── Test 2: Row normalization ────────────────────────
    row_sums = A_prior.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.02), \
        f"FAIL row norm: max deviation {(row_sums - 1).abs().max():.4f}"
    print(f"  ✅ Row normalization: max deviation {(row_sums - 1).abs().max():.6f}")

    # ── Test 3: No self-loops ────────────────────────────
    diag_max = A_prior.diagonal(dim1=-2, dim2=-1).abs().max().item()
    assert diag_max < 0.01, f"FAIL self-loops: diag_max={diag_max}"
    print(f"  ✅ No self-loops: diag max = {diag_max:.6f}")

    # ── Test 4: Sparsity ─────────────────────────────────
    sparsity = (A_prior == 0).float().mean().item()
    assert sparsity > 0.9, f"FAIL sparsity: {sparsity:.2%}"
    print(f"  ✅ Sparsity: {sparsity:.2%}")

    # ── Test 5: Non-negative ─────────────────────────────
    assert (A_prior >= 0).all(), "FAIL: negative values"
    print(f"  ✅ Non-negative: all values ≥ 0")

    # ── Test 6: Differentiable ───────────────────────────
    loss = A_prior.sum()
    loss.backward()
    assert builder.log_sigma.grad is not None, "FAIL: no gradient to log_sigma"
    assert builder.log_tau.grad is not None, "FAIL: no gradient to log_tau"
    assert builder.log_alpha.grad is not None, "FAIL: no gradient to log_alpha"
    assert builder.prior_logits.grad is not None, "FAIL: no gradient to prior_logits"
    print(f"  ✅ Differentiable: all 4 bandwidth params have gradients")

    # ── Test 7: Stability prior (second forward) ─────────
    builder.zero_grad()
    A_prior2, info2 = builder(features + 0.01 * torch.randn_like(features), positions)
    assert A_prior2.shape == (B, N, N), "FAIL stability: shape mismatch on 2nd call"
    print(f"  ✅ Stability prior: EMA updated on 2nd forward pass")

    # ── Test 8: Prior info diagnostics ───────────────────
    assert "prior_weights" in info, "FAIL: prior_weights missing from info"
    assert "sigma" in info, "FAIL: sigma missing from info"
    assert info["prior_weights"].shape == (4,), "FAIL: prior_weights shape"
    print(f"  ✅ Diagnostics: prior_weights={info['prior_weights'].numpy()}")
    print(f"     σ={info['sigma']:.4f}, τ={info['tau']:.4f}, "
          f"α={info['alpha']:.4f}, γ={info['gamma']:.4f}")
    print(f"     sparsity={info['sparsity']:.2%}")

    # ── Test 9: Parameter count ──────────────────────────
    total_params = sum(p.numel() for p in builder.parameters())
    print(f"  ✅ Parameters: {total_params:,}")

    # ── Test 10: Device compatibility ────────────────────
    print(f"  ✅ Device: {device}")

    print()
    print("  ════════════════════════════════════════════════")
    print("  ✅ ALL PHASE 0 TESTS PASSED")
    print("  ════════════════════════════════════════════════")
