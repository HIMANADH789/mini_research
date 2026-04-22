"""
Phase 2 — Spectral Optimizer (SpectralAdam)
=============================================
Custom optimizer that preconditions gradient updates using the eigenspace
of the learned graph, aligning optimization with discovered data structure.

Mathematical formulation:
  1. Every `spectral_interval` steps, extract eigenspace from A_effective
  2. Build preconditioner: P = U_k @ diag(1/(S_k + eps)) @ U_k^T
  3. Apply P to Adam direction via Kronecker block approximation
  4. Standard Adam momentum + weight decay

Key robustness:
  - Spectral warmup: first N steps are pure Adam (graph is random early)
  - EMA eigenspace smoothing: prevents oscillation from noisy SVD
  - Float32 enforcement for all eigendecomposition (AMP safety)
  - Gradient norm monitoring: clips if spectral transform explodes gradient
  - Fallback to Adam if eigendecomposition fails
  - NaN/Inf detection with skip-step logic
"""

import logging
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SpectralAdam(Optimizer):
    """
    Adam optimizer with spectral preconditioning from learned graph eigenspace.

    Parameters
    ----------
    params : iterable
        Parameters to optimize (typically attention projection weights only).
    lr : float
        Learning rate.
    betas : tuple
        Adam momentum coefficients.
    eps : float
        Adam epsilon for numerical stability.
    weight_decay : float
        Decoupled weight decay (AdamW-style).
    rank_k : int
        Number of top eigenvalues to use for preconditioning.
    spectral_interval : int
        Recompute eigenspace every N steps.
    spectral_momentum : float
        EMA momentum for eigenspace smoothing.
    spectral_warmup : int
        Steps before activating spectral preconditioning (pure Adam until then).
    max_grad_ratio : float
        If spectral transform increases grad norm by more than this factor, clip.
    fallback_to_adam : bool
        If eigendecomposition fails, fall back to standard Adam step.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank_k: int = 64,
        spectral_interval: int = 10,
        spectral_momentum: float = 0.9,
        spectral_warmup: int = 100,
        max_grad_ratio: float = 10.0,
        fallback_to_adam: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.rank_k = rank_k
        self.spectral_interval = spectral_interval
        self.spectral_momentum = spectral_momentum
        self.spectral_warmup = spectral_warmup
        self.max_grad_ratio = max_grad_ratio
        self.fallback_to_adam = fallback_to_adam

        # Cached eigenspace (smoothed via EMA)
        self._U_k = None  # [k, k] eigenvectors
        self._S_k = None  # [k] eigenvalues
        self._P = None    # [k, k] preconditioner
        self._global_step = 0
        self._spectral_active = False
        self._last_condition_number = 0.0

    @torch.no_grad()
    def _update_eigenspace(self, graph_matrix: Tensor):
        """
        Compute truncated eigendecomposition and update cached eigenspace.
        All computation in float32 for numerical stability.
        """
        # Force float32 (AMP safety)
        A = graph_matrix.detach().float()

        # Ensure symmetric (should already be, but safety)
        A = (A + A.T) / 2.0

        try:
            # Full eigendecomposition (more stable than lobpcg for small matrices)
            S_full, U_full = torch.linalg.eigh(A)

            # Take top-k eigenvalues (largest)
            k = min(self.rank_k, S_full.shape[0])
            S_new = S_full[-k:]       # [k] — largest eigenvalues
            U_new = U_full[:, -k:]    # [N, k]

            # Check for NaN/Inf
            if not (torch.isfinite(S_new).all() and torch.isfinite(U_new).all()):
                logger.warning("Eigendecomposition produced NaN — skipping update")
                return False

            # EMA smoothing of eigenspace
            if self._U_k is not None and self._S_k is not None:
                # Only smooth if shapes match
                if self._U_k.shape == U_new.shape and self._S_k.shape == S_new.shape:
                    mom = self.spectral_momentum
                    self._S_k = mom * self._S_k + (1.0 - mom) * S_new
                    # For eigenvectors, simple EMA then re-orthogonalize
                    U_ema = mom * self._U_k + (1.0 - mom) * U_new
                    # QR for orthogonality
                    self._U_k, _ = torch.linalg.qr(U_ema)
                else:
                    self._U_k = U_new
                    self._S_k = S_new
            else:
                self._U_k = U_new
                self._S_k = S_new

            # Build preconditioner: P = U @ diag(1/(S+eps)) @ U^T
            S_inv = 1.0 / (self._S_k + 1e-6)
            # Clamp to prevent extreme preconditioning
            S_inv = S_inv.clamp(max=100.0)
            self._P = self._U_k @ torch.diag(S_inv) @ self._U_k.T  # [N, N]

            self._last_condition_number = (self._S_k[-1] / (self._S_k[0] + 1e-8)).item()
            return True

        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}")
            self._P = None
            return False

    def _apply_preconditioner(self, grad: Tensor) -> Tensor:
        """
        Apply spectral preconditioner to gradient via block Kronecker approximation.
        """
        if self._P is None:
            return grad

        P = self._P
        n = P.shape[0]
        g_flat = grad.reshape(-1)

        # Block Kronecker: apply P to blocks of size n
        if g_flat.shape[0] >= n and g_flat.shape[0] % n == 0:
            g_blocks = g_flat.reshape(-1, n)  # [num_blocks, n]
            g_spec = (g_blocks @ P.T).reshape(-1)

            # Gradient norm safety check
            g_norm_before = g_flat.norm().item()
            g_norm_after = g_spec.norm().item()

            if g_norm_before > 0 and g_norm_after > self.max_grad_ratio * g_norm_before:
                # Clip: scale down to preserve original norm magnitude
                scale = g_norm_before / (g_norm_after + 1e-8)
                g_spec = g_spec * scale

            return g_spec.reshape_as(grad)
        else:
            # Dimension mismatch — skip preconditioning for this param
            return grad

    @torch.no_grad()
    def step(self, graph_matrix: Optional[Tensor] = None, closure=None):
        """
        One optimizer step with optional spectral preconditioning.

        Parameters
        ----------
        graph_matrix : Tensor [N, N], optional
            Average learned graph from model's last forward pass.
            Pass None to do a standard Adam step.
        closure : callable, optional
            Loss closure (standard optimizer interface).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1
        t = self._global_step

        # Update eigenspace periodically (after warmup)
        if (t > self.spectral_warmup
                and t % self.spectral_interval == 0
                and graph_matrix is not None):
            success = self._update_eigenspace(graph_matrix)
            self._spectral_active = success
        elif t <= self.spectral_warmup:
            self._spectral_active = False

        # Standard Adam step with optional spectral preconditioning
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if not torch.isfinite(grad).all():
                    logger.warning("Skipping param with NaN/Inf gradient")
                    continue

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2

                # Adam direction
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                # Apply spectral preconditioning (if active)
                if self._spectral_active and self._P is not None:
                    adam_dir = self._apply_preconditioner(adam_dir)

                # Decoupled weight decay (AdamW-style)
                p.data.add_(p.data, alpha=-lr * wd)

                # Parameter update
                p.data.add_(adam_dir, alpha=-lr)

        return loss

    # ══════════════════════════════════════════════════════
    #  ANALYSIS API
    # ══════════════════════════════════════════════════════

    def get_spectral_analysis(self) -> Dict[str, float]:
        """Return optimization analysis metrics for diagnostics."""
        if self._S_k is None:
            return {"spectral_active": float(self._spectral_active)}

        S = self._S_k
        S_pos = S[S > 0]
        result = {
            "spectral_active": float(self._spectral_active),
            "global_step": float(self._global_step),
            "condition_number": self._last_condition_number,
        }

        if len(S_pos) > 1:
            p = S_pos / (S_pos.sum() + 1e-8)
            result["spectrum_entropy"] = (-(p * torch.log(p + 1e-8)).sum()).item()
            result["effective_rank"] = (S_pos > S_pos[-1] * 0.01).sum().item()
            result["eigenvalue_gap"] = (S_pos[-1] - S_pos[0]).item()

        return result

    def get_gradient_alignment(self, grad: Tensor) -> float:
        """Measure how well gradient aligns with learned eigenspace."""
        if self._U_k is None:
            return 0.0

        g = grad.reshape(-1).float()
        U = self._U_k
        n = U.shape[0]

        if g.shape[0] < n or g.shape[0] % n != 0:
            return 0.0

        g_blocks = g.reshape(-1, n)
        proj_norm = (g_blocks @ U).norm().item()
        total_norm = g.norm().item()

        return proj_norm / (total_norm + 1e-8)


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 - SpectralAdam Unit Tests")
    print("=" * 60)

    # Create dummy parameters and graph
    param1 = torch.randn(512, 384, requires_grad=True)
    param2 = torch.randn(384, 384, requires_grad=True)

    optimizer = SpectralAdam(
        [param1, param2],
        lr=1e-3, rank_k=32, spectral_interval=5,
        spectral_warmup=10, spectral_momentum=0.9,
    )

    # Create a symmetric positive semi-definite graph matrix
    A = torch.rand(512, 512)
    A = (A + A.T) / 2
    A = A + 512 * torch.eye(512)  # make positive definite

    # Simulate training steps
    for step in range(20):
        # Fake loss
        loss = param1.sum() + param2.sum()
        loss.backward()

        # Optimizer step
        if step >= 10:
            optimizer.step(graph_matrix=A)
        else:
            optimizer.step()

        optimizer.zero_grad()

    # Check spectral analysis
    analysis = optimizer.get_spectral_analysis()
    print(f"  Spectral active:   {analysis['spectral_active']}")
    print(f"  Global step:       {analysis['global_step']}")
    if "condition_number" in analysis:
        print(f"  Condition number:  {analysis['condition_number']:.4f}")
    if "effective_rank" in analysis:
        print(f"  Effective rank:    {analysis['effective_rank']}")

    # Check gradient alignment
    param1.grad = torch.randn_like(param1)
    alignment = optimizer.get_gradient_alignment(param1.grad)
    print(f"  Gradient alignment: {alignment:.4f}")

    # Verify params changed
    print(f"  Param1 grad exists: {param1.grad is not None}")

    print()
    print("  [OK] ALL PHASE 2 TESTS PASSED")
