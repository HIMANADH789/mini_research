# SABiT — FINAL Implementation Plan

> [!CAUTION]
> **FINAL DRAFT — No further revisions.** This is the version we code starting next chat.

---

## STATUS

| Component | Status |
|---|---|
| Pipeline infrastructure (17 requirements) | ✅ Done |
| Trainer v5 (bi-level, spectral hooks, grad accum) | ✅ Done |
| Comparison & ablation utilities | ✅ Done |
| SABiT registered in build.py | ✅ Done |
| **SABiT model (this plan)** | 🔴 **Starts next chat** |

**Updated SOTA targets** (based on 2024-2025 leaderboard research):

| Method | WT Dice | TC Dice | ET Dice |
|---|---|---|---|
| nnU-Net ensemble (2023) | 92.3 | 87.8 | 88.2 |
| Swin UNETR (2022) | 92.1 | 88.5 | 83.1 |
| MedNeXt (2024) | 92.5 | 88.1 | 88.5 |
| **SABiT Target** | **≥93.0** | **≥89.0** | **≥88.5** |

---

## FILE LAYOUT

```
src/boilerplates/models/sabit/
├── __init__.py                    ← Exports: SABiT
├── sabit_model.py                 ← Main encoder-decoder assembly     (~400 lines)
├── patch_embed.py                 ← 3D patch tokenization             (~80 lines)
├── prior_builder.py               ← Phase 0: Hyper-fidelity priors    (~250 lines)
├── structure_net.py               ← Phase 1A: Evidential graph learner (~300 lines)
├── graph_attention.py             ← Phase 1B: Gated structure-biased MHSA (~270 lines)
├── transformer_block.py           ← Phase 1C: SABiTBlock assembly     (~140 lines)
├── decoder.py                     ← UNet decoder + deep supervision   (~160 lines)
└── spectral_optimizer.py          ← Phase 2: Spectral optimizer       (~350 lines)

src/boilerplates/losses/
└── sabit_loss.py                  ← Phase 3: Multi-objective loss     (~250 lines)

Total: ~2,200 lines of novel research code
```

---

## ═══════════════════════════════════════════════════
## PHASE 0 — Hyper-Fidelity Prior Builder
## ═══════════════════════════════════════════════════

**File**: `prior_builder.py`
**Class**: `HyperFidelityPriorBuilder(nn.Module)`

### Purpose

Constructs a learnable, sparse structural prior `A_prior ∈ ℝ^{B×N×N}` from four domain-specific kernels encoding anatomical knowledge. This prior anchors the graph throughout all transformer layers.

### Constructor Signature

```python
def __init__(self,
             embed_dim: int,             # 384 at bottleneck stage
             d_prior: int = 16,          # feature projection dim
             top_k: int = 32,            # sparsification: edges per node
             sigma_init: float = 0.1,    # spatial bandwidth
             tau_init: float = 0.5,      # intensity bandwidth
             alpha_init: float = 5.0,    # boundary sharpness
             gamma_init: float = 2.0,    # stability decay rate
             ema_momentum: float = 0.99, # stability prior EMA
             prior_temperature: float = 1.0,  # softmax temp for weight mixing
             ):
```

### All Parameters

```python
# Learnable kernel bandwidths (stored in log-space, applied via softplus)
self.log_sigma = nn.Parameter(inverse_softplus(sigma_init))
self.log_tau   = nn.Parameter(inverse_softplus(tau_init))
self.log_alpha = nn.Parameter(inverse_softplus(alpha_init))
self.log_gamma = nn.Parameter(inverse_softplus(gamma_init))

# Prior mixing weights — 4 kernels
self.prior_logits = nn.Parameter(torch.tensor([0.4, 0.4, 0.15, 0.05]))
self.prior_temperature = prior_temperature  # controls sharpness of weight distribution

# Intensity feature projection — 2-layer MLP for rich fingerprinting
self.intensity_proj = nn.Sequential(
    nn.Linear(embed_dim, d_prior * 2),
    nn.GELU(),
    nn.Linear(d_prior * 2, d_prior),
)

# Boundary detector — learned MLP on pairwise feature differences
self.boundary_mlp = nn.Sequential(
    nn.Linear(d_prior, d_prior),
    nn.GELU(),
    nn.Linear(d_prior, 1),
)

# Stability prior EMA buffer (not a parameter — no gradients)
self.register_buffer('feature_ema', None)
self.ema_momentum = ema_momentum
self.top_k = top_k
```

### Forward: `forward(features, positions) → A_prior, prior_info`

#### Input Shapes
- `features: [B, N, d]` — token embeddings at current encoder stage
- `positions: [N, 3]` — normalized 3D coordinates from `build_position_grid()`

#### Computation Flow

**Step 1 — Spatial Prior** `A_spatial [N, N]`:

```
Equation: A_spatial(i,j) = exp(−||p_i − p_j||² / 2σ²)

σ = softplus(self.log_sigma)
dist_sq = torch.cdist(positions, positions).pow(2)  # [N, N]
A_spatial = torch.exp(-dist_sq / (2 * σ² + 1e-8))

# Remove self-loops (diagonal = 0)
A_spatial.fill_diagonal_(0)
```

**Caching**: `A_spatial` is position-only → compute once per resolution, cache with `register_buffer`.

**Step 2 — Intensity Prior** `A_intensity [B, N, N]`:

```
Equation: z = MLP(h), A_intensity(i,j) = exp(−||z_i − z_j||² / 2τ²)

z = self.intensity_proj(features)                    # [B, N, d_prior=16]
τ = softplus(self.log_tau)
feat_dist_sq = torch.cdist(z, z).pow(2)              # [B, N, N]
A_intensity = torch.exp(-feat_dist_sq / (2 * τ² + 1e-8))
A_intensity = A_intensity * (1 - torch.eye(N, device=...))  # remove self-loops
```

**Step 3 — Boundary Prior** `A_boundary [B, N, N]`:

```
Equation: Δz_ij = |z_i − z_j|, b_ij = MLP(Δz_ij), A_boundary = 1 − sigmoid(α · b_ij)

# Pairwise absolute differences in projected feature space
# Memory-efficient: compute using broadcasting, NOT materializing [B,N,N,d]
# Instead: use the already-computed feat_dist or chunked computation

α = softplus(self.log_alpha)

# Efficient chunked boundary computation (avoids OOM for large N):
A_boundary = torch.zeros(B, N, N, device=features.device)
chunk_size = min(N, 128)  # process 128 rows at a time
for i_start in range(0, N, chunk_size):
    i_end = min(i_start + chunk_size, N)
    z_i = z[:, i_start:i_end].unsqueeze(2)  # [B, chunk, 1, d_prior]
    z_j = z.unsqueeze(1)                      # [B, 1, N, d_prior]
    delta = (z_i - z_j).abs()                 # [B, chunk, N, d_prior]
    b = self.boundary_mlp(delta).squeeze(-1)  # [B, chunk, N]
    A_boundary[:, i_start:i_end, :] = 1.0 - torch.sigmoid(α * b)
```

**Why chunked**: For N=512, the full `[B, 512, 512, 16]` tensor = 32MB per batch. Chunking keeps peak allocation at ~8MB.

**Step 4 — Stability Prior** `A_stability [B, N, N]`:

```
Equation: s_i = exp(−γ · ||z_i − z̄_i||²), A_stability(i,j) = s_i · s_j

γ = softplus(self.log_gamma)

# Update EMA of features (momentum buffer)
if self.feature_ema is None:
    self.feature_ema = z.detach().clone()
else:
    self.feature_ema = self.ema_momentum * self.feature_ema + (1 - self.ema_momentum) * z.detach()

# Per-token stability score
drift = (z - self.feature_ema.detach()).pow(2).sum(-1)  # [B, N]
stability = torch.exp(-γ * drift)                        # [B, N] ∈ (0, 1]

# Outer product → pairwise stability
A_stability = stability.unsqueeze(-1) * stability.unsqueeze(-2)  # [B, N, N]
A_stability = A_stability * (1 - torch.eye(N, device=...))      # remove self-loops
```

**Step 5 — Combined Prior**:

```
Equation: w = softmax(logits / T), A = Σ w_k · A_k

w = F.softmax(self.prior_logits / self.prior_temperature, dim=0)  # [4]

A_combined = (w[0] * A_spatial.unsqueeze(0)  # broadcast [1,N,N] → [B,N,N]
            + w[1] * A_intensity
            + w[2] * A_boundary
            + w[3] * A_stability)

# Symmetry enforcement
A_combined = (A_combined + A_combined.transpose(-1, -2)) / 2.0

# Clamp to [0, 1] before sparsification
A_combined = A_combined.clamp(min=0.0, max=1.0)
```

**Step 6 — Top-k Sparsification + Row Normalization**:

```
vals, idx = A_combined.topk(self.top_k, dim=-1)  # [B, N, top_k]
A_sparse = torch.zeros_like(A_combined)
A_sparse.scatter_(-1, idx, vals)

# Row-normalize → stochastic matrix
row_sums = A_sparse.sum(-1, keepdim=True).clamp(min=1e-8)
A_prior = A_sparse / row_sums
```

#### Return Value

```python
prior_info = {
    'prior_weights': w.detach(),
    'sigma': σ.item(),
    'tau': τ.item(),
    'alpha': α.item(),
    'gamma': γ.item(),
    'sparsity': (A_sparse == 0).float().mean().item(),
    'mean_stability': stability.mean().item(),
}
return A_prior, prior_info
```

#### Static Helper

```python
@staticmethod
def build_position_grid(D: int, H: int, W: int) -> Tensor:
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, D),
        torch.linspace(0, 1, H),
        torch.linspace(0, 1, W),
        indexing='ij'
    ), dim=-1)
    return coords.reshape(-1, 3)  # [N, 3]
```

#### Robustness Features

1. **Self-loop removal** on every kernel — prevents trivial self-attention bias
2. **Chunked boundary computation** — prevents OOM for large N
3. **EMA feature buffer** for stability prior — `register_buffer` ensures proper device handling and checkpoint compatibility
4. **Temperature-controlled prior mixing** — allows sharpening weight distribution during ablation experiments
5. **Clamped spatial prior cache** — computed once per resolution, reused across forward passes
6. **Detached EMA** — stability prior doesn't backpropagate through the momentum history

#### Unit Test Assertions

```python
B, N, d = 2, 512, 384
builder = HyperFidelityPriorBuilder(embed_dim=d, d_prior=16, top_k=32)
pos = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8)
feats = torch.randn(B, N, d)

A, info = builder(feats, pos)
assert A.shape == (B, N, N),              "Shape mismatch"
assert torch.allclose(A.sum(-1), torch.ones(B,N), atol=0.02), "Not row-normalized"
assert A.diagonal(dim1=-2, dim2=-1).abs().max() < 0.01,       "Self-loops present"
assert (A == 0).float().mean() > 0.9,     "Not sparse enough"
assert (A >= 0).all(),                     "Negative values"
assert A.requires_grad,                    "Not differentiable"

# Test backward
loss = A.sum()
loss.backward()
assert builder.log_sigma.grad is not None, "No gradient to sigma"
print("✅ Phase 0 ALL TESTS PASSED")
```

---

## ═══════════════════════════════════════════════════
## PHASE 1A — Evidential Graph Learner (StructureNet)
## ═══════════════════════════════════════════════════

**File**: `structure_net.py`
**Class**: `EvidentialGraphLearner(nn.Module)`

### Constructor

```python
def __init__(self,
             embed_dim: int,
             d_pair: int = 64,          # pair feature dimension
             n_evidence: int = 4,       # Dirichlet evidence classes
             top_k: int = 32,           # edges to score per node
             mu_scale_init: float = 0.01,  # initial residual scale (near zero)
             dropout: float = 0.1,
             ):
```

### Key Robustness Features

1. **Residual scaling initialization** — `μ` output scaled by `mu_scale_init=0.01` so initial graph ≈ prior:
   ```python
   self.mu_scale = nn.Parameter(torch.tensor(mu_scale_init))
   μ = tanh(mu_raw) * softplus(self.mu_scale)  # starts near 0, grows during training
   ```

2. **Edge dropout** during training — randomly drop 10% of edges for regularization:
   ```python
   if self.training:
       edge_mask = torch.bernoulli(torch.full_like(A_gated, 1 - self.edge_drop_rate))
       A_gated = A_gated * edge_mask
   ```

3. **Dirichlet numerical stability** — clamp α before lgamma to prevent overflow:
   ```python
   α = softplus(alpha_head(pair_feat)) + 1.0  # α > 1
   α = α.clamp(min=1.01, max=1000.0)          # prevent lgamma(α) overflow
   ```

4. **Float32 enforcement** for KL computation — even under AMP:
   ```python
   @staticmethod
   @torch.amp.custom_fwd(cast_inputs=torch.float32)
   def evidential_kl_loss(alpha):
       # All lgamma, digamma ops in float32
       ...
   ```

5. **Efficient sparse gather** — only score edges with prior support (top-k from A_prior):
   ```python
   _, neighbor_idx = A_prior.topk(top_k, dim=-1)  # [B, N, k]
   h_j = torch.gather(h.unsqueeze(1).expand(-1,N,-1,-1), 2,
                       neighbor_idx.unsqueeze(-1).expand(-1,-1,-1,d))
   ```

### Forward: `forward(h, positions, A_prior) → A_effective, uncertainty, ev_info`

**Detailed flow**:

```
1. SPARSE PAIR FEATURES [B, N, k, d_pair]:
   h_i = h.unsqueeze(2).expand(-1,-1,k,-1)         # [B,N,k,d]
   h_j = gather(h, neighbor_idx)                     # [B,N,k,d]
   Δpos = gather(positions, neighbor_idx) - positions # [B,N,k,3]
   pair = cat([h_i, h_j, h_i⊙h_j, Δpos], dim=-1)   # [B,N,k,2d+d+3]
   pair_feat = pair_proj(pair)                        # [B,N,k,d_pair]

2. EVIDENTIAL PREDICTIONS:
   μ = tanh(mu_head(pair_feat).squeeze(-1)) * softplus(mu_scale)  # [B,N,k]
   α = softplus(alpha_head(pair_feat)) + 1.0                      # [B,N,k,C]
   α = α.clamp(1.01, 1000.0)

3. UNCERTAINTY [B, N, k]:
   S = α.sum(-1)             # total evidence
   u = C / S                  # uncertainty ∈ (0, 1)
   u = u.clamp(max=0.999)    # never exactly 1 (would zero out edge)

4. EFFECTIVE GRAPH:
   A_prior_vals = gather(A_prior, neighbor_idx)       # [B,N,k]
   A_raw = A_prior_vals + μ                            # residual learning
   A_gated = A_raw * (1 - u)                           # uncertainty suppression
   
   # Edge dropout (training only)
   if training: A_gated *= bernoulli_mask
   
   # Scatter to dense [B,N,N]
   A_effective = zeros(B,N,N)
   A_effective.scatter_(-1, neighbor_idx, A_gated)
   
   # Symmetrize + ReLU (non-negative) + row-normalize
   A_effective = (A_effective + A_effective.T) / 2
   A_effective = relu(A_effective)
   A_effective = A_effective / (A_effective.sum(-1, keepdim=True) + 1e-8)

5. RETURN:
   uncertainty_map = scatter(u, neighbor_idx)          # [B,N,N]
   ev_info = {α, mean_uncertainty, evidence_strength}
```

---

## ═══════════════════════════════════════════════════
## PHASE 1B — Gated Structure-Biased Attention
## ═══════════════════════════════════════════════════

**File**: `graph_attention.py`
**Class**: `GraphBiasedAttention(nn.Module)`

### Constructor

```python
def __init__(self,
             dim: int,
             num_heads: int,
             qkv_bias: bool = True,
             attn_drop: float = 0.0,
             proj_drop: float = 0.0,
             bias_scale_init: float = 1.0,
             ):
```

### Parameters

```python
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
self.attn_drop = nn.Dropout(attn_drop)
self.proj = nn.Linear(dim, dim)
self.proj_drop = nn.Dropout(proj_drop)

# Gate network: per-edge confidence in learned vs prior
self.gate_net = nn.Sequential(
    nn.Linear(2, 32),
    nn.GELU(),
    nn.Linear(32, 1),
)

# Bias projection: graph value → per-head attention bias
self.bias_proj = nn.Sequential(
    nn.Linear(1, num_heads),
)

# Per-head trust scale (each head independently learns graph reliance)
self.bias_scale = nn.Parameter(torch.full((num_heads,), bias_scale_init))
```

### Forward

```python
def forward(self, x, A_effective, A_prior):
    B, N, d = x.shape
    heads = self.num_heads
    d_h = d // heads
    
    # Standard QKV
    qkv = self.qkv(x).reshape(B, N, 3, heads, d_h).permute(2,0,3,1,4)
    Q, K, V = qkv.unbind(0)                      # [B, heads, N, d_h]
    attn_logits = (Q @ K.transpose(-2,-1)) * (d_h ** -0.5)  # [B, heads, N, N]
    
    # Gated structural bias
    gate_in = torch.stack([A_effective, A_prior], dim=-1)    # [B, N, N, 2]
    g = torch.sigmoid(self.gate_net(gate_in))                # [B, N, N, 1]
    A_blended = g * A_effective.unsqueeze(-1) + (1-g) * A_prior.unsqueeze(-1)  # [B,N,N,1]
    
    bias = self.bias_proj(A_blended)              # [B, N, N, heads]
    bias = bias.permute(0, 3, 1, 2)               # [B, heads, N, N]
    bias = F.softplus(bias) * self.bias_scale.view(1, -1, 1, 1)
    
    # Biased attention
    attn = F.softmax(attn_logits + bias, dim=-1)
    attn = self.attn_drop(attn)
    
    out = (attn @ V).transpose(1, 2).reshape(B, N, d)
    return self.proj_drop(self.proj(out))
```

### Robustness

1. **Softplus bias** — always non-negative, can only boost attention (never suppress important semantic connections)
2. **Per-head scaling** — heads learn independent graph trust levels
3. **NaN guard** — clamp attn_logits before softmax: `attn_logits = attn_logits.clamp(-50, 50)`

---

## ═══════════════════════════════════════════════════
## PHASE 1C — SABiTBlock Assembly
## ═══════════════════════════════════════════════════

**File**: `transformer_block.py`
**Class**: `SABiTBlock(nn.Module)`

### Architecture

```
x → EvidentialGraphLearner(x, pos, A_prior) → A_effective, uncertainty, ev_info
  → LayerNorm → GraphBiasedAttention(·, A_eff, A_prior) → DropPath → ×layer_scale → +Residual
  → LayerNorm → FFN → DropPath → ×layer_scale → +Residual
```

### Key Enhancement: **Layer Scale** (from CaiT/DeiT-III)

```python
# Initialize layer scale to small value → residual starts near identity
self.layer_scale_1 = nn.Parameter(torch.full((dim,), 1e-4))
self.layer_scale_2 = nn.Parameter(torch.full((dim,), 1e-4))

# In forward:
x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x), A_eff, A_prior))
x = x + self.drop_path(self.layer_scale_2 * self.ffn(self.norm2(x)))
```

**Why**: Layer scale prevents early training instability in deep transformer stacks. The 1e-4 initialization means residual contributions start near-zero and gradually increase, giving the model a stable warmup.

### Return

```python
return x, A_effective, uncertainty, ev_info
```

---

## ═══════════════════════════════════════════════════
## PHASE 1D — Full Model Assembly
## ═══════════════════════════════════════════════════

**File**: `sabit_model.py`, `patch_embed.py`, `decoder.py`
**Class**: `SABiT(nn.Module)`

### Architecture

```
Encoder:
  Stages 0-2: Swin Transformer blocks (REUSE from swinunetr.py)
  Stage 3:    SABiT blocks with Prior Builder + StructureNet + GraphBiasedAttn

  Input [B,4,128³] → PatchEmbed → Stage0(64³,48) → Stage1(32³,96) → Stage2(16³,192)
    → PatchMerge → Stage3(8³,384) [SABiT: 2 blocks, 512 tokens, full graph attention]

Decoder:
  ConvTranspose3d upsampling + skip concatenation + residual conv blocks
  Deep supervision: ds_head from dec4 and dec3

Prior Builder:
  Created once at SABiT init
  Called at Stage 3 resolution (8×8×8 = 512 tokens)
  position_grid built once, registered as buffer
```

### Key Design Features

1. **Reuse Swin blocks for stages 0-2** — import directly from `swinunetr.py`:
   ```python
   from src.boilerplates.models.swinunetr import (
       PatchEmbed3D, PatchMerging3D, BasicLayer3D,
       EncoderBlock, DecoderBlock, DropPath,
   )
   ```

2. **SABiT blocks only at bottleneck** (Stage 3, N=512 tokens):
   - Computationally feasible for full N×N graph attention
   - Captures global structural relationships between brain regions
   - Isolates contribution cleanly for ablation

3. **Graph caching** for spectral optimizer and diagnostics:
   ```python
   # During forward, cache for post-forward use by optimizer/trainer
   self._cached_A_prior = A_prior.detach()
   self._cached_A_effective = A_effective.detach()
   self._cached_uncertainty = uncertainty.detach()
   ```

4. **AMP safety for eigendecomposition**:
   ```python
   @torch.amp.custom_fwd(cast_inputs=torch.float32)
   def _compute_eigenvalues(self, A):
       """Always float32 for numerical stability."""
       S, U = torch.linalg.eigh(A)
       return S, U
   ```

5. **Gradient checkpointing** for Swin stages:
   ```python
   def enable_gradient_checkpointing(self):
       self._use_checkpoint = True
   # In forward:
   if self._use_checkpoint:
       s0 = torch.utils.checkpoint.checkpoint(self.stage0, x_embed, use_reentrant=False)
   else:
       s0 = self.stage0(x_embed)
   ```

### Model API (consumed by Trainer v5)

| Method | Purpose |
|---|---|
| `get_optimizer_groups(lr)` | Returns `{'primary': [...], 'auxiliary': [...]}` for bi-level optimizer |
| `get_spectral_metrics()` | Returns condition_number, eigenvalue_gap, spectrum_entropy, effective_rank, graph_sparsity, mean_uncertainty |
| `get_tensor_artifacts()` | Returns A_prior, A_learned, uncertainty_map, eigenvalues tensors |
| `get_auxiliary_losses()` | Returns `{'prior': (L_prior, 0.1), 'smooth': (L_smooth, 0.01), ...}` |
| `enable_gradient_checkpointing()` | Enables checkpoint for stages 0-2 to save ~40% VRAM |

---

## ═══════════════════════════════════════════════════
## PHASE 2 — Spectral Optimizer (SpectralAdam)
## ═══════════════════════════════════════════════════

**File**: `spectral_optimizer.py`
**Class**: `SpectralAdam(torch.optim.Optimizer)`

### Constructor

```python
def __init__(self, params,
             lr=1e-3,
             betas=(0.9, 0.999),
             eps=1e-8,
             weight_decay=0.01,
             rank_k=64,
             spectral_interval=10,       # recompute every N steps
             spectral_momentum=0.9,      # EMA smoothing of eigenspace
             spectral_warmup=100,        # steps before activating spectral preconditioning
             fallback_to_adam=True,
             ):
```

### Key Robustness Features

1. **Spectral warmup** — first 100 steps are pure Adam (graph structure is random early on)

2. **EMA eigenspace smoothing** — prevents oscillation from noisy SVD:
   ```python
   U_k = mom * U_k_old + (1-mom) * U_k_new
   S_k = mom * S_k_old + (1-mom) * S_k_new
   ```

3. **Fallback detection** — if SVD produces NaN/Inf, skip spectral step and log warning:
   ```python
   if not torch.isfinite(S).all():
       logger.warning("Spectral decomposition produced NaN — falling back to Adam")
       return standard_adam_step()
   ```

4. **Float32 enforcement** for all eigendecomposition:
   ```python
   with torch.amp.autocast(device_type='cuda', enabled=False):
       A_f32 = graph_matrix.float()
       S, U = torch.linalg.eigh(A_f32)
   ```

5. **Adaptive rank** (logged but fixed in main paper, adaptive in appendix):
   ```python
   # Effective rank: number of eigenvalues > 1% of max
   effective_k = (S > S.max() * 0.01).sum().item()
   ```

6. **Gradient norm monitoring** before/after spectral transform:
   ```python
   grad_norm_before = g.norm().item()
   g_spec = apply_preconditioner(g, P)
   grad_norm_after = g_spec.norm().item()
   # If spectral transform explodes gradient, clip or skip
   if grad_norm_after > 10 * grad_norm_before:
       g_spec = g_spec * (grad_norm_before / grad_norm_after)
   ```

### Step Method

```python
def step(self, graph_matrix=None):
    """
    One optimizer step with optional spectral preconditioning.
    graph_matrix: [N, N] — average learned graph from model's last forward.
    """
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            state = self.state[p]
            
            # Initialize state
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            state['step'] += 1
            t = state['step']
            
            # Adam moments
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            β1, β2 = group['betas']
            exp_avg.mul_(β1).add_(grad, alpha=1-β1)
            exp_avg_sq.mul_(β2).addcmul_(grad, grad, value=1-β2)
            
            # Bias correction
            m_hat = exp_avg / (1 - β1**t)
            v_hat = exp_avg_sq / (1 - β2**t)
            adam_dir = m_hat / (v_hat.sqrt() + group['eps'])
            
            # Spectral preconditioning (after warmup, every N steps)
            if (t > self.spectral_warmup and 
                t % self.spectral_interval == 0 and 
                graph_matrix is not None):
                
                P = self._compute_preconditioner(graph_matrix)
                if P is not None:
                    adam_dir = self._apply_preconditioner(adam_dir, P)
            
            # Weight decay + update
            p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
            p.data.add_(adam_dir, alpha=-group['lr'])
```

---

## ═══════════════════════════════════════════════════
## PHASE 3 — Multi-Objective Loss Stack
## ═══════════════════════════════════════════════════

**File**: `sabit_loss.py`
**Class**: `SABiTLoss(nn.Module)`

### Total Loss

```
L_total = L_seg + α(t)·L_prior + β(t)·L_smooth + δ(t)·L_eig + ε(t)·L_evid
```

### Components

| Loss | Formula | Weight | Warmup |
|---|---|---|---|
| L_seg | `combined_loss_with_boundary(pred, target)` | 1.0 | Epoch 0 |
| L_prior | `‖A_eff − A_prior‖²_F / N²` | α=0.1 | Epoch 20→50 |
| L_smooth | `Tr(h^T · (D−A) · h) / N` (graph Laplacian) | β=0.01 | Epoch 20→50 |
| L_eig | `Σ p_i · log(p_i)` (negative entropy → penalize collapse) | δ=0.005 | Epoch 30→60 |
| L_evid | `KL(Dir(α) ‖ Dir(1))` (Dirichlet regularizer) | ε=0.01 | Epoch 10→30 |

### Robustness

1. **Per-component NaN guard** — if any loss component is NaN, set it to 0 and log warning:
   ```python
   if not torch.isfinite(L_prior):
       logger.warning("L_prior is NaN — zeroing this component")
       L_prior = torch.zeros(1, device=device, requires_grad=False)
   ```

2. **Gradient magnitude balancing** — if one loss dominates the gradient, dampen it:
   ```python
   # Compute gradient magnitudes of each component
   # Scale components so no single loss contributes >50% of total gradient norm
   ```

3. **Float32 for spectral entropy** — lgamma under AMP can overflow:
   ```python
   @torch.amp.custom_fwd(cast_inputs=torch.float32)
   def spectral_entropy_loss(eigenvalues):
       p = eigenvalues / (eigenvalues.sum() + 1e-8)
       return (p * torch.log(p + 1e-8)).sum()
   ```

### Forward

```python
def forward(self, pred, target, model_outputs, epoch):
    """
    Args:
        pred: [B, C, D, H, W] logits (or list for deep supervision)
        target: [B, D, H, W] labels
        model_outputs: dict with 'A_effective', 'A_prior', 'features',
                       'alpha', 'eigenvalues'
        epoch: int — current epoch for warmup schedule
    Returns:
        total_loss: Tensor
        components: dict of {name: float} for logging
    """
```

---

## CODING TIMELINE

### Session Plan

| Session | Day | Phases | Files | Est. Lines | Deliverable |
|---|---|---|---|---|---|
| **1** | Day 1 | Phase 0 | `__init__.py`, `prior_builder.py` | 260 | Prior builder with 4 kernels, unit tested |
| **2** | Day 1 | Phase 1A + 1B | `structure_net.py`, `graph_attention.py` | 570 | StructureNet + GraphBiasedAttention, unit tested |
| **3** | Day 2 | Phase 1C + 1D | `transformer_block.py`, `patch_embed.py`, `decoder.py`, `sabit_model.py` | 750 | Full model forward pass on dummy [1,4,128³] |
| **4** | Day 2 | Phase 2 + 3 | `spectral_optimizer.py`, `sabit_loss.py` | 550 | Optimizer step + loss backward, no NaN |
| **5** | Day 3 | Config + Wire | `configs/sabit_v1.yaml`, pipeline integration | 80 | All 17 evidence files generated |
| **6** | Day 3 | Smoke Test | 5-epoch training on BraTS | — | End-to-end verified |

### Total Timeline

```
Days 1-3:  Code all 6 phases + integration test  (~2,200 lines)
Day 4:     First real BraTS training run (5 epochs → verify logging)
Days 5-10: Full 300-epoch training on GPU PC
Day 11:    Ablation runs start (3-4 configurations)
Days 12-17: Ablation training completes
Day 18:    Statistical analysis + paper tables generated automatically by pipeline
```

**Total time to paper-ready results: ~18 days** (3 coding + 15 training/ablation)

> [!IMPORTANT]
> **Next chat: We code Session 1 (Phase 0 — `prior_builder.py`) immediately.** All specifications above are exact — we translate them directly into working PyTorch code.
