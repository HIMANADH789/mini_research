"""
Phase 1D — SABiT: Structure-Aware Bi-Level Transformer
========================================================
Full encoder-decoder model for 3D brain tumor segmentation.

Architecture:
  Encoder Stages 0-2: Standard Swin Transformer blocks (REUSED from swinunetr.py)
  Encoder Stage 3:    SABiT blocks with Prior Builder + StructureNet + GraphBiasedAttention
  Decoder:            UNet-style (REUSED DecoderBlock from swinunetr.py) + deep supervision

  Input [B,4,128,128,128] → PatchEmbed → Stage0(64³,48) → Stage1(32³,96) → Stage2(16³,192)
    → PatchMerge → Stage3(8³,384) [2× SABiTBlock: 512 tokens, full graph attention]
    → Decoder → Output [B,4,128,128,128]

Key design:
  - Only Stage 3 uses novel SABiT attention (N=512 tokens — computationally feasible)
  - Stages 0-2 are standard Swin blocks (proven, stable)
  - This isolates the contribution cleanly for ablation
  - Model exposes API for Trainer v5: get_optimizer_groups, get_spectral_metrics, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from typing import Dict, List, Optional, Tuple

# Reuse proven components from SwinUNETR
from src.boilerplates.models.swinunetr import (
    PatchEmbed3D,
    PatchMerging3D,
    BasicLayer3D,
    EncoderBlock,
    DecoderBlock,
    DropPath,
)

# Our novel components
from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
from src.boilerplates.models.sabit.transformer_block import SABiTBlock


class SABiT(nn.Module):
    """
    Structure-Aware Bi-Level Transformer for Brain Tumor Segmentation.

    Config fields consumed:
      model.in_channels:       4     (MRI modalities)
      model.out_channels:      4     (BG, WT, TC, ET)
      model.feature_size:      48    (base embedding dimension)
      model.window_size:       7     (Swin window size for stages 0-2)
      model.depths:            [2,2,2,2]  (blocks per stage)
      model.num_heads:         [3,6,12,24]
      model.drop_path_rate:    0.2
      model.mlp_ratio:         4.0
      model.prior_top_k:       32
      model.prior_d_prior:     16
      model.structure_d_pair:  64
      model.structure_n_evidence: 4
      model.gradient_checkpointing: false
    """

    def __init__(self, config):
        super().__init__()

        # ── Config extraction ─────────────────────────────────
        in_c   = getattr(config.model, "in_channels", 4)
        out_c  = getattr(config.model, "out_channels", 4)
        fs     = getattr(config.model, "feature_size", 48)
        ws     = getattr(config.model, "window_size", 7)
        depths = list(getattr(config.model, "depths", [2, 2, 2, 2]))
        heads  = list(getattr(config.model, "num_heads", [3, 6, 12, 24]))
        dpr    = float(getattr(config.model, "drop_path_rate", 0.2))
        mlp_r  = float(getattr(config.model, "mlp_ratio", 4.0))

        # SABiT-specific config
        top_k      = int(getattr(config.model, "prior_top_k", 32))
        d_prior    = int(getattr(config.model, "prior_d_prior", 16))
        d_pair     = int(getattr(config.model, "structure_d_pair", 64))
        n_evidence = int(getattr(config.model, "structure_n_evidence", 4))

        self.fs = fs
        self.out_c = out_c
        self._use_checkpoint = bool(getattr(config.model, "gradient_checkpointing", False))

        # Stochastic depth schedule across all stages
        total_depth = sum(depths)
        dpr_list = [x.item() for x in torch.linspace(0, dpr, total_depth)]

        # ══════════════════════════════════════════════════════
        #  ENCODER
        # ══════════════════════════════════════════════════════

        # Raw input skip projection (128³ → 128³ × fs)
        self.enc0 = EncoderBlock(in_c, fs)

        # Patch Embedding: 128³ → 64³ × fs
        self.patch_embed = PatchEmbed3D(in_c, fs, patch_size=2)

        # ── Stage 0 (Swin): 64³ × fs ─────────────────────────
        dp = 0
        self.stage0 = BasicLayer3D(
            dim=fs, depth=depths[0], num_heads=heads[0],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[0]],
        )
        dp += depths[0]
        self.merge0 = PatchMerging3D(fs)

        # ── Stage 1 (Swin): 32³ × 2fs ────────────────────────
        self.stage1 = BasicLayer3D(
            dim=2*fs, depth=depths[1], num_heads=heads[1],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[1]],
        )
        dp += depths[1]
        self.merge1 = PatchMerging3D(2*fs)

        # ── Stage 2 (Swin): 16³ × 4fs ────────────────────────
        self.stage2 = BasicLayer3D(
            dim=4*fs, depth=depths[2], num_heads=heads[2],
            window_size=ws, mlp_ratio=mlp_r, qkv_bias=True,
            drop=0.0, attn_drop=0.0,
            drop_path=dpr_list[dp: dp + depths[2]],
        )
        dp += depths[2]
        self.merge2 = PatchMerging3D(4*fs)

        # ── Stage 3 (SABiT): 8³ × 8fs — OUR INNOVATION ──────
        bottleneck_dim = 8 * fs  # 384 for fs=48
        self.prior_builder = HyperFidelityPriorBuilder(
            embed_dim=bottleneck_dim,
            d_prior=d_prior,
            top_k=top_k,
        )

        # Stack of SABiT blocks at the bottleneck
        self.sabit_blocks = nn.ModuleList([
            SABiTBlock(
                dim=bottleneck_dim,
                num_heads=heads[3],
                d_pair=d_pair,
                n_evidence=n_evidence,
                top_k=top_k,
                mlp_ratio=mlp_r,
                drop_path=dpr_list[dp + i] if (dp + i) < len(dpr_list) else 0.0,
            )
            for i in range(depths[3])
        ])

        # ── Skip connection projectors ────────────────────────
        self.proj_s0 = nn.Sequential(nn.Linear(fs, fs), nn.LayerNorm(fs))
        self.proj_s1 = nn.Sequential(nn.Linear(2*fs, 2*fs), nn.LayerNorm(2*fs))
        self.proj_s2 = nn.Sequential(nn.Linear(4*fs, 4*fs), nn.LayerNorm(4*fs))
        self.proj_bot = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

        # ══════════════════════════════════════════════════════
        #  DECODER (identical to SwinUNETR)
        # ══════════════════════════════════════════════════════

        self.dec4 = DecoderBlock(8*fs, 4*fs, 4*fs)
        self.dec3 = DecoderBlock(4*fs, 2*fs, 2*fs)
        self.dec2 = DecoderBlock(2*fs, fs, fs)
        self.dec1 = DecoderBlock(fs, fs, fs // 2)

        # ── Output + deep supervision heads ───────────────────
        self.out_head = nn.Conv3d(fs // 2, out_c, kernel_size=1)
        self.ds_head4 = nn.Conv3d(4*fs, out_c, kernel_size=1)
        self.ds_head3 = nn.Conv3d(2*fs, out_c, kernel_size=1)

        # ── Cached tensors for diagnostics/optimizer ──────────
        self._cached_A_prior = None
        self._cached_A_effective = None
        self._cached_uncertainty = None
        self._cached_eigenvalues = None
        self._cached_features = None
        self._cached_ev_info = None
        self._cached_prior_info = None
        self._aux_losses = {}

        # ── Epoch tracking for warmup-aware auxiliary losses ──
        self._current_epoch = 0

        # Base weights and warmup schedule
        self._aux_base_weights = {
            "prior":  0.1,
            "smooth": 0.01,
            "eig":    0.005,
            "evid":   0.01,
        }
        self._aux_warmup = {
            "prior":  (20, 50),   # (start_epoch, end_epoch)
            "smooth": (20, 50),
            "eig":    (30, 60),
            "evid":   (10, 30),
        }

        # Position grid buffer (built once, on first forward)
        self.register_buffer("_position_grid", None)

        self._init_weights()

    def _init_weights(self):
        """Standard init for conv/linear/norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for Swin stages to save ~40% VRAM."""
        self._use_checkpoint = True

    def set_epoch(self, epoch: int):
        """Called by Trainer v5 at start of each epoch for warmup scheduling."""
        self._current_epoch = epoch

    def _warmup_weight(self, name: str) -> float:
        """Compute current weight for an auxiliary loss given warmup schedule."""
        base = self._aux_base_weights.get(name, 0.0)
        start, end = self._aux_warmup.get(name, (0, 1))
        epoch = self._current_epoch
        if epoch < start:
            return 0.0
        if epoch >= end:
            return base
        return base * (epoch - start) / max(end - start, 1)

    @staticmethod
    def _to_channels_first(x: Tensor) -> Tensor:
        """[B, D, H, W, C] -> [B, C, D, H, W]"""
        return x.permute(0, 4, 1, 2, 3).contiguous()

    # ══════════════════════════════════════════════════════
    #  FORWARD
    # ══════════════════════════════════════════════════════

    def forward(self, x: Tensor) -> object:
        """
        Full forward pass.

        Parameters
        ----------
        x : Tensor [B, 4, 128, 128, 128]
            4-channel MRI input.

        Returns
        -------
        During training: [main, ds1, ds2] for deep supervision
        During eval: main only
        """
        B = x.shape[0]

        # ── Raw skip (full resolution) ────────────────────────
        if self._use_checkpoint:
            skip_enc0 = cp.checkpoint(self.enc0, x, use_reentrant=False)
        else:
            skip_enc0 = self.enc0(x)

        # ── Patch Embed: [B,4,128³] → [B,64,64,64,fs] ────────
        x_embed = self.patch_embed(x)

        # ── Stage 0 (Swin): 64³ × fs ─────────────────────────
        if self._use_checkpoint:
            s0 = cp.checkpoint(self.stage0, x_embed, use_reentrant=False)
        else:
            s0 = self.stage0(x_embed)

        # ── Stage 1 (Swin): 32³ × 2fs ────────────────────────
        if self._use_checkpoint:
            s1 = cp.checkpoint(self.stage1, self.merge0(s0), use_reentrant=False)
        else:
            s1 = self.stage1(self.merge0(s0))

        # ── Stage 2 (Swin): 16³ × 4fs ────────────────────────
        if self._use_checkpoint:
            s2 = cp.checkpoint(self.stage2, self.merge1(s1), use_reentrant=False)
        else:
            s2 = self.stage2(self.merge1(s1))

        # ── Stage 3 (SABiT): 8³ × 8fs — THE INNOVATION ──────
        bot_5d = self.merge2(s2)  # [B, 8, 8, 8, 8fs]
        Bd, Dd, Hd, Wd, Cd = bot_5d.shape

        # Flatten spatial dims for token sequence: [B, N, d]
        bot_tokens = bot_5d.reshape(B, -1, Cd)  # [B, 512, 384]
        N = bot_tokens.shape[1]

        # Build position grid (cached after first call)
        if self._position_grid is None or self._position_grid.shape[0] != N:
            self._position_grid = HyperFidelityPriorBuilder.build_position_grid(
                Dd, Hd, Wd
            ).to(bot_tokens.device)

        positions = self._position_grid

        # Phase 0: Build structural prior
        A_prior, prior_info = self.prior_builder(bot_tokens, positions)

        # Phase 1: Apply SABiT blocks
        x_out = bot_tokens
        A_effective = A_prior
        uncertainty = None
        ev_info = {}
        all_ev_infos = []

        for block in self.sabit_blocks:
            x_out, A_effective, uncertainty, ev_info = block(
                x_out, positions, A_prior
            )
            all_ev_infos.append(ev_info)

        # Cache for spectral optimizer and diagnostics
        self._cached_A_prior = A_prior.detach()
        self._cached_A_effective = A_effective.detach()
        self._cached_uncertainty = uncertainty.detach() if uncertainty is not None else None
        self._cached_features = x_out.detach()
        self._cached_ev_info = all_ev_infos
        self._cached_prior_info = prior_info

        # Compute eigenvalues for spectral metrics (float32 for safety)
        with torch.no_grad():
            A_avg = A_effective.mean(0).float()
            try:
                eigvals = torch.linalg.eigvalsh(A_avg)
                self._cached_eigenvalues = eigvals[-64:]  # top-64
            except Exception:
                self._cached_eigenvalues = None

        # Build auxiliary losses for trainer v5 (warmup-aware weights)
        self._aux_losses = {}

        # L_prior: ||A_effective - A_prior||²_F / N²
        L_prior = (A_effective - A_prior).pow(2).mean()
        w_prior = self._warmup_weight("prior")
        if w_prior > 0:
            self._aux_losses["prior"] = (L_prior, w_prior)

        # L_smooth: Tr(h^T L h) / N where L = D - A (graph Laplacian)
        D_diag = A_effective.sum(-1)  # [B, N]
        hTDh = (x_out.pow(2) * D_diag.unsqueeze(-1)).sum()
        hTAh = torch.bmm(A_effective, x_out)  # [B, N, d]
        hTAh = (x_out * hTAh).sum()
        L_smooth = (hTDh - hTAh) / (N * B)
        w_smooth = self._warmup_weight("smooth")
        if w_smooth > 0:
            self._aux_losses["smooth"] = (L_smooth, w_smooth)

        # L_eig: spectral entropy (penalize collapse)
        if self._cached_eigenvalues is not None:
            S = self._cached_eigenvalues.clamp(min=1e-8)
            p = S / (S.sum() + 1e-8)
            L_eig = (p * torch.log(p + 1e-8)).sum()  # negative entropy
            w_eig = self._warmup_weight("eig")
            if w_eig > 0:
                self._aux_losses["eig"] = (L_eig, w_eig)

        # L_evid: KL divergence for evidential regularization
        if all_ev_infos and "alpha" in all_ev_infos[-1]:
            from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner
            alpha = all_ev_infos[-1]["alpha"]
            L_evid = EvidentialGraphLearner.evidential_kl_loss(alpha)
            w_evid = self._warmup_weight("evid")
            if w_evid > 0:
                self._aux_losses["evid"] = (L_evid, w_evid)

        # Reshape back to spatial: [B, N, d] → [B, D, H, W, d]
        bot_out = x_out.reshape(B, Dd, Hd, Wd, Cd)

        # ── Project to channels-first for decoder ─────────────
        skip_s0 = self._to_channels_first(self.proj_s0(s0))
        skip_s1 = self._to_channels_first(self.proj_s1(s1))
        skip_s2 = self._to_channels_first(self.proj_s2(s2))
        bottleneck = self._to_channels_first(self.proj_bot(bot_out))

        # ── Decoder ──────────────────────────────────────────
        d4 = self.dec4(bottleneck, skip_s2)
        d3 = self.dec3(d4, skip_s1)
        d2 = self.dec2(d3, skip_s0)
        d1 = self.dec1(d2, skip_enc0)

        # ── Output ───────────────────────────────────────────
        main = self.out_head(d1)

        if self.training:
            ds1 = self.ds_head4(d4)
            ds2 = self.ds_head3(d3)
            return [main, ds1, ds2]

        return main

    # ══════════════════════════════════════════════════════
    #  TRAINER v5 API
    # ══════════════════════════════════════════════════════

    def get_optimizer_groups(self, lr: float) -> Dict[str, list]:
        """
        Returns bi-level parameter groups for Trainer v5.
        Primary: all params EXCEPT StructureNet + PriorBuilder
        Auxiliary: StructureNet + PriorBuilder params (separate LR)
        """
        structure_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "graph_learner" in name or "prior_builder" in name:
                structure_params.append(param)
            else:
                other_params.append(param)

        return {
            "primary": [{"params": other_params, "lr": lr}],
            "auxiliary": [{"params": structure_params, "lr": lr * 0.1}],
        }

    def get_spectral_metrics(self) -> Dict[str, float]:
        """Called by Trainer v5 each epoch for diagnostics."""
        metrics = {}

        if self._cached_eigenvalues is not None:
            S = self._cached_eigenvalues
            S_pos = S[S > 0]
            if len(S_pos) > 1:
                metrics["condition_number"] = (S_pos[-1] / (S_pos[0] + 1e-8)).item()
                metrics["eigenvalue_gap"] = (S_pos[-1] - S_pos[0]).item()
                p = S_pos / (S_pos.sum() + 1e-8)
                metrics["spectrum_entropy"] = (-(p * torch.log(p + 1e-8)).sum()).item()
                metrics["effective_rank"] = (S_pos > S_pos[-1] * 0.01).sum().item()

        if self._cached_A_effective is not None:
            metrics["graph_sparsity"] = (self._cached_A_effective == 0).float().mean().item()

        if self._cached_uncertainty is not None:
            metrics["mean_uncertainty"] = self._cached_uncertainty.mean().item()

        if self._cached_prior_info is not None:
            pw = self._cached_prior_info.get("prior_weights")
            if pw is not None:
                for i, name in enumerate(["spatial", "intensity", "boundary", "stability"]):
                    metrics[f"prior_w_{name}"] = pw[i].item()

        return metrics

    def get_tensor_artifacts(self) -> Dict[str, Optional[Tensor]]:
        """Called by Trainer v5 for artifact storage."""
        return {
            "A_prior": self._cached_A_prior,
            "A_learned": self._cached_A_effective,
            "uncertainty_map": self._cached_uncertainty,
            "eigenvalues": self._cached_eigenvalues,
        }

    def get_auxiliary_losses(self) -> Dict[str, Tuple[Tensor, float]]:
        """Called by Trainer v5 — returns (loss_tensor, weight) pairs."""
        return self._aux_losses


# ══════════════════════════════════════════════════════════════
#  STANDALONE UNIT TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, types
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))

    print("=" * 60)
    print("  Phase 1D - SABiT Full Model Unit Tests")
    print("=" * 60)

    device = "cpu"  # full model too large for unplanned GPU test

    # Build minimal config
    model_cfg = types.SimpleNamespace(
        in_channels=4, out_channels=4, feature_size=48,
        window_size=7, depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24], drop_path_rate=0.1,
        mlp_ratio=4.0, prior_top_k=32, prior_d_prior=16,
        structure_d_pair=64, structure_n_evidence=4,
        gradient_checkpointing=False,
    )
    config = types.SimpleNamespace(model=model_cfg)

    model = SABiT(config).to(device)
    model.train()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # Test optimizer groups
    groups = model.get_optimizer_groups(lr=1e-3)
    n_primary = sum(p.numel() for g in groups["primary"] for p in g["params"])
    n_aux = sum(p.numel() for g in groups["auxiliary"] for p in g["params"])
    print(f"  Primary params:   {n_primary:,}")
    print(f"  Auxiliary params:  {n_aux:,}")
    assert n_primary + n_aux == trainable

    # Test forward (small input for speed)
    print("  Running forward pass (this may take a moment on CPU)...")
    x = torch.randn(1, 4, 64, 64, 64, device=device)
    outputs = model(x)
    assert isinstance(outputs, list) and len(outputs) == 3
    print(f"  Main output:  {outputs[0].shape}")
    print(f"  DS1 output:   {outputs[1].shape}")
    print(f"  DS2 output:   {outputs[2].shape}")

    # Test spectral metrics
    metrics = model.get_spectral_metrics()
    print(f"  Spectral metrics: {list(metrics.keys())}")

    # Test auxiliary losses
    aux = model.get_auxiliary_losses()
    print(f"  Auxiliary losses: {list(aux.keys())}")
    for name, (loss, weight) in aux.items():
        print(f"    {name}: {loss.item():.6f} (weight={weight})")

    # Test tensor artifacts
    artifacts = model.get_tensor_artifacts()
    for name, tensor in artifacts.items():
        shape = tensor.shape if tensor is not None else None
        print(f"  Artifact {name}: {shape}")

    print()
    print("  [OK] ALL PHASE 1D TESTS PASSED")
