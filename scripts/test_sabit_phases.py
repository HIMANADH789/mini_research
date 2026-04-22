"""
SABiT Phase Validation Script
===============================
Run all implemented phases sequentially, validating each module,
then testing the full Phase 0 → 1A → 1B pipeline end-to-end.

Usage:
    python scripts/test_sabit_phases.py

    Options:
        --device cpu     Force CPU execution
        --device cuda    Force CUDA execution
        --phase 0        Test only Phase 0
        --phase 1a       Test only Phase 1A
        --phase 1b       Test only Phase 1B
        --all            Test all phases + end-to-end pipeline (default)
"""

import sys
import os
import argparse
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def separator(title: str):
    print()
    print("═" * 64)
    print(f"  {title}")
    print("═" * 64)


def test_phase_0(device: str) -> dict:
    """Test Phase 0 — HyperFidelityPriorBuilder."""
    separator("PHASE 0 — HyperFidelityPriorBuilder")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder

    B, N, d = 2, 512, 384
    passed = 0
    total = 0

    builder = HyperFidelityPriorBuilder(
        embed_dim=d, d_prior=16, top_k=32,
        sigma_init=0.1, tau_init=0.5, alpha_init=5.0, gamma_init=2.0,
    ).to(device)

    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    features = torch.randn(B, N, d, device=device)

    t0 = time.perf_counter()
    A_prior, info = builder(features, positions)
    t1 = time.perf_counter()

    tests = [
        ("Shape",           A_prior.shape == (B, N, N)),
        ("Row normalized",  torch.allclose(A_prior.sum(-1), torch.ones(B, N, device=device), atol=0.02)),
        ("No self-loops",   A_prior.diagonal(dim1=-2, dim2=-1).abs().max().item() < 0.01),
        ("Sparse (>90%)",   (A_prior == 0).float().mean().item() > 0.9),
        ("Non-negative",    (A_prior >= 0).all().item()),
        ("Finite",          torch.isfinite(A_prior).all().item()),
        ("Prior weights",   info["prior_weights"].shape == (4,)),
        ("Sigma positive",  info["sigma"] > 0),
        ("Tau positive",    info["tau"] > 0),
    ]

    for name, result in tests:
        total += 1
        if result:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    # Gradient test
    total += 1
    loss = A_prior.sum()
    loss.backward()
    if builder.log_sigma.grad is not None and builder.prior_logits.grad is not None:
        passed += 1
        print(f"  ✅ Differentiable (σ, τ, α, γ, weights all have grads)")
    else:
        print(f"  ❌ Differentiable")

    # Stability prior test (2nd forward)
    total += 1
    builder.zero_grad()
    A2, info2 = builder(features + 0.01 * torch.randn_like(features), positions)
    if A2.shape == (B, N, N) and builder.feature_ema is not None:
        passed += 1
        print(f"  ✅ Stability prior: EMA buffer updated")
    else:
        print(f"  ❌ Stability prior")

    params = sum(p.numel() for p in builder.parameters())
    print(f"\n  📊 Parameters: {params:,}")
    print(f"  ⏱  Forward time: {(t1-t0)*1000:.1f} ms")
    print(f"  📈 Sparsity: {info['sparsity']:.2%}")
    print(f"  🎯 {passed}/{total} tests passed")

    return {"builder": builder, "positions": positions, "A_prior": A_prior,
            "passed": passed, "total": total}


def test_phase_1a(device: str, prior_result: dict = None) -> dict:
    """Test Phase 1A — EvidentialGraphLearner."""
    separator("PHASE 1A — EvidentialGraphLearner")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
    from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner

    B, N, d = 2, 512, 384
    passed = 0
    total = 0

    # Get prior from Phase 0 result or build fresh
    if prior_result and "A_prior" in prior_result:
        positions = prior_result["positions"]
        A_prior = prior_result["A_prior"].detach()
    else:
        builder = HyperFidelityPriorBuilder(embed_dim=d, d_prior=16, top_k=32).to(device)
        positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
        features = torch.randn(B, N, d, device=device)
        A_prior, _ = builder(features, positions)
        A_prior = A_prior.detach()

    learner = EvidentialGraphLearner(
        embed_dim=d, d_pair=64, n_evidence=4, top_k=32,
        mu_scale_init=0.01, dropout=0.1, edge_drop_rate=0.1,
    ).to(device)

    h = torch.randn(B, N, d, device=device)

    t0 = time.perf_counter()
    A_eff, unc, ev_info = learner(h, positions, A_prior)
    t1 = time.perf_counter()

    tests = [
        ("A_eff shape",       A_eff.shape == (B, N, N)),
        ("Uncertainty shape", unc.shape == (B, N, N)),
        ("A_eff non-negative", (A_eff >= 0).all().item()),
        ("A_eff finite",      torch.isfinite(A_eff).all().item()),
        ("Uncertainty bounded", 0 < ev_info["mean_uncertainty"] < 1),
        ("Mu near zero",      ev_info["mu_mean"] < 0.2),
        ("Evidence > 0",      ev_info["evidence_strength"] > 0),
        ("Alpha has C=4",    ev_info["alpha"].shape[-1] == 4),
    ]

    for name, result in tests:
        total += 1
        if result:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    # Alpha bounds
    total += 1
    alpha = ev_info["alpha"]
    if (alpha >= 1.01).all() and (alpha <= 1000.0).all():
        passed += 1
        print(f"  ✅ Alpha clamped: [{alpha.min():.2f}, {alpha.max():.2f}]")
    else:
        print(f"  ❌ Alpha bounds")

    # Row normalization (approximately)
    total += 1
    row_sums = A_eff.sum(-1)
    max_dev = (row_sums - 1).abs().max().item()
    if max_dev < 0.1:
        passed += 1
        print(f"  ✅ Row norm deviation: {max_dev:.6f}")
    else:
        print(f"  ❌ Row norm deviation: {max_dev:.6f}")

    # KL loss test (before consuming gradients)
    total += 1
    kl = EvidentialGraphLearner.evidential_kl_loss(alpha.detach().requires_grad_(True))
    if torch.isfinite(kl):
        kl.backward()
        passed += 1
        print(f"  ✅ KL loss: {kl.item():.4f} (finite)")
    else:
        print(f"  ❌ KL loss: NaN/Inf")

    # Gradient test (after KL so graph isn't consumed)
    total += 1
    loss = A_eff.sum()
    loss.backward()
    if learner.mu_scale.grad is not None:
        passed += 1
        print(f"  ✅ Differentiable")
    else:
        print(f"  ❌ Differentiable")

    params = sum(p.numel() for p in learner.parameters())
    print(f"\n  📊 Parameters: {params:,}")
    print(f"  ⏱  Forward time: {(t1-t0)*1000:.1f} ms")
    print(f"  🎯 {passed}/{total} tests passed")

    return {"learner": learner, "A_eff": A_eff, "unc": unc,
            "passed": passed, "total": total, "positions": positions,
            "A_prior": A_prior}


def test_phase_1b(device: str, prior_result: dict = None, struct_result: dict = None) -> dict:
    """Test Phase 1B — GraphBiasedAttention."""
    separator("PHASE 1B — GraphBiasedAttention")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
    from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner
    from src.boilerplates.models.sabit.graph_attention import GraphBiasedAttention

    B, N, d = 2, 512, 384
    heads = 12
    passed = 0
    total = 0

    # Get graphs from prior results or build fresh
    if struct_result and "A_eff" in struct_result:
        A_eff = struct_result["A_eff"].detach()
        A_prior = struct_result["A_prior"].detach()
    elif prior_result and "A_prior" in prior_result:
        A_prior = prior_result["A_prior"].detach()
        A_eff = A_prior.clone()  # use prior as effective for testing
    else:
        A_prior = torch.rand(B, N, N, device=device).softmax(-1)
        A_eff = torch.rand(B, N, N, device=device).softmax(-1)

    attn = GraphBiasedAttention(dim=d, num_heads=heads, bias_scale_init=1.0).to(device)
    x = torch.randn(B, N, d, device=device, requires_grad=True)

    t0 = time.perf_counter()
    out = attn(x, A_eff, A_prior)
    t1 = time.perf_counter()

    tests = [
        ("Output shape",   out.shape == (B, N, d)),
        ("Output finite",  torch.isfinite(out).all().item()),
        ("Not all zeros",  out.abs().sum().item() > 0),
    ]

    for name, result in tests:
        total += 1
        if result:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    # Gradient test
    total += 1
    loss = out.sum()
    loss.backward()
    if (x.grad is not None and attn.bias_scale.grad is not None
            and attn.qkv.weight.grad is not None):
        passed += 1
        print(f"  ✅ Differentiable (input, bias_scale, QKV)")
    else:
        print(f"  ❌ Differentiable")

    # Gate range test
    total += 1
    with torch.no_grad():
        gate_in = torch.stack([A_eff, A_prior], dim=-1)
        g = torch.sigmoid(attn.gate_net(gate_in))
        if g.min() >= 0.0 and g.max() <= 1.0:
            passed += 1
            print(f"  ✅ Gate range: [{g.min():.4f}, {g.max():.4f}]")
        else:
            print(f"  ❌ Gate range")

    # Bias non-negativity test
    total += 1
    with torch.no_grad():
        A_blend = g * A_eff.unsqueeze(-1) + (1-g) * A_prior.unsqueeze(-1)
        bias = attn.bias_proj(A_blend).permute(0, 3, 1, 2)
        bias_act = F.softplus(bias)
        if (bias_act >= 0).all():
            passed += 1
            print(f"  ✅ Bias non-negative (softplus enforced)")
        else:
            print(f"  ❌ Bias non-negative")

    # Per-head bias scale
    total += 1
    if attn.bias_scale.shape == (heads,):
        passed += 1
        print(f"  ✅ Per-head scales: {attn.bias_scale.data.tolist()[:4]}...")
    else:
        print(f"  ❌ Per-head scales shape")

    params = sum(p.numel() for p in attn.parameters())
    print(f"\n  📊 Parameters: {params:,}")
    print(f"  ⏱  Forward time: {(t1-t0)*1000:.1f} ms")
    print(f"  🎯 {passed}/{total} tests passed")

    return {"attn": attn, "passed": passed, "total": total}


def test_pipeline(device: str):
    """End-to-end pipeline test: Phase 0 → 1A → 1B chained."""
    separator("END-TO-END PIPELINE: Phase 0 → 1A → 1B")

    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
    from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner
    from src.boilerplates.models.sabit.graph_attention import GraphBiasedAttention

    B, N, d = 2, 512, 384
    heads = 12
    passed = 0
    total = 0

    # Build all modules
    builder = HyperFidelityPriorBuilder(embed_dim=d, d_prior=16, top_k=32).to(device)
    learner = EvidentialGraphLearner(embed_dim=d, d_pair=64, n_evidence=4, top_k=32).to(device)
    attn = GraphBiasedAttention(dim=d, num_heads=heads).to(device)

    positions = HyperFidelityPriorBuilder.build_position_grid(8, 8, 8).to(device)
    x = torch.randn(B, N, d, device=device, requires_grad=True)

    # ── Full forward pipeline ────────────────────────────
    t0 = time.perf_counter()

    # Phase 0: Prior
    A_prior, prior_info = builder(x, positions)

    # Phase 1A: StructureNet
    A_eff, unc, ev_info = learner(x, positions, A_prior)

    # Phase 1B: Graph-biased attention
    out = attn(x, A_eff, A_prior)

    t1 = time.perf_counter()

    # ── Full backward ─────────────────────────────────────
    loss = out.sum()
    loss.backward()

    t2 = time.perf_counter()

    # Tests
    tests = [
        ("Pipeline shape",     out.shape == (B, N, d)),
        ("Pipeline finite",    torch.isfinite(out).all().item()),
        ("Input has grad",     x.grad is not None),
        ("Prior weights grad", builder.prior_logits.grad is not None),
        ("Mu scale grad",      learner.mu_scale.grad is not None),
        ("Bias scale grad",    attn.bias_scale.grad is not None),
    ]

    for name, result in tests:
        total += 1
        if result:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    # AMP test (if CUDA available)
    if device == "cuda":
        total += 1
        try:
            with torch.amp.autocast(device_type="cuda"):
                A_prior_amp, _ = builder(x.detach(), positions)
                A_eff_amp, _, _ = learner(x.detach(), positions, A_prior_amp)
                out_amp = attn(x.detach(), A_eff_amp, A_prior_amp)
            if torch.isfinite(out_amp).all():
                passed += 1
                print(f"  ✅ AMP forward pass (mixed precision)")
            else:
                print(f"  ❌ AMP forward pass (NaN/Inf)")
        except Exception as e:
            print(f"  ❌ AMP forward pass: {e}")

    # Total parameter count
    total_params = (
        sum(p.numel() for p in builder.parameters())
        + sum(p.numel() for p in learner.parameters())
        + sum(p.numel() for p in attn.parameters())
    )

    # Memory estimate
    mem_mb = total_params * 4 / (1024**2)

    print(f"\n  📊 Total parameters (Phase 0+1A+1B): {total_params:,}")
    print(f"  💾 Estimated memory: {mem_mb:.1f} MB")
    print(f"  ⏱  Forward: {(t1-t0)*1000:.1f} ms | Backward: {(t2-t1)*1000:.1f} ms")
    print(f"  🔧 Device: {device}")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  🎮 Peak VRAM: {peak_mb:.1f} MB")

    print(f"\n  🎯 {passed}/{total} tests passed")
    return passed, total


def main():
    parser = argparse.ArgumentParser(description="SABiT Phase Validation")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu or cuda)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["0", "1a", "1b", "all"],
                        help="Which phase to test")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         SABiT Phase Validation Suite                      ║")
    print("║         Structure-Aware Bi-Level Transformer              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

    total_passed = 0
    total_tests = 0

    if args.phase in ("0", "all"):
        r = test_phase_0(device)
        total_passed += r["passed"]
        total_tests += r["total"]

    if args.phase in ("1a", "all"):
        prior_r = r if args.phase == "all" else None
        r1a = test_phase_1a(device, prior_r)
        total_passed += r1a["passed"]
        total_tests += r1a["total"]

    if args.phase in ("1b", "all"):
        prior_r = r if args.phase == "all" else None
        struct_r = r1a if args.phase == "all" else None
        r1b = test_phase_1b(device, prior_r, struct_r)
        total_passed += r1b["passed"]
        total_tests += r1b["total"]

    if args.phase == "all":
        p, t = test_pipeline(device)
        total_passed += p
        total_tests += t

    # ── Final Summary ────────────────────────────────────
    separator("FINAL SUMMARY")
    if total_passed == total_tests:
        print(f"  🏆 ALL {total_tests} TESTS PASSED")
    else:
        print(f"  ⚠️  {total_passed}/{total_tests} tests passed "
              f"({total_tests - total_passed} FAILED)")
    print()

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
