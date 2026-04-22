"""Full SABiT integration test: model + loss + optimizer."""
import sys, types, torch
sys.path.insert(0, ".")

print("=" * 60)
print("  SABiT Full Integration Test")
print("=" * 60)

from src.boilerplates.models.sabit.sabit_model import SABiT
from src.boilerplates.losses.sabit_loss import SABiTLoss
from src.boilerplates.models.sabit.spectral_optimizer import SpectralAdam
print("  [OK] All imports successful")

# Build config
model_cfg = types.SimpleNamespace(
    in_channels=4, out_channels=4, feature_size=48,
    window_size=7, depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24], drop_path_rate=0.1,
    mlp_ratio=4.0, prior_top_k=32, prior_d_prior=16,
    structure_d_pair=64, structure_n_evidence=4,
    gradient_checkpointing=False,
)
config = types.SimpleNamespace(model=model_cfg)

model = SABiT(config)
model.train()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  [OK] SABiT created: {total:,} params ({trainable:,} trainable)")

# Optimizer groups
groups = model.get_optimizer_groups(lr=1e-3)
n_p = sum(p.numel() for g in groups["primary"] for p in g["params"])
n_a = sum(p.numel() for g in groups["auxiliary"] for p in g["params"])
print(f"  [OK] Optimizer groups: primary={n_p:,}, auxiliary={n_a:,}")
assert n_p + n_a == trainable, "Param count mismatch!"

# Forward pass (64^3 for CPU speed)
print("  ... Running forward pass (64^3 on CPU) ...")
x = torch.randn(1, 4, 64, 64, 64)
out = model(x)
assert isinstance(out, list) and len(out) == 3
print(f"  [OK] Forward: main={out[0].shape}, ds1={out[1].shape}, ds2={out[2].shape}")

# Spectral metrics
metrics = model.get_spectral_metrics()
print(f"  [OK] Spectral metrics: {list(metrics.keys())}")

# Auxiliary losses
aux = model.get_auxiliary_losses()
print(f"  [OK] Aux losses: {list(aux.keys())}")

# Tensor artifacts
artifacts = model.get_tensor_artifacts()
for name, tensor in artifacts.items():
    shape = tensor.shape if tensor is not None else None
    print(f"       artifact {name}: {shape}")

# Loss computation  
loss_fn = SABiTLoss(num_classes=4)
target = torch.randint(0, 4, (1, 64, 64, 64))
loss, info = loss_fn(out, target, {"aux_losses": aux}, epoch=60)
print(f"  [OK] Loss: {loss.item():.4f}")
for k, v in sorted(info.items()):
    print(f"       {k}: {v:.6f}" if isinstance(v, float) else f"       {k}: {v}")

# Backward pass
loss.backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
print(f"  [OK] Backward: {grad_count} params have gradients")

# SpectralAdam test
spec_params = groups["auxiliary"]
opt = SpectralAdam(spec_params, lr=1e-4, rank_k=32, spectral_warmup=0)
A_graph = model.get_tensor_artifacts().get("A_learned")
if A_graph is not None:
    A_avg = A_graph.mean(0)
    opt.step(graph_matrix=A_avg)
    print(f"  [OK] SpectralAdam step completed")
    analysis = opt.get_spectral_analysis()
    print(f"       spectral_active={analysis['spectral_active']}")

print()
print("  " + "=" * 50)
print("  ALL INTEGRATION TESTS PASSED")
print("  " + "=" * 50)
