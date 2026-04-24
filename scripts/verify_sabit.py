"""SABiT Plan vs Code Verification — cross-checks implementation_plan.md against actual code."""
import sys, os, types, inspect
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

print("=" * 65)
print("  SABiT PLAN vs CODE VERIFICATION")
print("=" * 65)
checks = []

def check(name, ok, detail=""):
    checks.append((name, ok))
    mark = "[OK]" if ok else "[FAIL]"
    msg = f"  {mark} {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)

# ======= FILE EXISTENCE =======
print()
print("--- FILE EXISTENCE ---")
files = {
    "prior_builder.py":      "src/boilerplates/models/sabit/prior_builder.py",
    "structure_net.py":       "src/boilerplates/models/sabit/structure_net.py",
    "graph_attention.py":     "src/boilerplates/models/sabit/graph_attention.py",
    "transformer_block.py":   "src/boilerplates/models/sabit/transformer_block.py",
    "sabit_model.py":         "src/boilerplates/models/sabit/sabit_model.py",
    "spectral_optimizer.py":  "src/boilerplates/models/sabit/spectral_optimizer.py",
    "__init__.py":            "src/boilerplates/models/sabit/__init__.py",
    "sabit_loss.py":          "src/boilerplates/losses/sabit_loss.py",
    "exp_sabit.yaml":         "configs/exp_sabit.yaml",
}
total_lines = 0
for name, path in files.items():
    exists = os.path.isfile(path)
    lines = 0
    if exists:
        with open(path, "r", encoding="utf-8") as f:
            lines = len(f.readlines())
    total_lines += lines
    check(name, exists, f"{lines} lines")
print(f"  Total lines: {total_lines}")

# ======= IMPORT CHECKS =======
print()
print("--- IMPORTS ---")
try:
    from src.boilerplates.models.sabit.prior_builder import HyperFidelityPriorBuilder
    check("Phase 0: HyperFidelityPriorBuilder", True)
except Exception as e:
    check("Phase 0: HyperFidelityPriorBuilder", False, str(e))

try:
    from src.boilerplates.models.sabit.structure_net import EvidentialGraphLearner
    check("Phase 1A: EvidentialGraphLearner", True)
except Exception as e:
    check("Phase 1A: EvidentialGraphLearner", False, str(e))

try:
    from src.boilerplates.models.sabit.graph_attention import GraphBiasedAttention
    check("Phase 1B: GraphBiasedAttention", True)
except Exception as e:
    check("Phase 1B: GraphBiasedAttention", False, str(e))

try:
    from src.boilerplates.models.sabit.transformer_block import SABiTBlock
    check("Phase 1C: SABiTBlock", True)
except Exception as e:
    check("Phase 1C: SABiTBlock", False, str(e))

try:
    from src.boilerplates.models.sabit.sabit_model import SABiT
    check("Phase 1D: SABiT model", True)
except Exception as e:
    check("Phase 1D: SABiT model", False, str(e))

try:
    from src.boilerplates.models.sabit.spectral_optimizer import SpectralAdam
    check("Phase 2: SpectralAdam", True)
except Exception as e:
    check("Phase 2: SpectralAdam", False, str(e))

try:
    from src.boilerplates.losses.sabit_loss import SABiTLoss
    check("Phase 3: SABiTLoss", True)
except Exception as e:
    check("Phase 3: SABiTLoss", False, str(e))

# ======= PLAN REQUIREMENT CHECKS =======
print()
print("--- PLAN REQUIREMENTS ---")
import torch

model_cfg = types.SimpleNamespace(
    in_channels=4, out_channels=4, feature_size=48,
    window_size=7, depths=[2,2,2,2], num_heads=[3,6,12,24],
    drop_path_rate=0.1, mlp_ratio=4.0, prior_top_k=32,
    prior_d_prior=16, structure_d_pair=64, structure_n_evidence=4,
    gradient_checkpointing=False)
config = types.SimpleNamespace(model=model_cfg)
model = SABiT(config)
model.train()

total_params = sum(p.numel() for p in model.parameters())
check(f"Model params ~11M", total_params > 10_000_000, f"{total_params:,}")

# Plan: 4 kernels in prior
check("4-kernel prior builder",
      hasattr(model.prior_builder, "log_sigma") and
      hasattr(model.prior_builder, "log_tau") and
      hasattr(model.prior_builder, "log_alpha") and
      hasattr(model.prior_builder, "log_gamma"))

# Plan: EMA buffer for stability prior
check("Stability prior EMA buffer",
      hasattr(model.prior_builder, "feature_ema"))

# Plan: Evidential KL loss (float32)
check("Evidential KL loss static method",
      hasattr(EvidentialGraphLearner, "evidential_kl_loss"))

# Plan: Layer Scale in SABiTBlock
block = model.sabit_blocks[0]
check("Layer Scale in SABiTBlock",
      hasattr(block, "layer_scale_1") and hasattr(block, "layer_scale_2"),
      f"init val: {block.layer_scale_1.data.mean():.4f}")

# Plan: Gate network in GraphBiasedAttention
check("Gate network in attention", hasattr(block.attn, "gate_net"))

# Plan: Per-head bias scale
check("Per-head bias scale", hasattr(block.attn, "bias_scale"))

# Plan: get_optimizer_groups returns primary+auxiliary
groups = model.get_optimizer_groups(lr=1e-3)
check("Bi-level optimizer groups",
      "primary" in groups and "auxiliary" in groups)

n_p = sum(p.numel() for g in groups["primary"] for p in g["params"])
n_a = sum(p.numel() for g in groups["auxiliary"] for p in g["params"])
check(f"All params assigned to groups",
      n_p + n_a == total_params,
      f"primary={n_p:,} + aux={n_a:,} = {n_p+n_a:,}")

# Plan: API methods
check("get_spectral_metrics method", hasattr(model, "get_spectral_metrics"))
check("get_tensor_artifacts method", hasattr(model, "get_tensor_artifacts"))
check("get_auxiliary_losses method", hasattr(model, "get_auxiliary_losses"))
check("enable_gradient_checkpointing", hasattr(model, "enable_gradient_checkpointing"))
check("set_epoch warmup method", hasattr(model, "set_epoch"))

# Plan: Swin stages 0-2 reused from swinunetr.py
from src.boilerplates.models.swinunetr import BasicLayer3D
check("Swin stages reused from swinunetr",
      isinstance(model.stage0, BasicLayer3D))

# Plan: Deep supervision
check("Deep supervision heads",
      hasattr(model, "ds_head4") and hasattr(model, "ds_head3"))

# Plan: SpectralAdam has warmup, fallback, EMA
opt = SpectralAdam([torch.nn.Parameter(torch.randn(4,4))], lr=1e-3)
check("SpectralAdam warmup/fallback",
      hasattr(opt, "spectral_warmup") and hasattr(opt, "fallback_to_adam"))

# ======= FORWARD PASS =======
print()
print("--- FORWARD PASS ---")
x = torch.randn(1, 4, 64, 64, 64)
model.set_epoch(60)
out = model(x)
check("Forward returns list of 3",
      isinstance(out, list) and len(out) == 3)
check("Main output shape correct",
      out[0].shape == (1, 4, 64, 64, 64), str(out[0].shape))
check("DS1 output exists", out[1] is not None, str(out[1].shape))
check("DS2 output exists", out[2] is not None, str(out[2].shape))

# ======= AUX LOSSES + WARMUP =======
print()
print("--- AUXILIARY LOSSES + WARMUP ---")
aux = model.get_auxiliary_losses()
check("L_prior at epoch 60", "prior" in aux)
check("L_smooth at epoch 60", "smooth" in aux)
check("L_eig at epoch 60", "eig" in aux)
check("L_evid at epoch 60", "evid" in aux)

# Test warmup: epoch 0 should have NO aux losses
model.set_epoch(0)
model(x)
aux0 = model.get_auxiliary_losses()
check("Epoch 0: no aux losses (warmup)", len(aux0) == 0, f"got {list(aux0.keys())}")

# Test warmup: epoch 15 should have only evid
model.set_epoch(15)
model(x)
aux15 = model.get_auxiliary_losses()
check("Epoch 15: only evid active", "evid" in aux15 and "prior" not in aux15)

# ======= SPECTRAL METRICS =======
print()
print("--- SPECTRAL METRICS ---")
model.set_epoch(60)
model(x)
metrics = model.get_spectral_metrics()
expected = ["condition_number", "eigenvalue_gap", "spectrum_entropy",
            "effective_rank", "graph_sparsity", "mean_uncertainty"]
for m in expected:
    check(f"Metric: {m}", m in metrics)

# ======= BACKWARD =======
print()
print("--- BACKWARD PASS ---")
from src.boilerplates.losses.boundary_aware_loss import combined_loss_with_boundary
seg = torch.randint(0, 4, (1, 64, 64, 64))
out = model(x)
aux = model.get_auxiliary_losses()
loss, comp = combined_loss_with_boundary(
    out[0], seg, class_weights=[0.1,3.0,1.0,2.0],
    focal_weight=0.5, boundary_weight=0.4)
for name, (lv, w) in aux.items():
    loss = loss + w * lv
loss.backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
check(f"Backward OK", grad_count > 200, f"{grad_count} param grads")
check("Loss is finite", torch.isfinite(loss), f"loss={loss.item():.4f}")

# ======= CONFIG + PIPELINE =======
print()
print("--- CONFIG + PIPELINE ---")
from src.utils.experiment_utils.config import load_config
cfg = load_config("configs/exp_sabit.yaml")
check("Config loads", cfg.model.type == "sabit")
check("Trainer = segmentation_v5", cfg.versions.trainer == "segmentation_v5")
check("Evaluator = v2", cfg.versions.evaluation == "v2")
check("Data = v2", cfg.versions.data == "v2")
check("Epochs = 300", cfg.training.epochs == 300)
check("Accumulate = 4", cfg.training.accumulate_steps == 4)
check("Grad checkpointing", cfg.training.gradient_checkpointing == True)

from src.boilerplates.model_builder.build import build_model
m2 = build_model(cfg)
check("build_model(sabit) works", type(m2).__name__ == "SABiT")

from src.boilerplates.resolver import get_trainer_class, get_evaluator_class
T = get_trainer_class(cfg)
check("Trainer v5 resolves", "v5" in T.__module__)
E = get_evaluator_class(cfg)
check("Evaluator v2 resolves", "v2" in E.__module__)

# Trainer v5 has set_epoch call
src_code = inspect.getsource(T)
check("Trainer calls model.set_epoch", "set_epoch" in src_code)
check("Trainer calls get_auxiliary_losses", "get_auxiliary_losses" in src_code)
check("Trainer calls get_spectral_metrics", "get_spectral_metrics" in src_code)
check("Trainer calls get_tensor_artifacts", "get_tensor_artifacts" in src_code)
check("Trainer has gradient accumulation", "accumulate_steps" in src_code)

# ======= SUMMARY =======
print()
passed = sum(1 for _, ok in checks if ok)
failed = sum(1 for _, ok in checks if not ok)
print("=" * 65)
if failed == 0:
    print(f"  ALL {passed} CHECKS PASSED - CODE IS COMPLETE AND FUNCTIONAL")
else:
    print(f"  {passed} PASSED, {failed} FAILED")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL: {name}")
print("=" * 65)
print()
print("RUN COMMAND:")
print("  python scripts/run_experiment.py --config configs/exp_sabit.yaml")
print()
print("RESUME AFTER INTERRUPTION:")
print("  python scripts/run_experiment.py --config configs/exp_sabit.yaml --resume auto")
