"""
system_utils.py
===============
Centralized hardware, software, and environment introspection for the
mini_research pipeline.

All collection functions are fully try/except guarded — they NEVER crash
training runs. Returns "unavailable" strings for any field that cannot
be collected (e.g. no psutil, no git, no GPU).

Exports:
    collect_system_info()       → system_info.json data
    collect_gpu_memory_used()   → peak VRAM after training
    collect_git_info()          → git commit / branch / dirty flag
    collect_environment_info()  → pip freeze, CUDA env vars, argv
    count_parameters(model)     → total / trainable / frozen params
    config_hash(config)         → short MD5 of config for dedup
    format_duration(s)          → "2h 34m 12s" string

These are imported by experiment_tracker.py, repro_utils.py,
and other utils — kept as the single source of truth.
"""

import os
import sys
import json
import time
import socket
import hashlib
import platform
import tempfile
import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS
# ══════════════════════════════════════════════════════════════

def _try_import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _safe(fn, default="unavailable"):
    """Execute fn(); return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  CPU NAME HELPER
# ══════════════════════════════════════════════════════════════

def _get_cpu_name() -> str:
    """Cross-platform CPU name extraction. Never raises."""
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Name=" in line:
                    name = line.split("=", 1)[-1].strip()
                    if name:
                        return name
        except Exception:
            pass

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[-1].strip()
        except Exception:
            pass

    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            pass

    return platform.processor() or "unavailable"


# ══════════════════════════════════════════════════════════════
#  SYSTEM INFO
# ══════════════════════════════════════════════════════════════

def collect_system_info() -> Dict[str, Any]:
    """
    Collect complete hardware and software environment info.

    Covers:
      - OS name/version, hostname, username, machine arch
      - CPU name, physical + logical core count
      - RAM total / available
      - PyTorch version, CUDA version, cuDNN version
      - GPU count, GPU names, VRAM total per GPU
      - Python version

    Returns dict safe for JSON serialization.
    All fields are try/except guarded.
    """
    psutil = _try_import_psutil()
    torch  = _try_import_torch()
    info: Dict[str, Any] = {}

    # ── OS / Host ───────────────────────────────────────────────
    info["hostname"]          = _safe(lambda: socket.gethostname())
    info["username"]          = _safe(lambda: os.environ.get("USERNAME") or
                                               os.environ.get("USER") or
                                               "unavailable")
    info["os"]                = _safe(lambda: platform.platform())
    info["os_name"]           = _safe(lambda: platform.system())
    info["os_version"]        = _safe(lambda: platform.version())
    info["machine_arch"]      = _safe(lambda: platform.machine())
    info["python_version"]    = _safe(lambda: sys.version.split()[0])
    info["python_executable"] = _safe(lambda: sys.executable)

    # ── CPU ─────────────────────────────────────────────────────
    info["cpu_name"]          = _safe(_get_cpu_name)
    if psutil:
        info["cpu_cores_physical"] = _safe(lambda: psutil.cpu_count(logical=False))
        info["cpu_cores_logical"]  = _safe(lambda: psutil.cpu_count(logical=True))
        info["ram_total_gb"]       = _safe(lambda: round(psutil.virtual_memory().total / 1e9, 2))
        info["ram_available_gb"]   = _safe(lambda: round(psutil.virtual_memory().available / 1e9, 2))
    else:
        for k in ["cpu_cores_physical", "cpu_cores_logical", "ram_total_gb", "ram_available_gb"]:
            info[k] = "unavailable (install psutil)"

    # ── PyTorch / CUDA ──────────────────────────────────────────
    if torch is not None:
        info["torch_version"]   = _safe(lambda: torch.__version__)
        info["cuda_available"]  = _safe(lambda: torch.cuda.is_available())
        info["cuda_version"]    = _safe(
            lambda: torch.version.cuda if torch.cuda.is_available() else "N/A"
        )
        info["cudnn_version"]   = _safe(
            lambda: str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A"
        )
        # ── GPU(s) ─────────────────────────────────────────────
        if _safe(lambda: torch.cuda.is_available(), False):
            n_gpus = _safe(lambda: torch.cuda.device_count(), 0)
            info["gpu_count"] = n_gpus
            gpus: List[Dict] = []
            for i in range(n_gpus):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index":               i,
                        "name":                props.name,
                        "memory_total_gb":     round(props.total_memory / 1e9, 2),
                        "compute_capability":  f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    })
                except Exception:
                    pass
            info["gpus"]                = gpus
            # Convenience: primary GPU
            if gpus:
                info["gpu_name"]             = gpus[0]["name"]
                info["gpu_memory_total_gb"]  = gpus[0]["memory_total_gb"]
            else:
                info["gpu_name"]             = "N/A"
                info["gpu_memory_total_gb"]  = "N/A"
        else:
            info["gpu_count"]            = 0
            info["gpus"]                 = []
            info["gpu_name"]             = "N/A"
            info["gpu_memory_total_gb"]  = "N/A"
    else:
        info["torch_version"] = "unavailable (torch not installed)"
        info["cuda_available"] = False
        info["cuda_version"]  = "N/A"
        info["cudnn_version"] = "N/A"
        info["gpu_count"]     = 0
        info["gpus"]          = []
        info["gpu_name"]      = "N/A"
        info["gpu_memory_total_gb"] = "N/A"

    info["collected_at"] = datetime.now(timezone.utc).isoformat()
    return info


# ══════════════════════════════════════════════════════════════
#  GPU MEMORY PEAK
# ══════════════════════════════════════════════════════════════

def collect_gpu_memory_used() -> Dict[str, Any]:
    """
    Collect peak GPU memory usage. Call AFTER training to get max allocation.
    """
    torch = _try_import_torch()
    if torch is None or not _safe(lambda: torch.cuda.is_available(), False):
        return {"gpu_memory_max_used_gb": "N/A", "gpu_memory_reserved_gb": "N/A"}

    mem_info: Dict[str, Any] = {}
    n_gpus = _safe(lambda: torch.cuda.device_count(), 0)
    for i in range(n_gpus):
        try:
            mem_info[f"gpu_{i}_max_allocated_gb"] = round(
                torch.cuda.max_memory_allocated(i) / 1e9, 3
            )
            mem_info[f"gpu_{i}_max_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(i) / 1e9, 3
            )
        except Exception:
            pass

    # Convenience: primary GPU
    mem_info["gpu_memory_max_used_gb"] = mem_info.get("gpu_0_max_allocated_gb", "N/A")
    mem_info["gpu_memory_reserved_gb"] = mem_info.get("gpu_0_max_reserved_gb", "N/A")
    return mem_info


# ══════════════════════════════════════════════════════════════
#  GIT INFO
# ══════════════════════════════════════════════════════════════

def collect_git_info() -> Dict[str, Any]:
    """
    Collect git metadata: commit hash, branch, dirty flag.
    Returns 'unavailable' for each field if git is not available.
    """
    info: Dict[str, Any] = {
        "git_commit": "unavailable",
        "git_branch": "unavailable",
        "git_dirty":  "unavailable",
    }
    try:
        cwd = os.getcwd()

        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if r.returncode == 0:
            info["git_commit"] = r.stdout.strip()

        r2 = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if r2.returncode == 0:
            info["git_branch"] = r2.stdout.strip()

        r3 = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if r3.returncode == 0:
            info["git_dirty"] = bool(r3.stdout.strip())

        # Full commit hash for reproducibility
        r4 = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd
        )
        if r4.returncode == 0:
            info["git_commit_full"] = r4.stdout.strip()

    except Exception:
        pass
    return info


# ══════════════════════════════════════════════════════════════
#  ENVIRONMENT (pip freeze, CUDA env vars, launch cmd)
# ══════════════════════════════════════════════════════════════

def collect_environment_info() -> Dict[str, Any]:
    """
    Collect runtime environment details:
      - pip_packages  : list from pip freeze (or conda list)
      - cuda_env_vars : CUDA-related environment variables
      - launch_command: sys.argv (command used to launch the script)
      - working_dir   : os.getcwd() at time of collection
    """
    env: Dict[str, Any] = {}

    # ── Launch command ──────────────────────────────────────────
    env["launch_command"] = " ".join(sys.argv)
    env["working_dir"]    = os.getcwd()
    env["collected_at"]   = datetime.now(timezone.utc).isoformat()

    # ── Pip freeze ─────────────────────────────────────────────
    packages = _collect_pip_packages()
    env["pip_packages"] = packages
    env["package_count"] = len(packages)

    # ── CUDA environment variables ──────────────────────────────
    cuda_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_DEVICE_ORDER",
        "CUDA_HOME",
        "CUDA_PATH",
        "NCCL_DEBUG",
        "TORCH_CUDA_ARCH_LIST",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]
    env["cuda_env_vars"] = {
        k: os.environ.get(k, "not_set") for k in cuda_vars
    }

    return env


def _collect_pip_packages() -> List[str]:
    """Run pip freeze and return list of package==version strings."""
    # Try pip freeze first
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return [ln.strip() for ln in result.stdout.strip().splitlines()
                    if ln.strip() and not ln.startswith("#")]
    except Exception:
        pass

    # Fallback: conda list
    try:
        result = subprocess.run(
            ["conda", "list", "--export"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [ln.strip() for ln in result.stdout.strip().splitlines()
                     if ln.strip() and not ln.startswith("#")]
            return lines
    except Exception:
        pass

    return ["unavailable (pip and conda not accessible)"]


# ══════════════════════════════════════════════════════════════
#  MODEL PARAMETER COUNTING
# ══════════════════════════════════════════════════════════════

def count_parameters(model) -> Dict[str, Any]:
    """
    Count total, trainable, and frozen parameters.
    Also estimates memory footprint of parameters in MB.

    Works with any torch.nn.Module; returns empty dict on error.
    """
    try:
        import torch
        total      = sum(p.numel() for p in model.parameters())
        trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen     = total - trainable

        # Memory estimate: assume float32 (4 bytes) for all params
        mem_mb = round(total * 4 / 1024 / 1024, 2)

        # Get full class path
        cls = type(model)
        class_path = f"{cls.__module__}.{cls.__qualname__}"

        return {
            "total_params":              total,
            "trainable_params":          trainable,
            "frozen_params":             frozen,
            "parameter_memory_estimate_mb": mem_mb,
            "model_class":               cls.__name__,
            "model_class_path":          class_path,
        }
    except Exception as e:
        logger.debug(f"[system_utils] count_parameters failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
#  CONFIG HASH
# ══════════════════════════════════════════════════════════════

def config_hash(config) -> str:
    """Short MD5 hash of the config (10 chars) for experiment identity tracking."""
    try:
        try:
            import yaml
            s = yaml.dump(
                vars(config) if hasattr(config, "__dict__") else str(config),
                default_flow_style=True, sort_keys=True
            )
        except Exception:
            s = str(config)
        return hashlib.md5(s.encode()).hexdigest()[:10]
    except Exception:
        return "unavailable"


# ══════════════════════════════════════════════════════════════
#  JSON WRITE HELPER
# ══════════════════════════════════════════════════════════════

def atomic_json_write(obj: Any, path: str) -> None:
    """Atomically write a JSON file (tempfile + os.replace). Never raises."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        dir_ = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            os.close(fd)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=_json_default)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning(f"[system_utils] atomic_json_write failed for {path}: {e}")


def _json_default(obj: Any) -> Any:
    """Handle non-JSON-serializable types gracefully."""
    if isinstance(obj, (bool, int, float)):
        return obj
    return str(obj)


# ══════════════════════════════════════════════════════════════
#  DURATION FORMATTING
# ══════════════════════════════════════════════════════════════

def format_duration(seconds: Optional[float]) -> str:
    """Format seconds into human-readable string: '2h 34m 12s'."""
    if seconds is None or not isinstance(seconds, (int, float)):
        return "N/A"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"
