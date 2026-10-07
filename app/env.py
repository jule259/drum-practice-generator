"""Runtime environment probing: Python, PyTorch, CUDA, GPU, VRAM, FFmpeg, models.

Every function here is defensive: this module runs *before* we know the
environment is healthy, and on a fresh machine half of these things are missing.
Nothing raises - missing pieces are reported as structured data so
``diagnose.bat`` and the startup banner can print an honest picture instead of a
traceback.

Torch is imported lazily on purpose.  It costs ~2 s and ~400 MB of RSS, and the
mixing/self-test paths must work without it.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .tools import free_disk_bytes

# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
# "auto" (default) prefers CUDA and falls back to CPU.
# "cpu" forces CPU - used by tests and by the --cpu debug switch.
# "cuda" insists on GPU and raises if unavailable.
DEVICE_PREFERENCE = os.environ.get("DPG_DEVICE", "auto").lower()


@dataclass
class GPUInfo:
    index: int = 0
    name: str = ""
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    compute_capability: str = ""
    multi_processor_count: int = 0
    # Blackwell (RTX 50xx) is sm_120; the field exists so we can explain the
    # "your GPU is newer than your PyTorch build" failure mode in plain words.
    is_blackwell: bool = False


@dataclass
class EnvReport:
    os_name: str = ""
    os_version: str = ""
    machine: str = ""
    python_version: str = ""
    python_executable: str = ""
    python_bits: int = 0
    is_venv: bool = False
    torch_installed: bool = False
    torch_version: str = ""
    torch_cuda_build: str = ""
    torch_cuda_available: bool = False
    torch_cuda_version: str = ""
    cudnn_version: str = ""
    gpu: GPUInfo | None = None
    gpu_count: int = 0
    error: str = ""
    ffmpeg: dict[str, Any] = field(default_factory=dict)
    demucs_installed: bool = False
    demucs_version: str = ""
    sphn_installed: bool = False
    numpy_version: str = ""
    soundfile_installed: bool = False
    free_disk_gb: float = 0.0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def _module_version(name: str) -> str:
    """Version of an installed module without importing it, where possible."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def _is_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def torch_report(report: EnvReport) -> None:
    """Fill in every torch/CUDA field.  Never raises."""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        report.problems.append(
            "未安装 PyTorch。请在项目目录运行 install.bat。"
        )
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover - broken install
        report.problems.append(f"PyTorch 已安装但无法导入：{exc}")
        return

    report.torch_installed = True
    report.torch_version = getattr(torch, "__version__", "")
    report.torch_cuda_build = getattr(torch.version, "cuda", "") or ""

    try:
        report.torch_cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        report.problems.append(f"torch.cuda.is_available() 调用失败：{exc}")
        report.torch_cuda_available = False

    if not report.torch_cuda_available:
        if not report.torch_cuda_build:
            report.problems.append(
                "安装的是 CPU 版 PyTorch（版本号没有 +cuXXX 后缀）。"
                "无法使用 GPU 加速，请重新运行 install.bat。"
            )
        else:
            report.problems.append(
                "PyTorch 带 CUDA 支持，但检测不到可用 GPU。"
                "请确认 NVIDIA 驱动正常、显卡未被独占。"
            )
        return

    try:
        report.torch_cuda_version = torch.version.cuda or ""
    except Exception:
        pass

    try:
        report.cudnn_version = str(torch.backends.cudnn.version() or "")
    except Exception:
        pass

    try:
        report.gpu_count = torch.cuda.device_count()
    except Exception:
        report.gpu_count = 0

    try:
        index = torch.cuda.current_device()
    except Exception:
        index = 0

    try:
        props = torch.cuda.get_device_properties(index)
        capability = f"{props.major}.{props.minor}"
        info = GPUInfo(
            index=index,
            name=props.name,
            total_vram_mb=int(props.total_memory / (1024 * 1024)),
            compute_capability=capability,
            multi_processor_count=getattr(props, "multi_processor_count", 0),
            is_blackwell=(props.major, props.minor) >= (12, 0),
        )
        # Free VRAM is not in device_properties; ask the driver.
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            info.free_vram_mb = int(free_bytes / (1024 * 1024))
            if not info.total_vram_mb:
                info.total_vram_mb = int(total_bytes / (1024 * 1024))
        except Exception:
            info.free_vram_mb = info.total_vram_mb
        report.gpu = info
    except Exception as exc:
        report.problems.append(f"无法读取 GPU 属性：{exc}")


def build_report() -> EnvReport:
    """Collect the complete environment picture."""
    report = EnvReport(
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_bits=64 if sys.maxsize > 2**32 else 32,
        is_venv=_is_venv(),
    )

    report.numpy_version = _module_version("numpy")
    report.demucs_version = _module_version("demucs")
    report.demucs_installed = importlib.util.find_spec("demucs") is not None
    report.sphn_installed = importlib.util.find_spec("sphn") is not None
    report.soundfile_installed = importlib.util.find_spec("soundfile") is not None

    report.free_disk_gb = free_disk_bytes(config.OUTPUT_DIR) / (1024**3)

    # FFmpeg (lazy import to keep this module torch/ffmpeg-independent at import time)
    from .audio import ffmpeg as ffmpeg_module

    info = ffmpeg_module.detect()
    if info is None:
        report.ffmpeg = {"found": False}
        report.problems.append(
            "未找到 FFmpeg。运行 install.bat 自动下载，或手动安装后加入 PATH。"
        )
    else:
        report.ffmpeg = info.to_dict()
        report.problems.extend(info.problems)

    torch_report(report)

    if report.free_disk_gb and report.free_disk_gb < 2.0:
        report.warnings.append(
            f"磁盘可用空间仅 {report.free_disk_gb:.1f} GB，处理长歌曲可能失败。"
        )

    if not report.is_venv:
        report.warnings.append(
            "当前不在虚拟环境中运行。建议使用 start.bat 启动以隔离依赖。"
        )

    return report


def describe_gpu() -> str:
    """One-line GPU summary for the startup banner."""
    report = build_report()
    if report.gpu is None:
        return "CPU 模式（无可用 CUDA GPU）"
    gpu = report.gpu
    return (
        f"{gpu.name} | VRAM {gpu.total_vram_mb} MB "
        f"(free {gpu.free_vram_mb} MB) | sm_{gpu.compute_capability.replace('.', '')}"
    )


# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------
def resolve_device(preference: str | None = None) -> str:
    """Return the torch device string to use.

    Raises ``CUDAUnavailableError`` only when the caller explicitly demanded
    CUDA; ``auto`` silently degrades so a missing GPU never blocks the app.
    """
    preference = (preference or DEVICE_PREFERENCE or "auto").lower()

    if preference == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        if preference == "cuda":
            from .errors import CUDAUnavailableError

            raise CUDAUnavailableError("PyTorch 未安装。") from None
        return "cpu"

    available = False
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if available:
        return "cuda"
    if preference == "cuda":
        from .errors import CUDAUnavailableError

        raise CUDAUnavailableError()
    return "cpu"


def release_gpu_memory() -> None:
    """Free cached VRAM allocations.  Safe to call when torch is absent."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def is_out_of_memory(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a CUDA OOM?"""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    markers = (
        "out of memory",
        "outofmemory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "insufficient memory",
        "显存不足",
    )
    return any(marker in text for marker in markers)


def healthy(report: EnvReport | None = None) -> bool:
    """True when the environment can actually run a separation job."""
    report = report or build_report()
    return bool(
        report.torch_installed
        and report.ffmpeg.get("found")
        and report.demucs_installed
    )
