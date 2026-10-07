"""Model cache management for Demucs.

Demucs 4.1.0 loads pretrained weights through ``huggingface_hub.hf_hub_download``,
which caches into the *HuggingFace* cache (``%USERPROFILE%\\.cache\\huggingface``
by default).  Spec section 14 wants models under the project's ``models/``
directory, so we redirect the HF cache with environment variables **set before
torch/huggingface_hub are imported**.

Nothing is downloaded behind the user's back: ``is_installed`` only inspects the
cache, and ``download`` is called from an explicit UI action.  Model provenance is
the official ``adefossez/*`` HuggingFace repositories published by the Demucs
author, with the legacy Meta AWS mirror as a fallback by Demucs itself.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import config

# Namespace confirmed from demucs/hf.py DEFAULT_NAMESPACE
HF_NAMESPACE = "adefossez"

# hf.py: hf_repo_name()'s mapping
_HF_REPO_OVERRIDES = {"htdemucs": "HTDemucs"}


def hf_repo_name(model: str) -> str:
    """Mirror of demucs.hf.hf_repo_name so we can inspect the cache ourselves."""
    if model in _HF_REPO_OVERRIDES:
        return _HF_REPO_OVERRIDES[model]
    if model.startswith("htdemucs_"):
        return "HTDemucs-" + model[len("htdemucs_") :]
    return "Demucs-" + model


def models_root() -> Path:
    """Project-local model storage root."""
    return config.MODELS_DIR


def configure_cache() -> Path:
    """Point every model cache at ``models/`` and return the HF root.

    Must be called before importing huggingface_hub/torch.

    Three separate caches are redirected, because Demucs can reach for any of
    them depending on which lookup succeeds:

    * ``HF_HOME`` / ``HF_HUB_CACHE`` - the modern path.  ``htdemucs`` is fetched
      from the ``adefossez/HTDemucs`` HuggingFace repo.
    * ``TORCH_HOME`` - the *legacy* path.  ``demucs.repo.ModelOnlyRepo`` falls
      back to ``torch.hub.load_state_dict_from_url`` against Meta's AWS mirror,
      and that writes to ``%USERPROFILE%\\.cache\\torch``.  Observed in practice:
      when the HF lookup does not complete, that fallback raised
      ``PermissionError: [WinError 5] ... '\\.cache'`` and model loading failed
      outright.  Redirecting TORCH_HOME removes the dependency on a writable
      user-profile cache entirely.

    ``setdefault`` is used throughout so an explicit user-provided value wins.
    """
    root = models_root() / "huggingface"
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    # Avoid the symlink farm blowing up on Windows without developer mode; the
    # duplicate-name warning is harmless and only printed once.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    torch_home = models_root() / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    return root


def hub_cache_dir() -> Path:
    """Resolved HuggingFace hub cache directory."""
    override = os.environ.get("HF_HUB_CACHE")
    if override:
        return Path(override)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    # Default location used by huggingface_hub on Windows.
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class ModelStatus:
    signature: str
    display_name: str
    description: str
    installed: bool
    size_mb: int
    cache_path: str = ""
    present_files: int = 0
    expected_files: int = 0

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "display_name": self.display_name,
            "description": self.description,
            "installed": self.installed,
            "size_mb": self.size_mb,
            "cache_path": self.cache_path,
            "present_files": self.present_files,
            "expected_files": self.expected_files,
        }


def _model_files(model: str) -> tuple[list[Path], bool]:
    """Return (expected cache snapshot dirs, whether the repo dir exists)."""
    cache = hub_cache_dir()
    repo_dir = cache / f"models--{HF_NAMESPACE}--{hf_repo_name(model)}"
    return ([repo_dir] if repo_dir.is_dir() else []), repo_dir.is_dir()


def is_installed(model: str) -> bool:
    """True when the model's weights are already in the local cache.

    Two storage layouts must be recognised, because Demucs can obtain a model
    either way:

    * modern - ``snapshots/<commit>/<sig>.safetensors`` from the HF hub;
    * legacy - a flat ``<sig>.th`` checkpoint fetched via ``torch.hub`` from
      Meta's AWS mirror (``demucs.repo.ModelOnlyRepo``).
    """
    dirs, exists = _model_files(model)
    if not exists:
        return False
    repo_dir = dirs[0]

    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            if (snapshot / f"{model}.yaml").is_file():
                return True
            if any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.th")):
                return True
        return False

    # Legacy flat layout: checkpoints sit directly in the repo/checkpoint dir.
    if any(repo_dir.glob("*.th")) or any(repo_dir.glob("*.safetensors")):
        return True
    return False


def _count_checkpoints(snapshot: Path) -> int:
    """Count downloaded weight files, whatever the storage format."""
    return sum(
        1 for pattern in ("*.safetensors", "*.th") for _ in snapshot.glob(pattern)
    )


def status(model: str) -> ModelStatus:
    """Full status for one model signature."""
    meta = config.AVAILABLE_MODELS.get(model, {})
    dirs, exists = _model_files(model)
    repo_dir = dirs[0] if dirs else hub_cache_dir() / f"models--{HF_NAMESPACE}--{hf_repo_name(model)}"

    present = 0
    if exists:
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            for snapshot in snapshots_dir.glob("*"):
                if snapshot.is_dir():
                    present += _count_checkpoints(snapshot)
        else:
            present += _count_checkpoints(repo_dir)

    return ModelStatus(
        signature=model,
        display_name=meta.get("name", model),
        description=meta.get("desc", ""),
        # A model counts as installed if either cache has it: the hub cache or
        # our offline local checkpoint.
        installed=is_installed(model) or local_checkpoint(model) is not None,
        size_mb=meta.get("size_mb", 0),
        cache_path=str(repo_dir),
        present_files=present,
        # htdemucs has 1 sub-model, htdemucs_ft / hdemucs_mmi have 4
        expected_files=4 if ("_ft" in model or "mmi" in model) else 1,
    )


def all_statuses() -> list[ModelStatus]:
    return [status(sig) for sig in config.AVAILABLE_MODELS]


def download(model: str, *, progress=None) -> Path:
    """Download a model's weights into the project cache.

    Runs ``demucs.pretrained.get_model`` in-process, which is exactly the same
    code path inference uses - so a successful download is proof the model will
    load, rather than just proof that bytes arrived.

    ``progress`` is an optional callable ``(fraction, message) -> None``.
    """
    from ..env import release_gpu_memory
    from ..errors import ModelDownloadError

    configure_cache()

    try:
        from demucs.pretrained import get_model
    except Exception as exc:
        raise ModelDownloadError(
            model, detail=f"无法导入 demucs：{exc}"
        ) from exc

    if progress:
        progress(0.05, f"正在下载模型 {model}（来自 HuggingFace: {HF_NAMESPACE}/{hf_repo_name(model)}）…")

    try:
        # get_model() performs the actual hub download and caches it.
        loaded = get_model(name=model)
    except Exception as exc:
        raise ModelDownloadError(model, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        release_gpu_memory()

    if progress:
        progress(1.0, "模型下载完成")

    if loaded is None:
        raise ModelDownloadError(model, detail="get_model() 返回 None")

    # Free it immediately - we only wanted the bytes on disk.
    del loaded
    release_gpu_memory()

    # Promote the freshly downloaded weights into the canonical offline cache so
    # later runs never need the network again.  See `local_checkpoint`.
    if progress:
        progress(0.98, "正在整理模型缓存…")
    cached = _promote_to_cache(model)
    if cached is not None and progress:
        progress(1.0, f"模型已缓存：{cached.name}")

    dirs, _ = _model_files(model)
    return dirs[0] if dirs else hub_cache_dir()


# --------------------------------------------------------------------------
# Offline local cache
# --------------------------------------------------------------------------
# Demucs has two remote sources and, on a slow or filtered route, both are
# painful:
#
#   * HuggingFace (`adefossez/HTDemucs`) - the intended path.  Measured here:
#     connecting to huggingface.co times out (WinError 10060) and the hub retries
#     for ~5 minutes before giving up;
#   * the legacy AWS mirror via `torch.hub.load_state_dict_from_url`, which
#     re-validated and re-downloaded the 80 MB checkpoint on every single load.
#
# So after a successful download we copy the checkpoint to a plain file under
# `models/cache/` and subsequently load *that file directly* - no network, no
# retries, no re-download.  `demucs.states.load_model` accepts a path, and for
# safetensors we load it ourselves.

def canonical_dir() -> Path:
    path = models_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_already_cached(file: Path) -> bool:
    try:
        return canonical_dir() in file.parents
    except OSError:
        return False


def _discover_checkpoint(model: str) -> Path | None:
    """Find the most recently written checkpoint belonging to ``model``.

    ``hf_repo_name`` gives the sub-model signature (htdemucs -> HTDemucs spec
    files are hashed, while the legacy AWS checkpoint is a UUID), so we match on
    the signature appearing in the filename and fall back to the newest file.
    """
    roots = [models_root()]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or _is_already_cached(path):
                continue
            if path.suffix not in (".th", ".safetensors", ".pt", ".bin"):
                continue
            if _looks_like_model_weights(path):
                candidates.append(path)

    if not candidates:
        return None

    signature = model.split("/")[-1]
    named = [p for p in candidates if signature.lower() in p.name.lower()]
    pool = named or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _looks_like_model_weights(path: Path) -> bool:
    """Reject stray files by size - model checkpoints are tens of MB."""
    try:
        return path.stat().st_size > 5 * 1024 * 1024
    except OSError:
        return False


def _promote_to_cache(model: str) -> Path | None:
    """Copy the downloaded checkpoint into ``models/cache/<model>.th``."""
    source = _discover_checkpoint(model)
    if source is None:
        return None

    target = canonical_dir() / f"{model}{source.suffix}"
    try:
        if source.resolve() == target.resolve():
            return target
        shutil.copy2(source, target)
    except OSError:
        return None
    return target


def local_checkpoint(model: str) -> Path | None:
    """Path to the offline checkpoint for ``model``, or None if not cached."""
    directory = models_root() / "cache"
    for suffix in (".th", ".safetensors", ".pt", ".bin"):
        candidate = directory / f"{model}{suffix}"
        if candidate.is_file() and _looks_like_model_weights(candidate):
            return candidate
    return None


def load_local_model(model: str):
    """Load a cached checkpoint without any network access.

    Returns the Demucs model, or ``None`` when nothing is cached.
    """
    checkpoint = local_checkpoint(model)
    if checkpoint is None:
        return None

    from demucs.states import load_model

    if checkpoint.suffix == ".safetensors":
        from demucs.hf import hf_repo_name as _ignored  # noqa: F401  (ensures module loads)
        from demucs.hf import load_safetensors_model

        loaded = load_safetensors_model(checkpoint)
    else:
        loaded = load_model(checkpoint)
    loaded.eval()
    return loaded


def cache_size_mb() -> float:
    """Total size of the project model cache in MB."""
    root = models_root()
    if not root.is_dir():
        return 0.0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def license_info() -> list[dict]:
    """Provenance/licence disclosure shown in the UI (spec section 14)."""
    return [
        {
            "component": "Demucs v4 (htdemucs / htdemucs_ft / hdemucs_mmi)",
            "license": "MIT",
            "source": f"https://huggingface.co/{HF_NAMESPACE}",
            "note": "权重由 Demucs 作者官方发布；MIT 许可允许个人与商业使用。",
        },
        {
            "component": "FFmpeg",
            "license": "LGPL v2.1+ / GPL (取决于构建)",
            "source": "https://ffmpeg.org",
            "note": "install.bat 下载的是 gyan.dev 的 LGPL 共享构建，可自由分发使用。",
        },
        {
            "component": "Rubber Band Library（可选）",
            "license": "GPL v2+",
            "source": "https://breakfastquay.com/rubberband/",
            "note": "仅当你手动放入 rubberband.exe 时启用；GPL 影响二次分发，个人使用无限制。",
        },
    ]
