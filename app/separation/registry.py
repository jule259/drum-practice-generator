"""Separation backend registry.

The orchestration layer resolves a backend by name through here, so adding an
MDX-Net or BSRNN implementation later means registering one class - not touching
the pipeline.
"""

from __future__ import annotations

from .base import SeparationBackend, SeparationResult  # noqa: F401  (re-export)
from .demucs_backend import DemucsBackend

# Registered backends.  "demucs" is the only production backend today; it covers
# htdemucs / htdemucs_ft / hdemucs_mmi as *model* choices.
_BACKENDS: dict[str, type] = {
    "demucs": DemucsBackend,
}


def available_backend_names() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str = "demucs", **kwargs) -> SeparationBackend:
    """Instantiate a backend by name.

    Raises ``KeyError`` for an unknown name - callers validate against
    ``available_backend_names()`` first and turn it into a user-facing error.
    """
    key = (name or "demucs").lower()
    if key not in _BACKENDS:
        raise KeyError(
            f"未知的分离后端：{name}（可用：{', '.join(available_backend_names())}）"
        )
    return _BACKENDS[key](**kwargs)


def register(name: str, backend_cls: type) -> None:
    """Register an additional backend implementation."""
    _BACKENDS[name.lower()] = backend_cls
