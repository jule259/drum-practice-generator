"""Separation backend interface.

Keeping this abstract is what makes the "don't hardcode one model" rule real:
the orchestration layer talks to ``SeparationBackend`` and never imports Demucs
directly.  A future MDX-Net or BSRNN backend only has to implement this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

# Callback invoked with fractional progress in [0, 1] plus a human-readable note.
# Returning normally means "keep going"; raising cancels the job.
ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


@dataclass
class SeparationResult:
    """Where the separated stems ended up, plus how it went."""

    drums: Path
    no_drums: Path
    sample_rate: int
    model: str = ""
    device: str = ""
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    stems: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 2),
            "processing_seconds": round(self.processing_seconds, 2),
            "realtime_factor": round(
                self.processing_seconds / self.duration_seconds, 3
            )
            if self.duration_seconds
            else None,
            "drums": str(self.drums),
            "no_drums": str(self.no_drums),
            "stems": {name: str(path) for name, path in self.stems.items()},
            "notes": list(self.notes),
        }


class SeparationCancelled(Exception):
    """Raised internally when the caller asks to stop."""


@runtime_checkable
class SeparationBackend(Protocol):
    """Contract every separation backend must satisfy."""

    name: str

    def is_available(self) -> tuple[bool, str]:
        """Can this backend run right now?  Returns (ok, reason_if_not)."""
        ...

    def separate(
        self,
        source_wav: Path,
        work_dir: Path,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> SeparationResult:
        """Separate ``source_wav`` (float32 WAV at the model rate) into stems."""
        ...
