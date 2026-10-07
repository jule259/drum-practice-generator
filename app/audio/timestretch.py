"""Pitch-preserving time-stretch with a pluggable backend.

Two backends, chosen automatically:

``rubberband``
    Rubber Band Library CLI (``rubberband.exe``).  Formant-preserving, the
    quality reference for musical time-stretching.  Optional: requires the user
    to drop the binary into ``tools/rubberband`` or have it on PATH.

``atempo``
    FFmpeg's built-in WSOLA stretcher.  Always available because FFmpeg is a hard
    dependency.  Slightly more smearing on transient material at extreme ratios,
    but entirely usable and it is what runs out of the box.

The spec asked for "Rubber Band preferred, atempo acceptable".  Rather than
forcing every user to install a third-party binary, we use atempo by default and
upgrade transparently when Rubber Band is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..errors import DrumPracticeError, FFmpegRunError
from ..tools import run, which
from . import ffmpeg as ffmpeg_module

BUNDLED_RB_DIRS = (
    config.TOOLS_DIR / "rubberband" / "bin",
    config.TOOLS_DIR / "rubberband",
)

# Rubber Band's quality presets.  -3 is "high quality, slower" which is the right
# trade-off offline; transients (drum hits) benefit noticeably from it.
RB_QUALITY = "3"       # corresponds to -3 / --fine
RB_PITCH = "0"         # no pitch shift - we only change tempo


@dataclass
class StretchInfo:
    backend: str
    tempo: float
    input_duration: float = 0.0
    output_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "tempo": round(self.tempo, 4),
            "input_duration": round(self.input_duration, 3),
            "output_duration": round(self.output_duration, 3),
        }


def find_rubberband() -> Path | None:
    """Locate a working Rubber Band CLI, or None."""
    for directory in BUNDLED_RB_DIRS:
        for exe in ("rubberband.exe", "rubberband"):
            candidate = directory / exe
            if candidate.is_file():
                return candidate

    found = which("rubberband")
    if found is None:
        return None

    # Verify it actually runs - a Store alias stub would otherwise "exist".
    try:
        proc = run([found, "--version"], timeout=20, check=False)
        if proc.returncode != 0:
            # Some builds only support -h / no --version; fall back to -h.
            proc = run([found, "-h"], timeout=20, check=False)
        if proc.returncode != 0:
            return None
    except Exception:
        return None
    return found


def available_backends() -> dict[str, bool]:
    """Which stretch backends the machine currently supports."""
    rb = find_rubberband()
    info = ffmpeg_module.detect()
    return {
        "atempo": bool(info and info.has_atempo),
        "rubberband": rb is not None,
        "rubberband_path": str(rb) if rb else None,
    }


def _rubberband_stretch(
    binary: Path,
    src: Path,
    dst: Path,
    tempo: float,
    sample_rate: int,
) -> None:
    """Invoke the Rubber Band CLI.

    ``--time`` takes a *ratio* where >1 means faster, matching our tempo
    convention.  ``--pitch 0`` guarantees the pitch is untouched.
    """
    args = [
        binary,
        "--time", f"{tempo:.10f}",
        "--pitch", RB_PITCH,
        "-q", RB_QUALITY,
        "--fine",
        "--threads", str(min(8, os.cpu_count() or 4)),
        str(src),
        str(dst),
    ]
    proc = run(args, timeout=3600, check=False)
    if proc.returncode != 0 or not dst.is_file():
        raise FFmpegRunError(
            "Rubber Band 变速失败。",
            detail=(proc.stderr_text or "")[-2000:],
            suggestions=["改用 FFmpeg atempo 后端（在高级设置中切换）。"],
        )


def stretch(
    src: str | Path,
    dst: str | Path,
    *,
    tempo: float,
    sample_rate: int = config.MODEL_SAMPLE_RATE,
    channels: int = config.MODEL_CHANNELS,
    backend: str = "auto",
) -> StretchInfo:
    """Time-stretch ``src`` into ``dst`` without changing pitch.

    ``tempo`` is the speed multiplier: 0.75 = 75% speed = slower.
    ``backend`` is ``"auto"``, ``"atempo"`` or ``"rubberband"``.
    """
    if tempo <= 0:
        raise DrumPracticeError("速度倍率必须大于 0。")

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise DrumPracticeError(f"待变速文件不存在：{src.name}")

    if abs(tempo - 1.0) < 1e-6 and backend != "rubberband":
        # No-op: copy through so callers always get a file at ``dst``.
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return StretchInfo(backend="none", tempo=1.0)

    chosen = backend
    if backend == "auto":
        chosen = "rubberband" if find_rubberband() else "atempo"

    if chosen == "rubberband":
        binary = find_rubberband()
        if binary is None:
            if backend == "rubberband":
                raise DrumPracticeError(
                    "选择了 Rubber Band 后端，但未找到 rubberband.exe。",
                    suggestions=[
                        "把 rubberband.exe 放到 tools\\rubberband 目录，",
                        "或改用 FFmpeg atempo 后端。",
                    ],
                )
            chosen = "atempo"
        else:
            # Rubber Band writes the format based on the output extension.
            _rubberband_stretch(binary, src, dst, tempo, sample_rate)
            return StretchInfo(backend="rubberband", tempo=tempo)

    ffmpeg_module.time_stretch(
        src, dst, tempo=tempo, sample_rate=sample_rate, channels=channels
    )
    return StretchInfo(backend="atempo", tempo=tempo)
