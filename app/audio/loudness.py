"""Level measurement via FFmpeg's ``ebur128`` filter.

I deliberately do **not** hand-roll a BS.1770 loudness meter.  Implementing the
K-weighting biquads correctly for arbitrary sample rates is a real source of
subtle error, and FFmpeg - already a hard dependency - ships a reference
implementation.  So we measure with FFmpeg and mix with numpy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import ffmpeg as ffmpeg_module

__all__ = ["measure_loudness", "integrated_lufs", "peak_dbfs_of_file", "describe_levels"]

_INTEGRATED_RE = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)
_LRA_RE = re.compile(r"^\s*LRA:\s*(-?\d+(?:\.\d+)?)\s*LU", re.MULTILINE)


def _parse_ebur128(stderr_text: str) -> dict | None:
    """Parse the summary block FFmpeg prints for the ebur128 filter."""
    if not stderr_text:
        return None

    integrated = _INTEGRATED_RE.search(stderr_text)
    if not integrated:
        return None

    try:
        value = float(integrated.group(1))
    except ValueError:
        return None

    lra_match = _LRA_RE.search(stderr_text)
    lra = None
    if lra_match:
        try:
            lra = float(lra_match.group(1))
        except ValueError:
            lra = None

    return {"integrated_lufs": value, "lra": lra}


def measure_loudness(path: str | Path) -> dict | None:
    """Measure EBU R128 loudness of a file.  Returns None if unmeasurable.

    Never raises: loudness is informational, and a song that cannot be measured
    must not stop a practice track from being produced.
    """
    path = Path(path)
    if not path.is_file():
        return None

    info = ffmpeg_module.detect()
    if info is None or "ebur128" not in info.filters:
        return None

    try:
        proc = ffmpeg_module.run(
            [
                info.ffmpeg, "-hide_banner", "-nostats",
                "-i", str(path),
                "-filter_complex", "ebur128=peak=true",
                "-f", "null", "-",
            ],
            timeout=900,
            check=False,
        )
    except Exception:  # pragma: no cover - measurement is best-effort
        return None

    # ebur128 writes its summary to stderr.
    return _parse_ebur128(proc.stderr_text or "")


def integrated_lufs(path: str | Path) -> float | None:
    """Integrated loudness in LUFS, or None."""
    result = measure_loudness(path)
    return result["integrated_lufs"] if result else None


def peak_dbfs_of_file(path: str | Path) -> float | None:
    """True peak (dBFS) of an audio file, or None if it cannot be measured."""
    path = Path(path)
    if not path.is_file():
        return None

    info = ffmpeg_module.detect()
    if info is None or "astats" not in info.filters:
        return None

    try:
        proc = ffmpeg_module.run(
            [
                info.ffmpeg, "-hide_banner", "-nostats",
                "-i", str(path),
                "-filter_complex", "astats=metadata=1:reset=0",
                "-f", "null", "-",
            ],
            timeout=900,
            check=False,
        )
    except Exception:  # pragma: no cover
        return None

    text = proc.stderr_text or ""
    match = re.findall(r"Peak level dB:\s*(-?\d+(?:\.\d+)?|-?inf)", text)
    if not match:
        return None
    try:
        return float(match[-1])
    except ValueError:
        return None


def describe_levels(audio: np.ndarray) -> str:
    """Short human-readable level summary, used in logs."""
    from .io import peak_dbfs, rms_dbfs

    return f"peak={peak_dbfs(audio):.2f} dBFS rms={rms_dbfs(audio):.2f} dBFS"
