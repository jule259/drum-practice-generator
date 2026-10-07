"""Audio subsystem: I/O, FFmpeg integration, mixing, loudness and time-stretch.

Submodules are imported explicitly (``from app.audio import mix``) so that a
missing optional dependency in one area - for example FFmpeg - never prevents
the pure-numpy mixing code from loading.
"""

from __future__ import annotations

__all__ = ["io", "ffmpeg", "mix", "loudness", "timestretch"]
