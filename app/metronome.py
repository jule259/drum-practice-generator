"""Metronome click-track synthesis.

The click is synthesised in numpy rather than loaded from a WAV asset: no binary
files in the repo, no licence questions, and the length automatically matches the
track being generated.

**Timing model - important and documented for the user.**  Clicks are placed on a
grid starting at t=0 at the *target* BPM (i.e. the tempo of the finished practice
track).  Detection accuracy of the song's actual downbeat is a different problem,
so ``offset_ms`` lets the user nudge the grid.  We deliberately do not claim
bar-alignment with the record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

__all__ = ["ClickSpec", "generate_click", "generate_click_track", "MetronomeSettings"]


@dataclass
class ClickSpec:
    """One synthesised click voice."""

    frequency: float = 1000.0
    duration_ms: float = 30.0
    decay_ms: float = 8.0
    gain: float = 1.0


# Downbeat is higher and louder - the classic "tick-tock" so a drummer can hear
# where bar one is without counting.
ACCENT_CLICK = ClickSpec(frequency=1500.0, duration_ms=34.0, decay_ms=9.0, gain=1.0)
NORMAL_CLICK = ClickSpec(frequency=1000.0, duration_ms=26.0, decay_ms=6.5, gain=0.62)


@dataclass
class MetronomeSettings:
    enabled: bool = False
    bpm: float = 120.0
    volume: float = config.DEFAULT_METRONOME_VOLUME
    time_signature: str = config.DEFAULT_TIME_SIGNATURE
    offset_ms: float = 0.0

    @property
    def beats_per_bar(self) -> int:
        return config.TIME_SIGNATURES.get(
            self.time_signature, config.TIME_SIGNATURES[config.DEFAULT_TIME_SIGNATURE]
        )["beats_per_bar"]

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "bpm": round(self.bpm, 2),
            "volume": round(self.volume, 4),
            "time_signature": self.time_signature,
            "offset_ms": round(self.offset_ms, 2),
            "beats_per_bar": self.beats_per_bar,
        }


def generate_click(spec: ClickSpec, rate: int) -> np.ndarray:
    """Render a single click: a decaying sine with a short fade-in.

    The 1 ms fade-in removes the DC step at onset, which is what makes a naive
    sine-burst click sound like a "pop" through a PA.
    """
    length = max(1, int(rate * spec.duration_ms / 1000.0))
    times = np.arange(length, dtype=np.float64) / rate

    envelope = np.exp(-times / max(spec.decay_ms / 1000.0, 1e-6))
    fade_samples = max(1, int(rate * 0.001))
    if fade_samples < length:
        envelope[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)

    tone = np.sin(2.0 * np.pi * spec.frequency * times)
    return (tone * envelope * spec.gain).astype(np.float32)


def _beat_is_accent(beat_index: int, beats_per_bar: int, time_signature: str) -> bool:
    """Which beats in the bar get the loud/high click."""
    position = beat_index % beats_per_bar
    if position == 0:
        return True
    # 6/8 is felt in two: accents on beat 1 and beat 4.
    if time_signature == "6/8" and position == 3:
        return True
    return False


def generate_click_track(
    settings: MetronomeSettings,
    rate: int,
    total_frames: int,
    *,
    channels: int = 1,
) -> np.ndarray:
    """Build a click track of exactly ``total_frames`` frames.

    Returns ``(channels, total_frames)`` float32.  An all-zero track is returned
    when the metronome is disabled or the parameters are nonsensical, so callers
    never need a special case.
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return np.zeros((channels, 0), dtype=np.float32)

    empty = np.zeros((channels, total_frames), dtype=np.float32)
    if not settings.enabled:
        return empty
    if settings.bpm <= 0 or not np.isfinite(settings.bpm):
        return empty
    if settings.volume <= 0:
        return empty

    beats_per_bar = max(1, settings.beats_per_bar)
    accent = generate_click(ACCENT_CLICK, rate)
    normal = generate_click(NORMAL_CLICK, rate)

    seconds_per_beat = 60.0 / float(settings.bpm)
    step_frames = seconds_per_beat * rate
    if step_frames < 1:
        return empty

    offset_frames = settings.offset_ms / 1000.0 * rate
    # Cover every beat that could start before the end, plus one for the tail.
    count = int(np.ceil((total_frames - offset_frames) / step_frames)) + 1
    count = max(0, min(count, 2_000_000))  # absurd-parameter guard

    mono = np.zeros(total_frames, dtype=np.float32)
    for beat_index in range(count):
        start = int(round(offset_frames + beat_index * step_frames))
        if start >= total_frames:
            break
        source = accent if _beat_is_accent(beat_index, beats_per_bar, settings.time_signature) else normal

        # A click may hang over the end of the track; clip it.
        available = min(source.size, total_frames - start)
        if available <= 0:
            continue
        mono[start : start + available] += source[:available]

    mono *= float(np.clip(settings.volume, 0.0, 4.0))

    if channels == 1:
        return mono[None, :]
    return np.repeat(mono[None, :], channels, axis=0)


def click_count(settings: MetronomeSettings, rate: int, total_frames: int) -> int:
    """How many clicks land inside the track - used for the log line."""
    if not settings.enabled or settings.bpm <= 0:
        return 0
    step = 60.0 / settings.bpm * rate
    if step < 1:
        return 0
    offset = settings.offset_ms / 1000.0 * rate
    return max(0, int(np.ceil((total_frames - offset) / step)))
