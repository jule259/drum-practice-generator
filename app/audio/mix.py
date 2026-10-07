"""The drum-level mixing core - the most important DSP in the project.

Spec section 4/5:

    output = vocals + bass + other + drums * drum_volume

Because we ask Demucs for a two-stem split with ``--other-method minus``, the
"everything except drums" part is ``original - drums_estimate`` rather than
``vocals + bass + other``.  That is deliberate: summing model estimates
accumulates each stem's artefacts, whereas subtracting the drum estimate from
the original keeps the residual phase-coherent with the source, so the backing
track sounds like the record with the drummer muted rather than a reconstruction.

Everything here is pure numpy - no FFmpeg, no torch - so it is fully unit
testable offline.  That matters: this is the code path that decides whether the
output sounds right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from .io import fit_shape, peak, peak_dbfs, rms_dbfs

__all__ = [
    "MixReport",
    "apply_gain",
    "mix_stems",
    "normalize_peak",
    "soft_limiter",
    "mix_with_drum_level",
]


@dataclass
class MixReport:
    """Diagnostics for one mix, surfaced in the log and the API response."""

    drum_volume: float = 0.0
    frames: int = 0
    sample_rate: int = 0
    peak_before: float = 0.0
    peak_after_limiter: float = 0.0
    limiter_reduction_db: float = 0.0
    normalized_gain_db: float = 0.0
    clipped: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "drum_volume": round(self.drum_volume, 4),
            "frames": self.frames,
            "sample_rate": self.sample_rate,
            "peak_before": round(self.peak_before, 6),
            "peak_dbfs_before": round(20 * np.log10(self.peak_before), 2)
            if self.peak_before > 0
            else None,
            "peak_after_limiter": round(self.peak_after_limiter, 6),
            "limiter_reduction_db": round(self.limiter_reduction_db, 3),
            "normalized_gain_db": round(self.normalized_gain_db, 3),
            "clipped": self.clipped,
            "notes": list(self.notes),
        }


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Multiply by a linear gain, staying in float32.

    Computed in float64 internally then cast back, because a 0.0 gain on a
    float32 array multiplied in place can leave -0.0 sentinels that show up as
    noise in some downstream meters.
    """
    if gain == 1.0:
        return audio
    if gain == 0.0:
        return np.zeros_like(audio)
    return (audio.astype(np.float64) * float(gain)).astype(np.float32)


def normalize_peak(
    audio: np.ndarray,
    target: float = config.PEAK_TARGET,
) -> tuple[np.ndarray, float]:
    """Scale so the absolute peak equals ``target``.

    Only ever *reduces* or mildly boosts: if the audio is already at or above
    target we scale down; if it is very quiet we boost, but the boost is capped
    so a near-silent stem cannot be amplified into a noise storm.

    Returns ``(audio, gain_applied)``.
    """
    current = peak(audio)
    if current <= 0.0:
        return audio, 1.0

    gain = target / current
    # Never boost more than +24 dB - protects against amplifying dither/noise.
    gain = min(gain, 10 ** (24.0 / 20.0))
    if abs(gain - 1.0) < 1e-6:
        return audio, 1.0
    return (audio.astype(np.float64) * gain).astype(np.float32), gain


def _smooth_gain(target_gain: np.ndarray, rate: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """One-pole attack/release smoothing of a per-frame gain curve.

    Fast attack so transients are actually caught; slow release so the gain
    does not "breathe" audibly between drum hits.
    """
    attack_coeff = float(np.exp(-1.0 / max(1.0, rate * attack_ms / 1000.0)))
    release_coeff = float(np.exp(-1.0 / max(1.0, rate * release_ms / 1000.0)))

    smoothed = np.empty_like(target_gain)
    current = float(target_gain[0])
    for index in range(target_gain.size):
        desired = float(target_gain[index])
        # Reducing gain is "attack"; restoring it is "release".
        coeff = attack_coeff if desired < current else release_coeff
        current = coeff * current + (1.0 - coeff) * desired
        smoothed[index] = current
    return smoothed


def soft_limiter(
    audio: np.ndarray,
    rate: int,
    ceiling: float = 0.97,
    lookahead_ms: float = 8.0,
    release_ms: float = 80.0,
) -> tuple[np.ndarray, float]:
    """Transparent look-ahead peak limiter.

    A hard ``clip`` would audibly distort the exact transients a drummer wants to
    hear, so instead we compute the gain envelope needed to keep the signal under
    ``ceiling`` and smooth it.  For the common case - a mix that only slightly
    exceeds the ceiling - this is inaudible gain riding.

    The attenuation starts exactly ``lookahead_ms`` before a transient: the
    forward sliding minimum propagates the required gain reduction backwards
    across the whole window, so the gain is already low when the peak arrives.
    (A second, backwards min pass would be a no-op - the forward pass already
    holds a constant value across the window, and the minimum of a constant is
    itself - so the reach is exactly one window.)

    Returns ``(audio, max_reduction_db)`` where reduction is negative dB.
    """
    if audio.size == 0:
        return audio, 0.0

    work = audio.astype(np.float64)
    mono = np.max(np.abs(work), axis=0)
    if mono.size == 0 or mono.max() <= ceiling:
        return audio, 0.0

    # Desired gain per sample: 1.0 unless the sample would exceed the ceiling.
    desired = np.ones_like(mono)
    over = mono > ceiling
    desired[over] = ceiling / np.maximum(mono[over], 1e-12)

    # Look-ahead: a peak must be attenuated *before* it arrives, so take a
    # sliding minimum forwards over the look-ahead window.  This holds the
    # reduction flat for the whole window preceding the transient.
    window = max(1, int(rate * lookahead_ms / 1000.0))
    if window > 1:
        padded = np.pad(desired, (0, window - 1), mode="edge")
        desired = np.lib.stride_tricks.sliding_window_view(padded, window).min(axis=1)

    smoothed = _smooth_gain(desired, rate, attack_ms=0.2, release_ms=release_ms)
    limited = (work * smoothed[None, :]).astype(np.float32)

    reduction = 20.0 * float(np.log10(max(float(smoothed.min()), 1e-12)))

    # Belt and braces: the smoothing can overshoot by a hair right at a peak.
    if peak(limited) > ceiling:
        limited = np.clip(limited, -ceiling, ceiling)

    return limited, reduction


def mix_stems(
    non_drum: np.ndarray,
    drums: np.ndarray,
    drum_volume: float,
    rate: int,
    *,
    drum_gain_db: float = 0.0,
    non_drum_gain_db: float = 0.0,
    do_normalize: bool = True,
    do_limit: bool = True,
    ceiling: float = config.PEAK_TARGET,
) -> tuple[np.ndarray, MixReport]:
    """Build the practice mix from the two Demucs stems.

    ``drum_volume`` is linear (0.0 = drums gone, 1.0 = original level).
    """
    drum_volume = float(np.clip(drum_volume, 0.0, 4.0))

    channels = max(non_drum.shape[0], drums.shape[0])
    frames = max(non_drum.shape[1], drums.shape[1])
    non_drum = fit_shape(non_drum, channels, frames)
    drums = fit_shape(drums, channels, frames)

    report = MixReport(drum_volume=drum_volume, frames=frames, sample_rate=rate)

    drum_gain = drum_volume * (10 ** (drum_gain_db / 20.0))
    bed_gain = 10 ** (non_drum_gain_db / 20.0)

    mixed = apply_gain(non_drum, bed_gain) + apply_gain(drums, drum_gain)

    report.peak_before = peak(mixed)

    if do_normalize:
        mixed, gain = normalize_peak(mixed, ceiling)
        report.normalized_gain_db = 20.0 * float(np.log10(gain)) if gain > 0 else 0.0
        if gain < 1.0:
            report.notes.append(
                f"峰值归一化：下调 {abs(report.normalized_gain_db):.2f} dB 以避免削波"
            )
        elif gain > 1.0:
            report.notes.append(f"峰值归一化：提升 {report.normalized_gain_db:.2f} dB")

    if do_limit:
        mixed, reduction = soft_limiter(mixed, rate, ceiling=ceiling)
        report.limiter_reduction_db = reduction
        if reduction < -0.1:
            report.notes.append(f"限幅器衰减 {abs(reduction):.2f} dB")

    report.peak_after_limiter = peak(mixed)
    report.clipped = report.peak_after_limiter > 1.0
    if report.clipped:
        report.notes.append("警告：输出仍超过 0 dBFS")

    return mixed.astype(np.float32), report


def mix_with_drum_level(
    non_drum: np.ndarray,
    drums: np.ndarray,
    drum_volume: float,
    rate: int,
    **kwargs,
) -> tuple[np.ndarray, MixReport]:
    """Convenience alias with a spec-shaped name."""
    return mix_stems(non_drum, drums, drum_volume, rate, **kwargs)


def describe(audio: np.ndarray) -> str:
    return f"peak={peak_dbfs(audio):.2f}dBFS rms={rms_dbfs(audio):.2f}dBFS"
