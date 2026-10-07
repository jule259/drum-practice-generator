"""Automatic BPM detection.

Spec section 7: detect the tempo on import, let the user override it, and
**never block the workflow** when detection is unreliable.

This is a numpy implementation of the standard onset-envelope -> tempo-prior ->
dynamic-programming beat-tracker pipeline (the same family of algorithm as
librosa's ``beat_track``).  I implemented it rather than depending on librosa
because:

* librosa 1.0 requires Python >= 3.12 while we target 3.11;
* pulling a large DSP dependency purely for one number is poor trade;
* the algorithm is ~150 lines and fully testable offline.

Everything degrades gracefully: if anything fails, ``detect`` returns a result
with ``ok=False`` and a reason, and the caller simply leaves the BPM box empty
for the user to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BpmResult", "detect", "detect_bpm", "onset_envelope", "tempo_from_envelope"]

# Analysis window.  Tempo is a global property; 90 s is plenty and keeps the
# STFT to a couple of seconds of CPU on a 5-minute song.
ANALYSIS_SECONDS = 90.0
HOP_LENGTH = 512
N_FFT = 2048
MEL_BANDS = 96
FMIN = 27.5
FMAX = 8000.0

# Factor by which the raw onset envelope is decimated before the tempo search.
# Kept small so the envelope still has ~10 frames per beat at 200 BPM; coarser
# frames make neighbouring tempi impossible to tell apart.
ENVELOPE_DECIMATION = 2

# Tempo range we trust for a practice track.  Outside it we look for the
# musically equivalent half/double within range.
PREFERRED_MIN = 65.0
PREFERRED_MAX = 185.0

# Absolute search bounds (some genres sit outside the preferred window).
MIN_BPM = 40.0
MAX_BPM = 240.0

# Tempo prior: log-normal around 120 BPM.  Without it, the autocorrelation is
# equally happy with a tempo and its octave, and the octave that sits closer to
# 120 wins for no musical reason.
PRIOR_CENTER = 120.0
PRIOR_STD = 1.0


@dataclass
class BpmResult:
    ok: bool
    bpm: float | None = None
    confidence: float = 0.0
    beat_times: list[float] = field(default_factory=list)
    reason: str = ""
    alternatives: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "bpm": round(self.bpm, 2) if self.bpm else None,
            "bpm_rounded": int(round(self.bpm)) if self.bpm else None,
            "confidence": round(self.confidence, 4),
            "beats_detected": len(self.beat_times),
            "alternatives": [round(float(value), 2) for value in self.alternatives],
            "reason": self.reason,
        }


# --------------------------------------------------------------------------
# Signal processing helpers
# --------------------------------------------------------------------------
def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=0).astype(np.float32)


def _hann_window(size: int) -> np.ndarray:
    return np.hanning(size + 1)[:size].astype(np.float32)


def _stft_magnitude(mono: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, shape ``(n_fft//2 + 1, frames)``."""
    if mono.size < N_FFT:
        mono = np.pad(mono, (0, N_FFT - mono.size))

    window = _hann_window(N_FFT)
    # frame the signal: (frames, N_FFT) then zero-pad the tail
    frames = np.lib.stride_tricks.sliding_window_view(mono, N_FFT)[::HOP_LENGTH]
    frames = frames * window[None, :]
    spectrum = np.fft.rfft(frames, axis=1)
    return np.abs(spectrum).T.astype(np.float32)


def _hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(rate: int, n_fft: int = N_FFT, bands: int = MEL_BANDS) -> np.ndarray:
    """Triangular mel filterbank, shape ``(bands, n_fft//2 + 1)``."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, rate / 2.0, n_bins)

    mel_points = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(min(FMAX, rate / 2.0)), bands + 2)
    hz_points = _mel_to_hz(mel_points)

    bank = np.zeros((bands, n_bins), dtype=np.float32)
    for index in range(bands):
        lower, center, upper = hz_points[index], hz_points[index + 1], hz_points[index + 2]
        if upper - lower <= 0:
            continue
        left = (fft_freqs - lower) / max(center - lower, 1e-9)
        right = (upper - fft_freqs) / max(upper - center, 1e-9)
        bank[index] = np.clip(np.minimum(left, right), 0.0, None)
    return bank


def _tempo_prior(bpms: np.ndarray) -> np.ndarray:
    """Log-normal weighting favouring musically plausible tempi.

    Without it, a comb at half the true tempo also fires reasonably well (every
    other beat still lands on a hit), and at double tempo half its beats land on
    off-beat hi-hats.
    """
    return np.exp(-0.5 * ((np.log2(bpms / PRIOR_CENTER)) / PRIOR_STD) ** 2)


def _downsample_envelope(envelope: np.ndarray, factor: int) -> np.ndarray:
    """Average ``factor`` consecutive frames to raise the frame period.

    The tempo search needs a fine frame resolution: at the raw 86 Hz frame rate a
    120 BPM period is 21.5 frames, so the nearest integer lags (21 and 22) sit on
    harmonic-looking periodicities and a plain autocorrelation picks nonsense.
    Presenting integer lag *errors* below ~0.15 frames makes the comb search
    land on the true tempo.
    """
    if factor <= 1 or envelope.size < factor * 8:
        return envelope
    usable = (envelope.size // factor) * factor
    return envelope[:usable].reshape(-1, factor).mean(axis=1)


def onset_envelope(audio: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength envelope.

    Returns ``(envelope, hop_in_seconds)`` where the envelope is a z-scored
    (zero-mean, unit-variance) onset strength curve.  Only *increases* in energy
    count, which is what makes onsets stand out from sustained notes.

    The zero mean matters for the tempo search: the comb filter's per-beat
    average is only comparable across tempi when the baseline is zero, otherwise
    longer grids accumulate more positive bias.
    """
    mono = _to_mono(audio)
    magnitude = _stft_magnitude(mono)
    if magnitude.shape[1] < 3:
        return np.zeros(0, dtype=np.float32), HOP_LENGTH / rate

    bank = _mel_filterbank(rate)
    mel = bank @ magnitude

    # log compression (dB) then half-wave-rectified first difference
    ref = max(float(mel.max()), 1e-10)
    log_mel = 20.0 * np.log10(np.maximum(mel, 1e-10) / ref)
    flux = np.diff(log_mel, axis=1)
    envelope = np.maximum(flux, 0.0).mean(axis=0)

    # Remove the local mean so a loud chorus does not dominate the whole track.
    if envelope.size > 16:
        kernel = np.ones(16, dtype=np.float32) / 16.0
        baseline = np.convolve(envelope, kernel, mode="same")
        envelope = np.maximum(envelope - baseline, 0.0)

    envelope = envelope.astype(np.float32)
    std = float(envelope.std())
    if std > 0:
        envelope = (envelope - envelope.mean()) / std

    envelope = _downsample_envelope(envelope, ENVELOPE_DECIMATION)
    # Re-centre after decimation: averaging changes the mean slightly.
    envelope = envelope - float(envelope.mean())
    return envelope, (HOP_LENGTH * ENVELOPE_DECIMATION) / rate


def _comb_energy(signal: np.ndarray, stride: int) -> tuple[float, int]:
    """Best lag-aligned onset energy for a pulse train of period ``stride``.

    This is a phase-aligned comb filter expressed as a strided-window sum: for
    every candidate phase offset we add up the envelope at frames
    ``offset, offset+stride, ...``.  The result is divided by the number of beats
    summed, so the score is an *average onset strength per beat* and is
    comparable across tempi - otherwise the fastest candidate would always win.

    Why not autocorrelation?  ACF(tau) sums ``x[n]*x[n-tau]``, and a signal that
    is quasi-periodic at ``tau/2`` can maximise that instead - exactly the octave
    error this replaces.  Summing onset strength *on the grid* asks the question
    that actually matters: "do the beats land where the hits are?"

    Returns ``(energy_per_beat, best_offset)``.
    """
    if stride <= 0 or signal.size < stride * 3:
        return float("-inf"), 0
    if stride == 1:
        return float(signal.mean()), 0

    length = signal.size
    # Trim to a whole number of periods so every phase offset covers the same
    # number of beats and the per-beat averages are directly comparable.
    n_beats = int(np.floor(length / stride))
    span = n_beats * stride
    if n_beats < 3:
        return float("-inf"), 0

    segment = signal[:span]
    windows = np.lib.stride_tricks.sliding_window_view(segment, n_beats)[::stride]
    if windows.shape[0] == 0:
        return float("-inf"), 0

    scores = windows.sum(axis=1) / float(n_beats)
    best_offset = int(np.argmax(scores))
    return float(scores[best_offset]), best_offset


def _global_acf(
    signal: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-normalised global autocorrelation over the candidate lags.

    Normalising by the number of overlapping samples (floored at half the signal
    so the estimate stays stable at long lags) is what lets tempi across the
    whole 40-240 BPM range be compared on one scale; the raw autocorrelation
    systematically favours short lags.
    """
    signal = signal - signal.mean()
    count = signal.size
    size = 1
    while size < 2 * count:
        size *= 2
    spectrum = np.fft.rfft(signal, n=size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[:count]

    lags = np.arange(min_lag, max_lag + 1)
    overlap = np.maximum(count - lags, count * 0.5)
    values = acf[lags] / overlap

    peak = float(np.max(values)) if values.size else 0.0
    if peak > 0:
        values = values / peak
    return lags, values


def tempo_from_envelope(
    envelope: np.ndarray,
    hop_seconds: float,
    *,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
    preferred_min: float = PREFERRED_MIN,
    preferred_max: float = PREFERRED_MAX,
) -> tuple[float, float]:
    """Estimate the beat tempo of an onset envelope.

    Autocorrelation of the onset envelope weighted by a musical tempo prior,
    followed by a parabolic refinement of the winning lag.

    **Known limitation, by design.**  Periodicity alone cannot always separate a
    tempo from its octave: a pattern repeating every two beats is genuinely
    periodic at half the beat rate.  So when the winner falls outside
    ``preferred_min..preferred_max`` we prefer the octave that lands inside the
    practice range, and the UI always lets the user override the value.

    Returns ``(bpm, confidence)``; ``bpm`` is None when no estimate is possible.
    """
    if envelope.size < 16 or hop_seconds <= 0:
        return None, 0.0

    signal = envelope.astype(np.float64)
    if not np.isfinite(signal).all() or float(np.abs(signal).sum()) <= 0:
        return None, 0.0

    frame_rate = 1.0 / hop_seconds
    min_lag = max(1, int(np.floor(frame_rate * 60.0 / max_bpm)))
    max_lag = min(envelope.size - 1, int(np.ceil(frame_rate * 60.0 / min_bpm)))
    if max_lag <= min_lag + 1:
        return None, 0.0

    lags, curve = _global_acf(signal, min_lag, max_lag)
    if curve.size == 0:
        return None, 0.0

    bpms = 60.0 * frame_rate / lags
    weighted = np.where(np.isfinite(curve), curve * _tempo_prior(bpms), -np.inf)
    best = int(np.argmax(weighted))
    chosen = float(bpms[best])
    raw_score = float(curve[best])
    ambiguous = False

    # If the winner sits outside the practice range, prefer the octave that
    # lands inside it - at comparable periodicity the in-range reading is the
    # more plausible one for a practice track.
    if not (preferred_min <= chosen <= preferred_max):
        for factor in (2.0, 0.5):
            alternative = chosen * factor
            if not (preferred_min <= alternative <= preferred_max):
                continue
            index = int(np.argmin(np.abs(lags - frame_rate * 60.0 / alternative)))
            # Only accept a genuinely periodic alternative, and flag the result
            # as ambiguous so confidence does not overstate the evidence.
            if curve[index] >= 0.35 * raw_score:
                chosen = alternative
                raw_score = 0.7 * max(raw_score, float(curve[index]))
                ambiguous = True
                break

    # Parabolic refinement of the winning lag.
    index = int(np.argmin(np.abs(lags - frame_rate * 60.0 / chosen)))
    if 0 < index < curve.size - 1:
        left, centre, right = curve[index - 1], curve[index], curve[index + 1]
        denom = left - 2.0 * centre + right
        if abs(denom) > 1e-12:
            delta = max(-1.0, min(1.0, 0.5 * (left - right) / denom))
            refined = lags[index] + delta
            if refined > 0:
                chosen = 60.0 * frame_rate / refined

    # Confidence: how far the winning peak stands above the typical level of the
    # curve.  A strongly periodic track scores high, an ambiguous one low.
    finite = curve[np.isfinite(curve)]
    mean = float(finite.mean()) if finite.size else 0.0
    std = float(finite.std()) if finite.size else 0.0
    z = (raw_score - mean) / std if std > 1e-12 else 0.0
    confidence = float(1.0 / (1.0 + np.exp(-0.8 * (z - 1.5))))
    if ambiguous:
        confidence *= 0.55

    if not np.isfinite(chosen) or chosen <= 0:
        return None, 0.0
    return float(chosen), float(np.clip(confidence, 0.0, 1.0))


def _track_beats(
    envelope: np.ndarray,
    hop_seconds: float,
    bpm: float,
) -> list[float]:
    """Dynamic-programming beat tracker (Ellis 2007 style).

    Scores beat sequences by onset strength minus a penalty for deviating from
    the estimated period, which is far more robust than peak picking.
    """
    if envelope.size < 4 or bpm <= 0 or hop_seconds <= 0:
        return []

    period = 60.0 / bpm / hop_seconds  # in frames
    if period < 1:
        return []

    tightness = 100.0
    # Backtracking window: +/- 1 period, clamped to something workable.
    window = int(round(period))
    window = max(1, min(window, 32))

    size = envelope.size
    cumulative = np.array(envelope, dtype=np.float64)
    backtrack = np.zeros(size, dtype=np.int64)

    for frame in range(size):
        low = max(0, frame - window)
        high = frame - int(round(period / 2.0))
        if high < low:
            continue
        candidates = np.arange(low, high + 1)
        if candidates.size == 0:
            continue
        # Squared log deviation from the ideal period, scaled by tightness.
        # candidates starts at 0 for the first frames, and log(0) is -inf, so
        # clamp to 1 frame - by far the largest penalty, which is what we want.
        safe = np.maximum(candidates, 1)
        penalty = tightness * (np.log(safe / period) ** 2)
        scores = cumulative[candidates] - penalty
        best_index = int(np.argmax(scores))
        cumulative[frame] += scores[best_index]
        backtrack[frame] = candidates[best_index]

    # Start from the strongest cumulative score in the last few periods.
    tail_start = max(0, size - int(round(2 * period)))
    tail = cumulative[tail_start:]
    if tail.size == 0:
        return []
    position = tail_start + int(np.argmax(tail))

    beats: list[int] = []
    while position > 0 and len(beats) < size:
        beats.append(position)
        previous = int(backtrack[position])
        if previous >= position:  # guard against a degenerate chain
            break
        position = previous

    beats.reverse()
    return [round(int(frame) * hop_seconds, 4) for frame in beats]


def detect_bpm(
    audio: np.ndarray,
    rate: int,
    *,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
) -> BpmResult:
    """Full BPM detection on a ``(channels, frames)`` float array.

    Never raises and never blocks the caller: an unreliable result is reported
    with ``ok=False`` plus a reason, so the UI can ask the user for the tempo
    instead of failing the whole job.
    """
    try:
        mono = _to_mono(audio)
        if mono.size == 0:
            return BpmResult(ok=False, reason="音频为空")

        duration = mono.size / rate
        if duration < 5.0:
            return BpmResult(
                ok=False,
                reason=f"音频过短（{duration:.1f} 秒），无法可靠检测 BPM",
            )

        # Analyse at most the first ANALYSIS_SECONDS.
        limit = int(min(mono.size, ANALYSIS_SECONDS * rate))
        envelope, hop_seconds = onset_envelope(mono[:limit], rate)
        if envelope.size == 0:
            return BpmResult(ok=False, reason="无法提取节拍特征（音频可能过于安静）")

        bpm, confidence = tempo_from_envelope(
            envelope, hop_seconds, min_bpm=min_bpm, max_bpm=max_bpm
        )
        if bpm is None:
            return BpmResult(ok=False, reason="未能找到稳定的节拍周期")

        beats = _track_beats(envelope, hop_seconds, bpm)

        # The half/double equivalents are musically plausible readings of the
        # same signal, so we surface them for the UI to offer as one-click fixes.
        alternatives = [
            round(bpm * factor, 2)
            for factor in (2.0, 0.5)
            if min_bpm <= bpm * factor <= max_bpm
        ]

        if confidence < 0.20:
            return BpmResult(
                ok=False,
                bpm=bpm,
                confidence=confidence,
                beat_times=beats,
                alternatives=alternatives,
                reason=(
                    f"检测置信度较低（{confidence:.2f}），请手动确认 BPM"
                    f"（初步估计 {bpm:.1f}）"
                ),
            )

        return BpmResult(
            ok=True,
            bpm=bpm,
            confidence=confidence,
            beat_times=beats,
            alternatives=alternatives,
        )
    except Exception as exc:  # never let detection break the import flow
        return BpmResult(ok=False, reason=f"BPM 检测失败：{type(exc).__name__}: {exc}")


def detect(path_or_audio, rate: int | None = None, **kwargs) -> BpmResult:
    """Detect BPM from either a file path or an in-memory array.

    File loading goes through :mod:`app.audio.io`, which handles resampling.
    """
    if isinstance(path_or_audio, (str, bytes)) or hasattr(path_or_audio, "suffix"):
        from ..audio import io as audio_io

        audio, actual_rate = audio_io.read_wav(path_or_audio)
        return detect_bpm(audio, actual_rate, **kwargs)

    if rate is None:
        return BpmResult(ok=False, reason="缺少采样率参数")
    return detect_bpm(np.asarray(path_or_audio), rate, **kwargs)


def bpm_for_speed(original_bpm: float | None, speed: float) -> float | None:
    """Target BPM implied by a speed multiplier."""
    if not original_bpm or original_bpm <= 0:
        return None
    return original_bpm * float(speed)


def speed_for_bpm(original_bpm: float | None, target_bpm: float | None) -> float | None:
    """Speed multiplier implied by a target BPM."""
    if not original_bpm or original_bpm <= 0 or not target_bpm or target_bpm <= 0:
        return None
    return float(target_bpm) / float(original_bpm)
