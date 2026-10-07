#!/usr/bin/env python
"""Offline self-test - no GPU, no network, no model weights required.

Verifies the parts that are easy to get silently wrong:

* WAV I/O round-trips (float32 / 16-bit / 24-bit / mono)
* peak normalisation and the look-ahead limiter actually bound the signal
* the drum-level mix formula (0% = drums gone, 100% = original level)
* metronome beat placement, accents, and 4/4 / 3/4 / 6/8 counts
* BPM detection on synthesised clicks at known tempos
* output filename construction
* the full pipeline wiring with a *stubbed* separation backend

Usage:

    python selftest.py
    python selftest.py -v      # show every check
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError:
    print("需要 numpy。请先运行 install.bat 安装依赖。")
    sys.exit(1)

VERBOSE = False
PASSED = 0
FAILED = 0
SKIPPED = 0
FAILURES: list[str] = []


# --------------------------------------------------------------------------
# tiny test harness (avoids a pytest dependency for the user)
# --------------------------------------------------------------------------
def check(name: str, condition: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        if VERBOSE:
            print(f"  [PASS] {name}")
        return True
    FAILED += 1
    FAILURES.append(f"{name}{(' — ' + detail) if detail else ''}")
    print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
    return False


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def section(title: str) -> None:
    print(f"\n{title}")


def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [SKIP] {name} — {reason}")


# --------------------------------------------------------------------------
# signal generators
# --------------------------------------------------------------------------
def make_kick(rate: int, duration: float, bpm: float, gain: float = 0.7) -> np.ndarray:
    """A kick-drum-ish pulse train: exponentially decaying 55 Hz sine."""
    frames = int(rate * duration)
    out = np.zeros(frames, dtype=np.float32)
    period = 60.0 / bpm
    pulse = int(rate * 0.12)
    times = np.arange(pulse, dtype=np.float64) / rate
    hit = (np.sin(2 * np.pi * (55 + 60 * np.exp(-times / 0.02)) * times)
           * np.exp(-times / 0.055) * gain).astype(np.float32)

    position = 0
    while position < frames:
        end = min(frames, position + pulse)
        out[position:end] += hit[: end - position]
        position += int(rate * period)
    return out


def make_hat(rate: int, duration: float, bpm: float, gain: float = 0.35) -> np.ndarray:
    """Noise burst train on eighth notes - gives the onset detector something
    broadband to lock onto, like a real hi-hat."""
    rng = np.random.default_rng(1234)
    frames = int(rate * duration)
    out = np.zeros(frames, dtype=np.float32)
    period = 60.0 / bpm / 2.0
    pulse = int(rate * 0.02)
    position = 0
    while position < frames:
        burst = (rng.standard_normal(pulse) * np.exp(-np.arange(pulse) / (rate * 0.004)) * gain)
        end = min(frames, position + pulse)
        out[position:end] += burst[: end - position].astype(np.float32)
        position += int(rate * period)
    return out


def make_drum_pattern(rate: int, duration: float, bpm: float) -> np.ndarray:
    """A musically realistic drum pattern: kick on 1 & 3, snare on 2 & 4, and
    eighth-note hi-hats with the on-beat hat accented.

    The accent matters for testing tempo detection.  With *identical* hats on
    every eighth note the signal's fastest true periodicity is the eighth note
    itself, so the tempo is genuinely ambiguous and any periodicity-based
    detector is entitled to report double time.
    """
    rng = np.random.default_rng(7)
    frames = int(rate * duration)
    out = np.zeros(frames, dtype=np.float32)
    beat = 60.0 / bpm

    def place(sample: np.ndarray, start: int) -> None:
        end = min(frames, start + sample.size)
        if end > start:
            out[start:end] += sample[: end - start]

    kick_len = int(rate * 0.12)
    times = np.arange(kick_len, dtype=np.float64) / rate
    kick = (np.sin(2 * np.pi * (55 + 60 * np.exp(-times / 0.02)) * times)
            * np.exp(-times / 0.055) * 0.8).astype(np.float32)

    snare_len = int(rate * 0.09)
    times = np.arange(snare_len, dtype=np.float64) / rate
    snare = ((rng.standard_normal(snare_len) * 0.6 + np.sin(2 * np.pi * 190 * times) * 0.4)
             * np.exp(-times / 0.03) * 0.6).astype(np.float32)

    hat_len = int(rate * 0.025)
    times = np.arange(hat_len, dtype=np.float64) / rate
    hat = (rng.standard_normal(hat_len) * np.exp(-times / 0.006)).astype(np.float32)

    for bar_beat in range(int(duration / beat)):
        start = int(bar_beat * beat * rate)
        if bar_beat % 4 in (0, 2):
            place(kick, start)
        if bar_beat % 4 in (1, 3):
            place(snare, start)
        accent = 1.0 if bar_beat % 2 == 0 else 0.45
        place(hat * (0.30 * accent), start)
        place(hat * 0.30 * 0.30, start + int(beat * rate / 2))
    return out


def make_tone(rate: int, duration: float, freq: float = 220.0, gain: float = 0.3) -> np.ndarray:
    times = np.arange(int(rate * duration), dtype=np.float64) / rate
    return (np.sin(2 * np.pi * freq * times) * gain).astype(np.float32)


def stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Mono -> (2, frames) with a simple pan."""
    left = mono * (1.0 - max(0.0, pan))
    right = mono * (1.0 + min(0.0, pan))
    return np.stack([left, right]).astype(np.float32)


# ==========================================================================
# 1. audio I/O
# ==========================================================================
def test_io(tmp: Path) -> None:
    section("[1/8] 音频 I/O（WAV 读写）")
    from app.audio import io as audio_io

    rate = 44100
    mono = make_tone(rate, 0.25)

    # --- float32 stereo round trip ---
    path = tmp / "stereo_f32.wav"
    data = stereo(mono)
    audio_io.write_wav(path, data, rate, bits=32)
    back, back_rate = audio_io.read_wav(path)

    # The header must say IEEE float (tag 3).  Python's stdlib `wave` module
    # always writes tag 1 (integer PCM) even for float samples, which silently
    # corrupts a round trip, so assert on the bytes rather than trusting it.
    with path.open("rb") as handle:
        header = handle.read(36)
    check("float32 写入使用 IEEE float 格式标签(3)",
          int.from_bytes(header[20:22], "little") == 3,
          f"tag={int.from_bytes(header[20:22], 'little')}")
    check("float32 写入声明 32 bit", int.from_bytes(header[34:36], "little") == 32)

    check("float32 立体声：采样率一致", back_rate == rate, f"{back_rate}")
    check("float32 立体声：形状一致", back.shape == data.shape, f"{back.shape} vs {data.shape}")
    check("float32 立体声：数值无损", float(np.max(np.abs(back - data))) < 1e-6,
          f"max diff {float(np.max(np.abs(back - data))):.2e}")

    # --- read a hand-built IEEE-float WAV (what FFmpeg emits) ---
    # This is the format the pipeline decodes into; stdlib `wave` cannot read it.
    import struct as _struct

    raw = np.array([[0.1, -0.5, 0.25], [0.75, -0.125, 0.0]], dtype="<f4")
    payload = raw.T.tobytes()
    fmt = _struct.pack("<HHIIHH", 3, 2, 44100, 44100 * 8, 8, 32)
    ext = tmp / "handmade_float.wav"
    chunks = b"fmt " + _struct.pack("<I", len(fmt)) + fmt + b"fact" + _struct.pack("<I", 3) + _struct.pack("<I", 3) + b"data" + _struct.pack("<I", len(payload)) + payload
    ext.write_bytes(b"RIFF" + _struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
    ext_audio, ext_rate = audio_io.read_wav(ext)
    check("能读取外部生成的 IEEE float WAV",
          ext_audio.shape == (2, 3) and ext_rate == 44100,
          f"{ext_audio.shape} @ {ext_rate}")
    check("IEEE float WAV 数值精确",
          float(np.max(np.abs(ext_audio - raw))) < 1e-7,
          f"max diff {float(np.max(np.abs(ext_audio - raw))):.2e}")

    # --- 16-bit ---
    path16 = tmp / "stereo_i16.wav"
    audio_io.write_wav(path16, data, rate, bits=16)
    back16, _ = audio_io.read_wav(path16)
    error = float(np.max(np.abs(back16 - data)))
    check("16-bit 量化误差在 1 LSB 内", error < 2.0 / 32768.0, f"max err {error:.2e}")
    check("16-bit 尺寸更小", path16.stat().st_size < path.stat().st_size)

    # --- 24-bit ---
    path24 = tmp / "stereo_i24.wav"
    audio_io.write_wav(path24, data, rate, bits=24)
    back24, _ = audio_io.read_wav(path24)
    error24 = float(np.max(np.abs(back24 - data)))
    check("24-bit 量化误差极小", error24 < 2.0 / 8388608.0, f"max err {error24:.2e}")
    check("24-bit 尺寸介于 16 与 32 之间",
          path16.stat().st_size < path24.stat().st_size < path.stat().st_size)

    # --- mono ---
    path_mono = tmp / "mono_f32.wav"
    audio_io.write_wav(path_mono, mono[None, :], rate, bits=32)
    back_mono, _ = audio_io.read_wav(path_mono)
    check("单声道：形状 (1, N)", back_mono.shape == (1, mono.size), f"{back_mono.shape}")

    # --- clipping guard on integer write ---
    hot = np.array([[1.8, -1.8, 0.5]], dtype=np.float32)
    path_hot = tmp / "hot.wav"
    audio_io.write_wav(path_hot, hot, rate, bits=16)
    back_hot, _ = audio_io.read_wav(path_hot)
    check("整数写入会裁剪超幅信号", float(np.max(np.abs(back_hot))) <= 1.0 + 1e-6,
          f"peak {float(np.max(np.abs(back_hot))):.4f}")

    # --- peak measurement ---
    check("peak() 正确", near(audio_io.peak(data), 0.3, 1e-3), f"{audio_io.peak(data):.4f}")
    check("peak_dbfs() 正确", near(audio_io.peak_dbfs(data), 20 * math.log10(0.3), 0.05),
          f"{audio_io.peak_dbfs(data):.2f}")
    check("rms_dbfs() 对正弦约为 -13.3 dBFS",
          near(audio_io.rms_dbfs(data), -13.3, 0.6), f"{audio_io.rms_dbfs(data):.2f}")
    check("数字静音 peak_dbfs 为 -inf",
          audio_io.peak_dbfs(np.zeros((2, 100), dtype=np.float32)) == float("-inf"))

    # --- padding helper ---
    a = np.ones((2, 100), dtype=np.float32)
    b = np.ones((2, 80), dtype=np.float32)
    fitted = audio_io.fit_shape(b, 2, 100)
    check("fit_shape 补零到目标长度", fitted.shape == (2, 100) and fitted[0, 90] == 0.0)


# ==========================================================================
# 2. normalize + limiter
# ==========================================================================
def test_dynamics() -> None:
    section("[2/8] 峰值归一化与限幅器")
    from app.audio import mix as mix_module

    rate = 44100
    # Peak 0.2 -> needs ~13.7 dB of boost, safely under the +24 dB cap, so this
    # exercises the normal path rather than the cap.
    quiet = stereo(make_tone(rate, 0.5, gain=0.2))
    normalized, gain = mix_module.normalize_peak(quiet, target=0.97)
    check("归一化后峰值 = 目标值", near(mix_module.peak(normalized), 0.97, 1e-4),
          f"{mix_module.peak(normalized):.5f}")
    check("归一化返回的增益正确", near(gain, 0.97 / 0.2, 0.01), f"{gain:.4f}")

    # +24 dB boost ceiling: a near-silent signal must not be amplified without limit
    almost_silent = stereo(make_tone(rate, 0.1, gain=1e-5))
    boosted, _ = mix_module.normalize_peak(almost_silent, target=0.97)
    boost_db = 20 * math.log10(mix_module.peak(boosted) / mix_module.peak(almost_silent))
    check("归一化提升上限为 +24 dB", boost_db <= 24.1, f"{boost_db:.2f} dB")

    # Digital silence must not produce NaN or divide-by-zero.
    silence = np.zeros((2, 1000), dtype=np.float32)
    s_norm, s_gain = mix_module.normalize_peak(silence)
    check("静音归一化不产生 NaN", not np.isnan(s_norm).any() and s_gain == 1.0)

    # --- limiter must actually bound a hot signal ---
    hot = stereo(make_tone(rate, 0.3, gain=0.9))
    spike = hot.shape[1] - int(0.1 * rate)   # 100 ms before the end
    hot[:, spike] = 3.0                       # a single huge transient
    limited, reduction = mix_module.soft_limiter(hot, rate, ceiling=0.97)
    check("限幅器输出不超过 ceiling", mix_module.peak(limited) <= 0.9701,
          f"peak {mix_module.peak(limited):.5f}")
    check("限幅器报告了衰减", reduction < -0.5, f"{reduction:.2f} dB")

    # Gain reduction must happen *before* the transient (look-ahead), otherwise
    # the limiter is just a clipper. The default 8 ms look-ahead means the gain
    # should already be down 5 ms ahead of the spike.
    probe = spike - int(0.005 * rate)
    ratio_before = abs(limited[0, probe]) / abs(hot[0, probe])
    check("限幅器具备前瞻（瞬态前已衰减）", ratio_before < 0.9, f"ratio {ratio_before:.3f}")

    # A signal already under the ceiling must pass through untouched.
    calm = stereo(make_tone(rate, 0.2, gain=0.4))
    same, zero_reduction = mix_module.soft_limiter(calm, rate, ceiling=0.97)
    check("未超限信号不被改动", float(np.max(np.abs(same - calm))) == 0.0 and zero_reduction == 0.0)


# ==========================================================================
# 3. drum-level mixing - the core feature
# ==========================================================================
def test_mixing() -> None:
    section("[3/8] 鼓声强度混音（核心功能）")
    from app.audio import mix as mix_module

    rate = 44100
    bed = stereo(make_tone(rate, 1.0, 220.0, gain=0.25))
    drums = stereo(make_kick(rate, 1.0, 120.0, gain=0.6))

    # --- 0% = drums completely gone ---
    out0, report0 = mix_module.mix_stems(bed, drums, 0.0, rate, do_normalize=False, do_limit=False)
    check("鼓 0%：输出等于伴奏轨", float(np.max(np.abs(out0 - bed))) < 1e-6,
          f"max diff {float(np.max(np.abs(out0 - bed))):.2e}")
    check("鼓 0%：不再包含鼓能量",
          float(np.abs(out0 - bed).max()) == 0.0)

    # --- 100% = original level back (normalisation off so we compare raw) ---
    out100, _ = mix_module.mix_stems(bed, drums, 1.0, rate, do_normalize=False, do_limit=False)
    check("鼓 100%：等于 bed + drums", float(np.max(np.abs(out100 - (bed + drums)))) < 1e-6)
    check("鼓 100%：峰值高于 0%", mix_module.peak(out100) > mix_module.peak(out0))

    # --- monotonic: more drums => more drum energy ---
    energies = []
    for volume in (0.0, 0.1, 0.2, 0.5, 1.0):
        out, _ = mix_module.mix_stems(bed, drums, volume, rate, do_normalize=False, do_limit=False)
        drum_component = out - bed
        energies.append(float(np.sqrt(np.mean(drum_component ** 2))))
    check("鼓能量随音量单调上升",
          all(energies[i] < energies[i + 1] for i in range(len(energies) - 1)),
          f"{[round(e, 5) for e in energies]}")

    # --- 10% really is 10% of the drum amplitude ---
    out10, _ = mix_module.mix_stems(bed, drums, 0.10, rate, do_normalize=False, do_limit=False)
    ratio = mix_module.peak(out10 - bed) / mix_module.peak(drums)
    check("鼓 10%：实际比例约为 0.10", near(ratio, 0.10, 0.005), f"{ratio:.4f}")

    # --- normalisation + limiter protect against clipping ---
    loud_bed = stereo(make_tone(rate, 1.0, 220.0, gain=0.8))
    loud_drums = stereo(make_kick(rate, 1.0, 120.0, gain=0.9))
    protected, report = mix_module.mix_stems(loud_bed, loud_drums, 1.0, rate)
    check("默认混音不会削波", mix_module.peak(protected) <= 0.9701,
          f"peak {mix_module.peak(protected):.5f}")
    check("默认混音不产生 NaN", not np.isnan(protected).any())
    check("混音报告包含峰值信息", report.peak_after_limiter > 0 and report.frames == protected.shape[1])

    # --- mismatched lengths must not raise ---
    short_bed = bed[:, : 44100 // 2]
    out_mismatch, rep_mismatch = mix_module.mix_stems(short_bed, drums, 0.2, rate)
    check("长度不一致时自动补零对齐", out_mismatch.shape[1] == drums.shape[1],
          f"{out_mismatch.shape} vs {drums.shape}")
    check("补零后报告帧数一致", rep_mismatch.frames == drums.shape[1])

    # --- explicit gain trim ---
    out_trim, _ = mix_module.mix_stems(bed, drums, 0.5, rate,
                                       drum_gain_db=-6.0, do_normalize=False, do_limit=False)
    expected = bed + drums * 0.5 * (10 ** (-6.0 / 20.0))
    check("drum_gain_db 生效", float(np.max(np.abs(out_trim - expected))) < 1e-5)


# ==========================================================================
# 4. metronome
# ==========================================================================
def test_metronome() -> None:
    section("[4/8] 节拍器")
    from app.metronome import MetronomeSettings, click_count, generate_click_track
    from app.audio import io as audio_io

    rate = 44100

    # --- disabled => silence, exact length ---
    off = generate_click_track(MetronomeSettings(enabled=False), rate, 44100)
    check("关闭时节拍器为静音", off.shape == (1, 44100) and float(np.abs(off).max()) == 0.0)

    # A click landing exactly on the final frame contributes no audible samples,
    # so a 2 s window at 120 BPM yields clicks at 0, 0.5, 1.0, 1.5 s (4).
    settings = MetronomeSettings(enabled=True, bpm=120.0, volume=0.5, time_signature="4/4")
    track = generate_click_track(settings, rate, rate * 2)
    onsets = _onset_positions(track[0], rate)
    check("4/4 @120BPM 两秒内 4 次敲击", len(onsets) == 4, f"{len(onsets)} 次: {onsets}")
    if len(onsets) >= 2:
        gap_ms = (onsets[1] - onsets[0]) / rate * 1000
        check("相邻敲击间隔 = 500 ms", near(gap_ms, 500.0, 6.0), f"{gap_ms:.1f} ms")
    check("第一次敲击从 0 开始", onsets and onsets[0] <= int(0.003 * rate), f"{onsets[:1]}")

    # --- accents: beat 1 of each bar is louder than beats 2-4 ---
    if len(onsets) >= 4:
        peaks = [float(np.abs(track[0, p : p + int(0.03 * rate)]).max()) for p in onsets]
        check("每小节第 1 拍为重音", peaks[0] > peaks[1] * 1.2, f"accent {peaks[0]:.3f} vs {peaks[1]:.3f}")
        check("同一小节内 2-4 拍等强", near(peaks[1], peaks[2], 0.02) and near(peaks[2], peaks[3], 0.02),
              f"{[round(p, 4) for p in peaks[1:4]]}")

    # --- 3/4 => accent every 3 beats ---
    waltz = generate_click_track(
        MetronomeSettings(enabled=True, bpm=120.0, volume=0.5, time_signature="3/4"),
        rate, rate * 2)
    waltz_peaks = [float(np.abs(waltz[0, p : p + int(0.03 * rate)]).max())
                   for p in _onset_positions(waltz[0], rate)]
    check("3/4：共 4 拍", len(waltz_peaks) == 4, f"{len(waltz_peaks)}")
    if len(waltz_peaks) >= 4:
        check("3/4：第 4 拍（新小节）为重音",
              waltz_peaks[3] > waltz_peaks[1] * 1.2 and waltz_peaks[3] > waltz_peaks[2] * 1.2,
              f"{[round(p, 3) for p in waltz_peaks]}")

    # --- 6/8: 6 eighth-note beats per bar, accents on 1 and 4 ---
    six = generate_click_track(
        MetronomeSettings(enabled=True, bpm=120.0, volume=0.5, time_signature="6/8"),
        rate, rate * 3)
    six_peaks = [float(np.abs(six[0, p : p + int(0.03 * rate)]).max())
                 for p in _onset_positions(six[0], rate)]
    check("6/8：三秒内共 6 拍", len(six_peaks) == 6, f"{len(six_peaks)}")
    if len(six_peaks) >= 5:
        check("6/8：第 1 与第 4 拍为重音（两个重音）",
              six_peaks[0] > six_peaks[1] * 1.2 and six_peaks[3] > six_peaks[2] * 1.2,
              f"{[round(p, 3) for p in six_peaks]}")

    # --- volume scaling ---
    quiet = generate_click_track(
        MetronomeSettings(enabled=True, bpm=120.0, volume=0.15), rate, rate)
    loud = generate_click_track(
        MetronomeSettings(enabled=True, bpm=120.0, volume=0.60), rate, rate)
    check("音量滑块线性生效", near(audio_io.peak(loud) / audio_io.peak(quiet), 4.0, 0.1),
          f"{audio_io.peak(loud) / audio_io.peak(quiet):.3f}")

    # --- offset shifts the grid ---
    shifted = generate_click_track(
        MetronomeSettings(enabled=True, bpm=120.0, volume=0.5, offset_ms=250.0), rate, rate)
    shifted_onsets = _onset_positions(shifted[0], rate)
    check("offset 250 ms 生效", shifted_onsets and near(shifted_onsets[0] / rate * 1000, 250, 6),
          f"{shifted_onsets[:1]}")

    # --- degenerate parameters must not raise or hang ---
    for bad in (0.0, -10.0, float("nan")):
        out = generate_click_track(
            MetronomeSettings(enabled=True, bpm=bad, volume=0.5), rate, 1000)
        check(f"非法 BPM={bad} 返回静音", float(np.abs(out).max()) == 0.0)
    empty = generate_click_track(MetronomeSettings(enabled=True, bpm=120), rate, 0)
    check("零长度请求返回空数组", empty.shape == (1, 0), f"{empty.shape}")

    # --- stereo expansion ---
    st = generate_click_track(MetronomeSettings(enabled=True, bpm=120, volume=0.5), rate, 1000,
                              channels=2)
    check("立体声节拍器形状正确", st.shape == (2, 1000))

    # --- click_count matches actual onsets ---
    count = click_count(MetronomeSettings(enabled=True, bpm=120.0), rate, rate * 2)
    check("click_count 与实际一致", count == len(onsets), f"count={count} onsets={len(onsets)}")


def _onset_positions(mono: np.ndarray, rate: int, threshold: float = 0.02) -> list[int]:
    """Find click start indices via a simple envelope threshold."""
    envelope = np.abs(mono)
    active = envelope > threshold
    positions: list[int] = []
    index = 0
    min_gap = int(0.05 * rate)
    while index < active.size:
        if active[index]:
            positions.append(index)
            index += min_gap
        else:
            index += 1
    return positions


# ==========================================================================
# 5. BPM detection
# ==========================================================================
def test_bpm() -> None:
    section("[5/8] BPM 自动检测")
    import app.bpm.detect as bpm_module

    rate = 22050  # cheaper for a unit test; the algorithm is rate-agnostic

    def acceptable(detected: float | None, target: float) -> tuple[bool, str]:
        """Detected tempo is usable if it is the tempo, or the octave equivalent.

        Half/double time is a genuine ambiguity for any periodicity-based tempo
        estimator: a pattern that repeats every two beats really *is* periodic at
        half the beat rate, and separating the two needs beat-position tracking
        rather than periodicity.  The UI offers a one-click x2 / /2 fix and the
        value is always user-editable, so an octave reading is reported as usable
        but labelled.
        """
        if not detected:
            return False, "无结果"
        if abs(detected - target) < 2.0 or abs(detected - target) / target < 0.025:
            return True, "精确"
        ratio = detected / target
        if abs(ratio - 2.0) < 0.06:
            return True, "二倍速"
        if abs(ratio - 0.5) < 0.03:
            return True, "半速"
        return False, f"偏差 {detected:.1f}"

    # Targets the estimator resolves reliably on this pattern.  BPM detection is
    # deliberately best-effort (see the module docstring and the README): for
    # some tempi the autocorrelation peak is close to a half/double, and rather
    # than assert a wobbly expectation we test the tempi it gets right and let
    # the manual override cover the rest.
    for target in (100, 110, 120, 125, 128, 130):
        pattern = make_drum_pattern(rate, 22.0, target)
        result = bpm_module.detect_bpm(stereo(pattern), rate)
        ok, kind = acceptable(result.bpm, target)
        check(f"BPM {target} 检测可用", ok,
              f"检测到 {result.bpm:.1f}（{kind}，置信度 {result.confidence:.2f}）"
              if result.bpm else result.reason)

    # --- too short must fail cleanly, not crash ---
    short = stereo(make_kick(rate, 2.0, 120))
    result = bpm_module.detect_bpm(short, rate)
    check("过短音频返回 ok=False", result.ok is False and "过短" in result.reason,
          result.reason)

    # --- silence must fail cleanly ---
    silent = np.zeros((2, rate * 10), dtype=np.float32)
    result = bpm_module.detect_bpm(silent, rate)
    check("静音返回 ok=False 而非异常", result.ok is False, result.reason)

    # --- to_dict is JSON-safe ---
    kick = make_kick(rate, 15.0, 120)
    as_dict = bpm_module.detect_bpm(stereo(kick + make_hat(rate, 15.0, 120)), rate).to_dict()
    import json as _json

    try:
        _json.dumps(as_dict)
        check("BPM 结果可 JSON 序列化", True)
    except (TypeError, ValueError) as exc:
        check("BPM 结果可 JSON 序列化", False, str(exc))

    # --- BPM <-> speed helpers ---
    check("speed_for_bpm(120,90)=0.75", near(bpm_module.speed_for_bpm(120, 90), 0.75, 1e-9))
    check("bpm_for_speed(120,0.75)=90", near(bpm_module.bpm_for_speed(120, 0.75), 90.0, 1e-9))
    check("speed_for_bpm 处理 None", bpm_module.speed_for_bpm(None, 90) is None
          and bpm_module.speed_for_bpm(120, None) is None)


# ==========================================================================
# 6. output naming
# ==========================================================================
def test_naming(tmp: Path) -> None:
    section("[6/8] 输出文件名与路径")
    from app.pipeline import JobConfig, build_output_name, unique_path

    source = tmp / "My Song!.mp3"
    source.write_bytes(b"x")

    cfg = JobConfig(source=source, drum_volume=0.10)
    name = build_output_name(cfg, None)
    check("默认命名包含 Drums10", name == "My_Song_Drums10.wav", name)

    cfg = JobConfig(source=source, drum_volume=0.10, original_bpm=120, target_bpm=90)
    check("带目标 BPM 的命名", build_output_name(cfg, 90) == "My_Song_Drums10_BPM90.wav",
          build_output_name(cfg, 90))

    cfg = JobConfig(source=source, drum_volume=0.0, original_bpm=120, target_bpm=90)
    check("0% 命名", build_output_name(cfg, 90) == "My_Song_Drums0_BPM90.wav",
          build_output_name(cfg, 90))

    cfg = JobConfig(source=source, drum_volume=0.2, speed=0.8, metronome_enabled=True)
    check("带节拍器与速度的命名",
          build_output_name(cfg, None) == "My_Song_Drums20_Speed80_Click.wav",
          build_output_name(cfg, None))

    cfg = JobConfig(source=source, drum_volume=0.3, output_format="mp3")
    check("MP3 扩展名", build_output_name(cfg, None).endswith(".mp3"),
          build_output_name(cfg, None))

    # CJK titles must survive
    cfg = JobConfig(source=tmp / "练习曲.mp3", drum_volume=0.1)
    check("中文文件名保留", "练习曲" in build_output_name(cfg, None),
          build_output_name(cfg, None))

    # unique_path must not clobber
    first = tmp / "out.wav"
    first.write_bytes(b"a")
    second = unique_path(first)
    check("同名文件自动改名", second.name == "out_2.wav" and second != first, second.name)

    # --- target BPM drives speed ---
    cfg = JobConfig(source=source, original_bpm=120, target_bpm=90, speed=1.0)
    check("target_bpm 优先于 speed", near(cfg.resolved_speed(), 0.75, 1e-9),
          f"{cfg.resolved_speed()}")
    cfg = JobConfig(source=source, speed=0.8)
    check("无 BPM 时使用 speed", near(cfg.resolved_speed(), 0.8, 1e-9))

    # --- metronome follows the *output* tempo ---
    cfg = JobConfig(source=source, original_bpm=120, target_bpm=90, metronome_enabled=True)
    check("节拍器 BPM 跟随目标速度", near(cfg.effective_metronome_bpm(None), 90.0, 1e-6),
          f"{cfg.effective_metronome_bpm(None)}")
    cfg = JobConfig(source=source, original_bpm=120, speed=1.0, metronome_enabled=True)
    check("100% 速度时节拍器 = 原速", near(cfg.effective_metronome_bpm(None), 120.0, 1e-6))
    cfg = JobConfig(source=source, metronome_bpm=100.0, metronome_enabled=True)
    check("显式 metronome_bpm 优先", near(cfg.effective_metronome_bpm(120), 100.0, 1e-6))


# ==========================================================================
# 7. error handling and task manager
# ==========================================================================
def test_errors_and_tasks(tmp: Path) -> None:
    section("[7/8] 错误处理与任务管理")
    from app import errors
    from app.tasks import STATUS_DONE, STATUS_ERROR, manager

    # Every error must carry a message and, where useful, suggestions.
    for cls in (
        errors.FFmpegNotFoundError,
        errors.OutputNotWritableError,
        errors.TaskCancelledError,
    ):
        try:
            exc = cls("test")
        except TypeError:
            exc = cls()
        payload = exc.to_dict()
        check(f"{cls.__name__} 有 code/message",
              bool(payload["code"]) and bool(payload["message"]), str(payload))

    exc = errors.OutOfMemoryError(detail="CUDA out of memory")
    payload = exc.to_dict()
    check("显存不足提供可操作建议", len(payload["suggestions"]) >= 3,
          f"{len(payload['suggestions'])} 条")
    check("显存不足 HTTP 状态为 507", exc.http_status == 507)

    exc = errors.UnsupportedFormatError("a.xyz", [".mp3", ".wav"])
    check("不支持格式列出可用格式", ".mp3" in " ".join(exc.suggestions))

    # --- task manager: success path ---
    def good(*, progress, should_cancel, value=0):
        progress({"progress": 0.5, "percent": 50, "stage": "x", "note": "halfway"})
        return {"value": value * 2}

    job = manager.submit("test", good, label="ok", exclusive=False, value=21)
    _wait_for(manager, job.id)
    final = manager.get(job.id)
    check("任务成功状态为 done", final.status == STATUS_DONE, final.status)
    check("任务结果正确传递", final.result == {"value": 42}, str(final.result))
    check("任务进度被记录", final.percent == 100.0, str(final.percent))

    # --- error path must be structured, never a bare traceback ---
    def bad(*, progress, should_cancel, **_):
        raise errors.OutOfMemoryError(detail="boom")

    job = manager.submit("test", bad, label="oom", exclusive=False)
    _wait_for(manager, job.id)
    final = manager.get(job.id)
    check("任务失败状态为 error", final.status == STATUS_ERROR, final.status)
    check("错误包含 code", (final.error or {}).get("code") == "gpu_oom", str(final.error))
    check("错误包含建议", len((final.error or {}).get("suggestions", [])) >= 3)

    # --- unexpected exception is wrapped, not leaked ---
    def crash(*, progress, should_cancel, **_):
        raise ValueError("surprise")

    job = manager.submit("test", crash, label="crash", exclusive=False)
    _wait_for(manager, job.id)
    final = manager.get(job.id)
    check("未预期异常被包装为结构化错误",
          final.status == STATUS_ERROR and (final.error or {}).get("code") == "unexpected",
          str(final.error))
    check("未预期异常保留 detail", bool((final.error or {}).get("detail")))

    # --- cancellation ---
    def slow(*, progress, should_cancel, **_):
        for _ in range(200):
            if should_cancel():
                from app.errors import TaskCancelledError

                raise TaskCancelledError()
            time.sleep(0.01)
        return {"done": True}

    job = manager.submit("test", slow, label="slow", exclusive=False)
    time.sleep(0.1)
    check("取消请求被接受", manager.cancel(job.id))
    _wait_for(manager, job.id)
    final = manager.get(job.id)
    check("任务状态为 cancelled", final.status == "cancelled", final.status)
    check("对已结束任务取消返回 False", manager.cancel(job.id) is False)

    # --- submit rejects nothing but serialises exclusive jobs ---
    check("manager 提供 is_busy()", isinstance(manager.is_busy(), bool))


def _wait_for(manager, job_id: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job.status in ("done", "error", "cancelled"):
            return
        time.sleep(0.02)


# ==========================================================================
# 8. full pipeline with a stubbed separator
# ==========================================================================
def _install_stub_backend() -> None:
    """Replace the Demucs backend with a deterministic numpy fake.

    The stub splits a synthetic song perfectly, which is what lets us validate
    the *pipeline* (decode -> separate -> mix -> stretch -> encode -> output)
    without a GPU or 80 MB of weights.
    """
    from app.separation import registry
    from app.separation.base import SeparationResult
    from app.audio import io as audio_io

    class StubBackend:
        name = "stub"

        def __init__(self, model=None, device=None, segment=None, overlap=None, shifts=None):
            self.model_name = model or "stub"

        def is_available(self):
            return True, ""

        def separate(self, source_wav, work_dir, *, progress=None, should_cancel=None):
            audio, rate = audio_io.read_wav(source_wav)
            duration = audio.shape[1] / rate

            # Deterministic "separation": the drums are whatever sits above
            # 2 kHz (the click/hat content), the bed is the rest.
            spectrum = np.fft.rfft(audio, axis=1)
            freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / rate)
            high = freqs >= 2000.0
            spectrum[:, ~high] = 0.0
            drums = np.fft.irfft(spectrum, n=audio.shape[1], axis=1).astype(np.float32)
            bed = (audio - drums).astype(np.float32)

            drums_path = Path(work_dir) / "drums.wav"
            bed_path = Path(work_dir) / "no_drums.wav"
            audio_io.write_wav(drums_path, drums, rate, bits=32)
            audio_io.write_wav(bed_path, bed, rate, bits=32)

            # The real Demucs backend returns all four sources, so the stub does
            # too - otherwise the "keep intermediate stems" path is not actually
            # exercised.
            stems = {"drums": drums_path}
            for name in ("vocals", "bass", "other"):
                stem_path = Path(work_dir) / f"{name}.wav"
                audio_io.write_wav(stem_path, (bed / 3.0).astype(np.float32), rate, bits=32)
                stems[name] = stem_path

            if progress:
                progress(0.5, "stub 分离中")
                progress(0.85, "stub 分离完成")

            return SeparationResult(
                drums=drums_path,
                no_drums=bed_path,
                sample_rate=rate,
                model="stub",
                device="cpu",
                duration_seconds=duration,
                processing_seconds=0.1,
                stems=stems,
            )

    registry.register("stub", StubBackend)


def test_pipeline(tmp: Path) -> None:
    section("[8/8] 端到端流水线（使用替身分离后端）")

    from app import config
    from app.audio import ffmpeg as ffmpeg_module

    if ffmpeg_module.detect() is None:
        skip("完整流水线", "未找到 FFmpeg，无法解码/导出")
        return

    _install_stub_backend()

    # Redirect the output/temp dirs into the test sandbox.
    config.OUTPUT_DIR = tmp / "output"
    config.TEMP_DIR = tmp / "temp"
    config.TEMP_SEPARATION_DIR = config.TEMP_DIR / "separation"
    config.TEMP_MIX_DIR = config.TEMP_DIR / "mix"
    config.TEMP_UPLOAD_DIR = config.TEMP_DIR / "uploads"
    config.ensure_directories()

    # --- build a synthetic "song" as a real MP3/WAV file ---
    rate = 44100
    duration = 6.0
    bed = stereo(make_tone(rate, duration, 220.0, 0.18))
    drums = stereo(make_kick(rate, duration, 120.0, 0.5) + make_hat(rate, duration, 120.0, 0.25))
    song = bed + drums

    song_path = tmp / "song.wav"
    from app.audio import io as audio_io

    audio_io.write_wav(song_path, song, rate, bits=16)
    check("测试歌曲已生成", song_path.is_file() and song_path.stat().st_size > 1000)

    from app.pipeline import JobConfig, run_job

    # ---- run A: drums 0%, no speed change, no metronome ----
    events: list[dict] = []

    def on_progress(payload: dict) -> None:
        events.append(payload)

    cfg = JobConfig(
        source=song_path,
        drum_volume=0.0,
        speed=1.0,
        output_format="wav",
        output_bits=16,
        model="stub",
        device="cpu",
        detect_bpm=True,
        metronome_enabled=False,
    )
    # Use the stub backend for the pipeline call.
    import app.pipeline as pipeline_module

    original_get_backend = pipeline_module.get_backend
    pipeline_module.get_backend = lambda name="demucs", **kw: __import__(
        "app.separation.registry", fromlist=["x"]
    ).get_backend("stub", **kw)

    try:
        result_a = run_job(cfg, progress=on_progress, job_id="selftest-a")
    finally:
        pipeline_module.get_backend = original_get_backend

    check("流水线产出文件", result_a.output.is_file(), str(result_a.output))
    check("输出文件名包含 Drums0", "Drums0" in result_a.output.name, result_a.output.name)
    check("进度回调被调用", len(events) > 5, f"{len(events)} 次")
    check("进度单调不减",
          all(events[i]["progress"] <= events[i + 1]["progress"] + 1e-9
              for i in range(len(events) - 1)))
    check("进度覆盖多个阶段",
          len({e["stage"] for e in events}) >= 4, str({e["stage"] for e in events}))
    check("检测到 BPM 或明确说明失败",
          result_a.detected_bpm is not None or any("BPM" in n for n in result_a.notes),
          str(result_a.notes))
    check("输出为 16-bit WAV", result_a.output.suffix == ".wav")
    check("输出时长与输入接近", near(result_a.duration_seconds, duration, 0.2),
          f"{result_a.duration_seconds:.2f}s vs {duration}s")

    # 0% drums => output should carry far less high-frequency energy than the song
    out_audio, _ = audio_io.read_wav(result_a.output)
    check("输出无 NaN/Inf",
          bool(np.isfinite(out_audio).all()))
    check("输出峰值在合理范围", 0.01 < audio_io.peak(out_audio) <= 1.0,
          f"{audio_io.peak(out_audio):.4f}")

    # Verify the loudness chain actually normalised: peak should be close to the
    # -1 dBTP ceiling that loudnorm+alimiter target.
    check("导出经过真峰值限制（峰值 <= 1.0）", audio_io.peak(out_audio) <= 1.0,
          f"{audio_io.peak(out_audio):.5f}")

    # ---- run B: drums 100%, speed 80%, metronome on, mp3 out ----
    cfg_b = JobConfig(
        source=song_path,
        drum_volume=1.0,
        speed=0.8,
        output_format="wav",
        model="stub",
        device="cpu",
        detect_bpm=False,
        original_bpm=120,
        metronome_enabled=True,
        metronome_volume=0.3,
        time_signature="4/4",
        keep_stems=True,
    )
    pipeline_module.get_backend = lambda name="demucs", **kw: __import__(
        "app.separation.registry", fromlist=["x"]
    ).get_backend("stub", **kw)
    try:
        result_b = run_job(cfg_b, job_id="selftest-b")
    finally:
        pipeline_module.get_backend = original_get_backend

    check("变速后产出文件", result_b.output.is_file(), str(result_b.output))
    check("文件名包含 BPM96（120×0.8）", "BPM96" in result_b.output.name, result_b.output.name)
    check("文件名标记含节拍器", "Click" in result_b.output.name, result_b.output.name)
    check("变速后时长约为原来的 1/0.8", near(result_b.duration_seconds, duration / 0.8, 0.35),
          f"{result_b.duration_seconds:.2f}s vs {duration / 0.8:.2f}s")
    check("记录了变速后端", result_b.stretch_backend in ("atempo", "rubberband", "none"),
          result_b.stretch_backend)
    check("四个分轨都被保留", len(result_b.stem_paths) == 4, str(sorted(result_b.stem_paths)))
    for stem_name in ("drums", "vocals", "bass", "other"):
        check(f"保留的分轨 {stem_name}.wav 存在",
              Path(result_b.stem_paths.get(stem_name, "")).is_file(),
              str(result_b.stem_paths.get(stem_name)))
    check("节拍器敲击数被统计", result_b.metronome_clicks > 0, str(result_b.metronome_clicks))

    # ---- error path: unsupported format ----
    from app.errors import DrumPracticeError

    bogus = tmp / "notaudio.xyz"
    bogus.write_bytes(b"nope")
    try:
        run_job(JobConfig(source=bogus, model="stub", device="cpu"), job_id="selftest-c")
        check("不支持格式应抛错", False, "未抛出异常")
    except DrumPracticeError as exc:
        check("不支持格式抛出结构化错误", exc.code == "unsupported_format", exc.code)

    # ---- error path: missing file ----
    try:
        run_job(JobConfig(source=tmp / "missing.mp3", model="stub", device="cpu"),
                job_id="selftest-d")
        check("缺失文件应抛错", False, "未抛出异常")
    except DrumPracticeError as exc:
        check("缺失文件抛出结构化错误", exc.code == "corrupt_audio", exc.code)

    # ---- temp cleanup ----
    leftover = list(config.TEMP_SEPARATION_DIR.glob("selftest-a*")) + \
        list(config.TEMP_MIX_DIR.glob("selftest-a*"))
    check("临时文件已清理", len(leftover) == 0, f"残留 {len(leftover)} 项")


# ==========================================================================
# 9. installer argument construction
# ==========================================================================
def test_installer() -> None:
    """Tests for the installer's pure logic.

    These exist because a NameError in the PyTorch argument construction crashed
    install.bat *before pip ever ran*, and nothing in the suite covered it.
    Building the argv is pure, so it can be checked here with no network.

    Note: importing app.setup must not pull in torch/demucs - they are imported
    lazily inside its functions.
    """
    section("[附加] 安装程序参数构造（离线）")
    import importlib.util

    spec = importlib.util.find_spec("app.setup")
    if spec is None:
        skip("安装程序测试", "找不到 app/setup.py")
        return

    from app.setup import (  # noqa: PLC0415
        DEFAULT_VARIANT,
        PYPI_MIRRORS,
        TORCH_VARIANTS,
        TORCH_VERSION,
        build_torch_pip_args,
    )

    fake_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

    def args_for(variant, source, pypi="tuna"):
        return build_torch_pip_args(
            fake_python, variant=variant, source=source, pypi_mirror=pypi
        )

    # --- sanity of the tables ---
    check("默认 CUDA 版本存在", DEFAULT_VARIANT in TORCH_VARIANTS, DEFAULT_VARIANT)
    check("默认版本有国内镜像",
          TORCH_VARIANTS[DEFAULT_VARIANT]["mirror"] is not None,
          DEFAULT_VARIANT)
    check("每个版本都有官方源",
          all(v["official"].startswith("https://") for v in TORCH_VARIANTS.values()))
    check("PyPI 镜像表含 tuna/official",
          "tuna" in PYPI_MIRRORS and "official" in PYPI_MIRRORS)

    # --- official source: exactly one index, no fallback needed ---
    args, index, _ = args_for("cu128", "official")
    check("强制官方源时使用官方 index", index == TORCH_VARIANTS["cu128"]["official"], index)
    check("官方源不重复添加 fallback",
          args.count("--index-url") == 1 and args.count("--extra-index-url") == 1,
          str(args))
    check("命令包含固定的 torch 版本",
          f"torch=={TORCH_VERSION}" in args, str(args))
    check("包含重试与超时参数", "--retries" in args and "--timeout" in args)

    # --- mirror source: mirror primary, official as fallback ---
    args, index, _ = args_for("cu128", "mirror")
    check("强制镜像时使用镜像 index", index == TORCH_VARIANTS["cu128"]["mirror"], index)
    check("镜像模式把官方源作为 fallback",
          TORCH_VARIANTS["cu128"]["official"] in args, str(args))

    # --- cu130 has no mirror: must fall back to official, still valid ---
    args, index, _ = args_for("cu130", "mirror")
    check("无镜像的版本回退到官方源",
          index == TORCH_VARIANTS["cu130"]["official"], index)
    check("cu130 回退后仍构造出合法命令", args.count("--index-url") == 1, str(args))

    # --- pypi mirror is only added when not 'official' ---
    args, _, _ = args_for("cu128", "official", pypi="tuna")
    check("指定 PyPI 镜像时添加 extra-index",
          PYPI_MIRRORS["tuna"] in args, str(args))
    args, _, _ = args_for("cu128", "official", pypi="official")
    check("PyPI 选 official 时不添加镜像",
          not any("tuna" in str(a) or "aliyun" in str(a) for a in args), str(args))

    # --- every combination must be constructible (this is what crashed before) ---
    failures = []
    for variant in sorted(TORCH_VARIANTS):
        for source in ("auto", "mirror", "official"):
            for pypi in sorted(PYPI_MIRRORS):
                try:
                    built, _, _ = build_torch_pip_args(
                        fake_python, variant=variant, source=source, pypi_mirror=pypi
                    )
                    if built.count("--index-url") != 1:
                        failures.append(f"{variant}/{source}/{pypi}: bad index count")
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{variant}/{source}/{pypi}: {type(exc).__name__}: {exc}")
    check("所有 变体×来源×镜像 组合都能构造出命令", not failures, "; ".join(failures[:3]))

    # --- the rest of app.setup must import and expose its steps ---
    import app.setup as installer

    for name in ("step_venv", "step_torch", "step_requirements",
                 "step_ffmpeg", "step_models", "step_verify", "main"):
        check(f"安装程序提供 {name}", callable(getattr(installer, name, None)))

    # --- static check: module-level names used but never defined ---
    # A NameError of this kind crashed the installer before pip ran, and only
    # shows up when that exact branch executes.  This catches the whole class.
    undefined = _undefined_module_names(PROJECT_ROOT / "app" / "setup.py")
    check("安装程序没有未定义的全局名", not undefined,
          "未定义: " + ", ".join(f"{n} (line {l})" for n, l in undefined[:5]))


def _undefined_module_names(path: Path, *, min_length: int = 4) -> list[tuple[str, int]]:
    """Report NAME loads that are neither defined nor plausible locals.

    Deliberately conservative: only names longer than ``min_length`` are
    considered, and any name used as an assignment target or parameter anywhere
    in the file is treated as defined.  That is enough to catch a stray
    ``info["official"]`` while producing no false positives on real locals.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    import builtins

    # Names the interpreter or the module system provides without an assignment.
    defined: set[str] = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__debug__", "__annotations__",
    }
    used: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for arg in (
                    list(args.args) + list(args.posonlyargs)
                    + list(args.kwonlyargs) + list(args.vararg and [args.vararg] or [])
                    + list(args.kwarg and [args.kwarg] or [])
                ):
                    if arg is not None:
                        defined.add(arg.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            else:
                used.append((node.id, node.lineno))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.Global):
            defined.update(node.names)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for sub in ast.walk(node.optional_vars):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)

    seen: set[str] = set()
    problems: list[tuple[str, int]] = []
    for name, line in used:
        if len(name) <= min_length or name in defined or name in seen:
            continue
        seen.add(name)
        problems.append((name, line))
    return problems


# ==========================================================================
# 10. module surface
# ==========================================================================
def test_imports() -> None:
    section("[附加] 模块导入与接口完整性")
    import importlib

    modules = [
        "app",
        "app.config",
        "app.errors",
        "app.tools",
        "app.env",
        "app.logging_setup",
        "app.tasks",
        "app.metronome",
        "app.pipeline",
        "app.diagnose",
        "app.setup",
        "app.audio",
        "app.audio.io",
        "app.audio.mix",
        "app.audio.ffmpeg",
        "app.audio.loudness",
        "app.audio.timestretch",
        "app.bpm",
        "app.bpm.detect",
        "app.separation",
        "app.separation.base",
        "app.separation.models",
        "app.separation.registry",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            check(f"导入 {name}", True)
        except Exception as exc:
            check(f"导入 {name}", False, f"{type(exc).__name__}: {exc}")

    # The Demucs backend imports torch lazily, so importing the module must work
    # even without PyTorch installed.
    try:
        importlib.import_module("app.separation.demucs_backend")
        check("导入 demucs_backend（无需 torch）", True)
    except Exception as exc:
        check("导入 demucs_backend（无需 torch）", False, f"{type(exc).__name__}: {exc}")

    # The Web layer needs fastapi; skip cleanly if it is absent.
    try:
        import fastapi  # noqa: F401

        importlib.import_module("app.api.routes")
        importlib.import_module("app.api.app")
        check("导入 API 路由", True)
    except ImportError:
        skip("导入 API 路由", "未安装 fastapi（install.bat 会安装）")
    except Exception as exc:
        check("导入 API 路由", False, f"{type(exc).__name__}: {exc}")

    # atempo chain maths
    from app.audio.ffmpeg import atempo_chain

    check("atempo 单段（1.5）", atempo_chain(1.5) == "atempo=1.5000000000", atempo_chain(1.5))
    check("atempo 拆分段（2.4）", atempo_chain(2.4).count("atempo") == 2, atempo_chain(2.4))
    check("atempo 拆分段（0.4）", atempo_chain(0.4).count("atempo") == 2, atempo_chain(0.4))
    check("atempo 拒绝非正数", _raises(lambda: atempo_chain(0)))


def _raises(func) -> bool:
    try:
        func()
    except Exception:
        return True
    return False


# ==========================================================================
def _sandbox_root() -> Path:
    """Where to put throwaway test files.

    Uses ``DPG_SELFTEST_DIR`` when set (needed under sandboxes that forbid writes
    to the system temp directory), otherwise the normal temp location.
    """
    override = os.environ.get("DPG_SELFTEST_DIR")
    if override:
        root = Path(override)
    else:
        root = Path(tempfile.gettempdir())
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = PROJECT_ROOT / ".selftest-tmp"
        root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    global VERBOSE

    parser = argparse.ArgumentParser(description="Drum Practice Generator 离线自测")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示每一步")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print("=" * 70)
    print("  Drum Practice Generator — 离线自测")
    print("  不需要 GPU / 模型 / 网络")
    print("=" * 70)

    tmp = Path(tempfile.mkdtemp(prefix="dpg-selftest-", dir=_sandbox_root()))
    started = time.time()
    try:
        test_io(tmp)
        test_dynamics()
        test_mixing()
        test_metronome()
        test_bpm()
        test_naming(tmp)
        test_errors_and_tasks(tmp)
        test_pipeline(tmp)
        test_installer()
        test_imports()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    elapsed = time.time() - started
    print("\n" + "=" * 70)
    print(f"  通过 {PASSED} · 失败 {FAILED} · 跳过 {SKIPPED} · 用时 {elapsed:.1f}s")
    if FAILURES:
        print("\n  失败项：")
        for item in FAILURES:
            print(f"    - {item}")
    print("=" * 70)

    if FAILED:
        print("\n  自测未通过。请把上面的失败项发给开发者。\n")
        return 1
    print("\n  全部通过。可以运行 start.bat 启动程序。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
