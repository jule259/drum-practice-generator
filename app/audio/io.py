"""Audio file I/O in float32.

All internal audio is represented as a numpy ``float32`` array shaped ``(channels,
frames)`` - i.e. **channel-major**, matching torch/Demucs conventions.  Keeping
one convention everywhere removes an entire class of transposition bugs.

Why the WAV handling is hand-rolled instead of using :mod:`wave`:

* Python's ``wave`` module **cannot read IEEE-float WAV** (format tag 3) - it
  raises ``wave.Error: unknown format: 3``.  That is exactly the format our
  FFmpeg decode step writes and the format the whole float pipeline uses.
* ``wave`` **cannot write IEEE-float WAV either**: it always emits format tag 1
  (integer PCM), so 32-bit float samples get written with a header claiming
  integer PCM.  Reading such a file back as integers silently scrambles it
  (0.1 comes back as 0.48) - a data-corruption bug that round-trips cleanly and
  is therefore invisible to a naive test.

So we parse and emit the RIFF header ourselves (a few dozen lines, stdlib only)
and support both integer PCM and IEEE float, including the
``WAVE_FORMAT_EXTENSIBLE`` wrapper that many encoders produce.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = [
    "read_wav",
    "write_wav",
    "read_audio",
    "peak",
    "peak_dbfs",
    "rms_dbfs",
    "check_same_shape",
    "fit_shape",
]

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

# Chunk ids we care about; everything else is skipped.
_FMT = b"fmt "
_DATA = b"data"
_RIFF = b"RIFF"
_WAVE = b"WAVE"


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------
def _parse_header(path: Path) -> tuple[int, int, int, int, int]:
    """Return ``(format_tag, channels, rate, bits, data_offset)``.

    ``data_offset`` points at the first payload byte; samples can then be read
    with :func:`numpy.fromfile` at that offset.
    """
    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[:4] != _RIFF or riff[8:12] != _WAVE:
            raise ValueError(f"不是有效的 WAV 文件（缺少 RIFF/WAVE 头）：{path.name}")

        format_tag = 0
        channels = 0
        rate = 0
        bits = 0
        data_offset = -1

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:8], "little")
            payload_start = handle.tell()

            if chunk_id == _FMT:
                body = handle.read(chunk_size)
                if len(body) < 16:
                    raise ValueError(f"WAV fmt 块损坏：{path.name}")
                format_tag = int.from_bytes(body[0:2], "little")
                channels = int.from_bytes(body[2:4], "little")
                rate = int.from_bytes(body[4:8], "little")
                bits = int.from_bytes(body[14:16], "little")
                if format_tag == WAVE_FORMAT_EXTENSIBLE and len(body) >= 26:
                    # The real format tag lives in the first 2 bytes of the
                    # SubFormat GUID at offset 24.
                    format_tag = int.from_bytes(body[24:26], "little")
            elif chunk_id == _DATA:
                data_offset = payload_start
                break

            # Chunks are word-aligned: skip the payload plus any pad byte.
            handle.seek(payload_start + chunk_size + (chunk_size % 2))

    if channels <= 0 or rate <= 0 or bits <= 0:
        raise ValueError(f"WAV 文件头不完整（缺少 fmt 块）：{path.name}")
    if data_offset < 0:
        raise ValueError(f"WAV 文件缺少 data 块：{path.name}")

    return format_tag, channels, rate, bits, data_offset


def _int_to_float(data: np.ndarray, bits: int) -> np.ndarray:
    """Convert integer PCM samples to normalised float32."""
    if bits == 8:
        # 8-bit WAV is unsigned.
        return ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if bits == 16:
        return (data.astype(np.float32) / 32768.0).astype(np.float32)
    if bits == 24:
        return (data.astype(np.float32) / 8388608.0).astype(np.float32)
    if bits == 32:
        return (data.astype(np.float32) / 2147483648.0).astype(np.float32)
    raise ValueError(f"Unsupported PCM bit depth: {bits}")


def _decode_int24(raw: np.ndarray) -> np.ndarray:
    """Sign-extend 24-bit little-endian samples held as a flat uint8 array."""
    raw = raw.reshape(-1, 3)
    padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
    padded[:, :3] = raw
    ints = (padded.view("<i4").reshape(-1).astype(np.int32) << 8) >> 8
    return _int_to_float(ints, 24)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file into ``(channels, frames)`` float32 plus its sample rate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音频文件：{path}")

    format_tag, channels, rate, bits, data_offset = _parse_header(path)

    is_float = format_tag == WAVE_FORMAT_IEEE_FLOAT
    with path.open("rb") as handle:
        handle.seek(data_offset)
        if is_float and bits == 32:
            data = np.fromfile(handle, dtype="<f4").astype(np.float32)
        elif is_float and bits == 64:
            data = np.fromfile(handle, dtype="<f8").astype(np.float32)
        elif bits == 32:
            data = _int_to_float(np.fromfile(handle, dtype="<i4"), 32)
        elif bits == 16:
            data = _int_to_float(np.fromfile(handle, dtype="<i2"), 16)
        elif bits == 8:
            data = _int_to_float(np.fromfile(handle, dtype=np.uint8), 8)
        elif bits == 24:
            data = _decode_int24(np.fromfile(handle, dtype=np.uint8))
        else:
            raise ValueError(f"不支持的 WAV 位深：{bits} bit（{path.name}）")

    if data.size == 0:
        return np.zeros((channels, 0), dtype=np.float32), rate

    usable = (data.size // channels) * channels
    data = data[:usable].reshape(-1, channels).T
    return np.ascontiguousarray(data, dtype=np.float32), rate


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_wav(
    path: str | Path,
    audio: np.ndarray,
    rate: int,
    *,
    bits: int = 32,
) -> Path:
    """Write ``(channels, frames)`` audio as a WAV file.

    ``bits=32`` writes a genuine IEEE-float WAV (format tag 3), which is the
    internal working format.  ``bits=16``/``24`` write integer PCM, clipping
    cleanly at full scale.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D audio, got shape {audio.shape}")

    channels, _ = audio.shape
    interleaved = np.ascontiguousarray(audio.T)  # (frames, channels)

    if bits == 32:
        format_tag = WAVE_FORMAT_IEEE_FLOAT
        payload = interleaved.astype("<f4").tobytes()
    else:
        format_tag = WAVE_FORMAT_PCM
        clipped = np.clip(interleaved, -1.0, 1.0)
        if bits == 16:
            payload = (clipped * 32767.0).astype("<i2").tobytes()
        elif bits == 24:
            ints = (clipped * 8388607.0).astype("<i4")
            payload = np.ascontiguousarray(ints.view(np.uint8).reshape(-1, 4)[:, :3]).tobytes()
        elif bits == 8:
            payload = ((clipped * 127.0) + 128.0).astype(np.uint8).tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {bits}")

    block_align = channels * (bits // 8)
    byte_rate = rate * block_align
    fmt_chunk = struct.pack(
        "<HHIIHH", format_tag, channels, rate, byte_rate, block_align, bits
    )
    # fact chunk is required for non-PCM formats by the spec.
    fact_chunk = struct.pack("<I", len(interleaved)) if format_tag != WAVE_FORMAT_PCM else b""

    chunks = [
        (_FMT, fmt_chunk),
        (b"fact", fact_chunk),
        (_DATA, payload),
    ]
    body = b"".join(
        chunk_id + struct.pack("<I", len(content)) + content
        + (b"\x00" if len(content) % 2 else b"")
        for chunk_id, content in chunks
        if content or chunk_id == _FMT
    )

    with path.open("wb") as handle:
        handle.write(_RIFF + struct.pack("<I", 4 + len(body)) + _WAVE + body)

    return path


def read_audio(
    path: str | Path,
    *,
    target_rate: int | None = None,
    target_channels: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read a WAV file, optionally enforcing a rate / channel layout.

    Resampling goes through FFmpeg (high quality, already a hard dependency);
    channel conversion follows Demucs/BS.1770 practice: down-mix by averaging,
    up-mix by duplication.
    """
    audio, rate = read_wav(path)

    if target_channels is not None and audio.shape[0] != target_channels:
        if target_channels == 1:
            audio = audio.mean(axis=0, keepdims=True)
        elif audio.shape[0] == 1:
            audio = np.repeat(audio, target_channels, axis=0)
        elif audio.shape[0] > target_channels:
            audio = audio[:target_channels]
        else:
            raise ValueError(
                f"Cannot convert {audio.shape[0]} channels to {target_channels}"
            )
        audio = np.ascontiguousarray(audio, dtype=np.float32)

    if target_rate is not None and rate != target_rate:
        # Imported lazily so this module stays importable without FFmpeg.
        from . import ffmpeg as ffmpeg_module

        tmp = Path(path).with_suffix(".resample.wav")
        ffmpeg_module.decode_to_wav(
            path,
            tmp,
            sample_rate=target_rate,
            channels=audio.shape[0],
            bits=32,
        )
        audio, rate = read_wav(tmp)
        tmp.unlink(missing_ok=True)

    return np.ascontiguousarray(audio, dtype=np.float32), rate


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------
def peak(audio: np.ndarray) -> float:
    """Absolute peak sample value (0.0 for silence)."""
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def peak_dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS; ``-inf`` for digital silence."""
    value = peak(audio)
    return 20.0 * float(np.log10(value)) if value > 0 else float("-inf")


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS; ``-inf`` for digital silence."""
    if audio.size == 0:
        return float("-inf")
    value = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * float(np.log10(value)) if value > 0 else float("-inf")


def check_same_shape(*arrays: np.ndarray) -> tuple[int, int]:
    """Return the smallest common shape across ``arrays``."""
    if not arrays:
        raise ValueError("check_same_shape needs at least one array")
    channels = min(a.shape[0] for a in arrays)
    frames = min(a.shape[1] for a in arrays)
    return channels, frames


def fit_shape(audio: np.ndarray, channels: int, frames: int) -> np.ndarray:
    """Zero-pad (or truncate) ``audio`` to ``(channels, frames)``.

    Separation stems should already match, but tolerating a frame of rounding
    difference is what stops a long job from failing at the very last step.
    """
    if audio.shape == (channels, frames):
        return audio

    out = np.zeros((channels, frames), dtype=np.float32)
    src_channels = min(channels, audio.shape[0])
    src_frames = min(frames, audio.shape[1])
    out[:src_channels, :src_frames] = audio[:src_channels, :src_frames]
    return out
