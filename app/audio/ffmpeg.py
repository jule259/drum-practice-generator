"""FFmpeg / FFprobe discovery, verification and the audio operations built on it.

The spec says "do not assume FFmpeg is already installed" - and on this machine
it genuinely is not.  So discovery order is:

1. ``FFMPEG_BINARY`` env var - explicit override for power users
2. ``tools/ffmpeg/bin``      - the copy install.bat downloads (preferred, pinned)
3. system PATH               - if the user already had FFmpeg

Every candidate is *executed* before being trusted.  A path that exists but
cannot run (broken download, App Execution Alias, 0-byte file) is rejected with
a clear reason, because "found FFmpeg" that then fails mid-task is far worse
than "FFmpeg not found" up front.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..errors import FFmpegNotFoundError, FFmpegRunError
from ..tools import env_with_tools, run, which

BUNDLED_BIN_DIRS = (
    config.TOOLS_DIR / "ffmpeg" / "bin",
    config.TOOLS_DIR / "ffmpeg",
)

# Remembered across calls so the error message can explain *why* detection
# failed rather than just saying "not found".
_DETECTION_PROBLEMS: list[str] = []


@dataclass
class FFmpegInfo:
    """Everything we know about a verified FFmpeg installation."""

    ffmpeg: Path
    ffprobe: Path | None = None
    version: str = ""
    source: str = ""  # "bundled" | "PATH" | "env"
    filters: set[str] = field(default_factory=set)
    encoders: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)

    @property
    def has_atempo(self) -> bool:
        return "atempo" in self.filters

    @property
    def has_loudnorm(self) -> bool:
        return "loudnorm" in self.filters

    @property
    def has_alimiter(self) -> bool:
        return "alimiter" in self.filters

    def to_dict(self) -> dict:
        return {
            "found": True,
            "path": str(self.ffmpeg),
            "ffprobe": str(self.ffprobe) if self.ffprobe else None,
            "version": self.version,
            "source": self.source,
            "problems": list(self.problems),
            "filters": {
                "atempo": self.has_atempo,
                "loudnorm": self.has_loudnorm,
                "alimiter": self.has_alimiter,
            },
        }


def _pip_bundled_ffmpeg() -> Path | None:
    """Locate the FFmpeg shipped inside the ``imageio-ffmpeg`` wheel.

    ``imageio-ffmpeg`` bundles a full static FFmpeg (gyan.dev essentials build,
    ~84 MB) and installs it from PyPI.  That makes it a far more reliable way to
    obtain FFmpeg than an HTTP download from a single host - pip handles mirrors,
    retries and resume - so install.bat falls back to it, and we look for it here.

    Note it ships **no ffprobe**; ``probe()`` falls back to parsing ffmpeg's own
    stderr for that case.
    """
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError:
        return None

    try:
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # pragma: no cover - broken/partial install
        return None

    if exe.is_file() and _probe_binary(exe):
        return exe
    return None


def _candidates() -> list[tuple[Path, str]]:
    """Ordered (executable-path, source-label) pairs to try.

    These are full executable paths, not directories.  That matters for the
    pip-bundled build, whose binary is version-stamped
    (``ffmpeg-win-x86_64-v7.1.exe``) and so would never be found by looking for
    a file literally named ``ffmpeg.exe``.
    """
    out: list[tuple[Path, str]] = []

    override = os.environ.get("FFMPEG_BINARY")
    if override:
        path = Path(override)
        if path.is_file():
            out.append((path, "env"))
        else:
            for name in ("ffmpeg.exe", "ffmpeg"):
                out.append((path / name, "env"))

    for directory in BUNDLED_BIN_DIRS:
        for name in ("ffmpeg.exe", "ffmpeg"):
            out.append((directory / name, "bundled"))

    pip_exe = _pip_bundled_ffmpeg()
    if pip_exe is not None:
        out.append((pip_exe, "pip"))

    return out


def _probe_binary(path: Path) -> str | None:
    """Return FFmpeg's version token if ``path`` really runs, else None."""
    try:
        proc = run([path, "-version"], timeout=20, check=False)
    except (OSError, ValueError, Exception):  # noqa: B014 - any launch failure disqualifies
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in (proc.stdout_text or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = re.search(r"ffmpeg version (\S+)", lines[0])
    return match.group(1) if match else lines[0].strip()


def _parse_listing(text: str) -> set[str]:
    """Parse a `-filters` / `-encoders` / `-decoders` table into a set of names.

    The flag column width differs by listing!  ``-filters`` uses **3** flag
    characters while ``-encoders``/``-decoders`` use **6**::

        TSC aap               AA->A      Apply Affine Projection ...
        V..... libx264              libx264 H.264 / AVC ...

    Matching a fixed width silently yields an empty set for one of them, which
    would make the app claim a filter is missing on a perfectly good build.
    """
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s*[A-Z.]{3,7}\s+(\S+)", line)
        if match:
            names.add(match.group(1))
    return names


def _finalise(ffmpeg_path: Path, version: str, source: str) -> FFmpegInfo:
    info = FFmpegInfo(ffmpeg=ffmpeg_path, version=version, source=source)

    # ffprobe normally sits next to ffmpeg
    for name in ("ffprobe.exe", "ffprobe"):
        sibling = ffmpeg_path.parent / name
        if sibling.is_file():
            info.ffprobe = sibling
            break
    if info.ffprobe is None:
        info.ffprobe = which("ffprobe", [ffmpeg_path.parent])

    # Capability listing - drives atempo availability and mp3 encoder choice.
    try:
        proc = run([ffmpeg_path, "-hide_banner", "-filters"], timeout=60, check=False)
        info.filters = _parse_listing(proc.stdout_text or "")
    except Exception:  # pragma: no cover - defensive
        info.problems.append("无法列出 FFmpeg 滤镜")

    try:
        proc = run([ffmpeg_path, "-hide_banner", "-encoders"], timeout=60, check=False)
        info.encoders = _parse_listing(proc.stdout_text or "")
    except Exception:  # pragma: no cover - defensive
        info.problems.append("无法列出 FFmpeg 编码器")

    if not info.has_atempo:
        info.problems.append("FFmpeg 缺少 atempo 滤镜：变速功能将不可用")
    if not info.has_loudnorm:
        info.problems.append("FFmpeg 缺少 loudnorm 滤镜：响度归一化将跳过")
    if info.ffprobe is None:
        info.problems.append(
            "未找到 ffprobe：改用解析 ffmpeg 输出的方式读取音频元数据（功能正常）"
        )
    return info


@functools.lru_cache(maxsize=1)
def _detect_cached() -> FFmpegInfo | None:
    del _DETECTION_PROBLEMS[:]

    for candidate, source in _candidates():
        if not candidate.is_file():
            continue
        version = _probe_binary(candidate)
        if version:
            return _finalise(candidate, version, source)
        _DETECTION_PROBLEMS.append(f"{candidate}（存在但无法执行）")

    found = which("ffmpeg")
    if found:
        version = _probe_binary(found)
        if version:
            return _finalise(found, version, "PATH")
        _DETECTION_PROBLEMS.append(f"{found}（存在但无法执行）")

    return None


def detect() -> FFmpegInfo | None:
    """Find and verify FFmpeg (cached).  Returns None when unavailable."""
    return _detect_cached()


def reset() -> None:
    """Forget the cached detection result (used after install.bat runs)."""
    _detect_cached.cache_clear()


def get_info() -> FFmpegInfo:
    """Return verified FFmpeg info or raise a user-facing error."""
    info = detect()
    if info is None:
        detail = "\n".join(_DETECTION_PROBLEMS) if _DETECTION_PROBLEMS else None
        raise FFmpegNotFoundError(detail=detail)
    return info


def ffmpeg_path() -> Path:
    return get_info().ffmpeg


def ffprobe_path() -> Path:
    info = get_info()
    if info.ffprobe is None:
        raise FFmpegRunError(
            "未找到 ffprobe，无法读取音频元数据。",
            suggestions=["重新运行 install.bat 以完整下载 FFmpeg（含 ffprobe）。"],
        )
    return info.ffprobe


def child_env() -> dict[str, str]:
    """Environment with FFmpeg on PATH.

    Required because Demucs shells out to a bare ``ffmpeg``/``ffprobe``.
    """
    info = get_info()
    return env_with_tools([info.ffmpeg.parent, *BUNDLED_BIN_DIRS])


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------
@dataclass
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str = ""
    bit_rate: int = 0
    format_name: str = ""
    title: str = ""

    @property
    def duration_hms(self) -> str:
        total = int(round(self.duration))
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "duration_hms": self.duration_hms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "bit_rate": self.bit_rate,
            "format_name": self.format_name,
            "title": self.title,
        }


def _as_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default  # reject NaN


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def probe(path: str | Path) -> AudioInfo:
    """Read audio metadata.

    Prefers ffprobe (exact, structured JSON).  When ffprobe is unavailable - the
    case for the pip-bundled FFmpeg, which ships only ``ffmpeg.exe`` - it falls
    back to parsing the stream info ffmpeg prints on stderr, so importing a song
    still shows duration/rate/channels instead of failing.

    Raises ``FFmpegRunError`` for anything unreadable so the caller can turn it
    into a "corrupt/unreadable audio" message.
    """
    path = Path(path)
    if not path.is_file():
        raise FFmpegRunError(f"文件不存在：{path.name}")

    info = detect()
    if info is None:
        raise FFmpegNotFoundError()

    if info.ffprobe is not None:
        return _probe_with_ffprobe(path, info.ffprobe)
    return _probe_with_ffmpeg(path, info.ffmpeg)


def _probe_with_ffprobe(path: Path, ffprobe: Path) -> AudioInfo:
    proc = run(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise FFmpegRunError(
            f"FFprobe 无法读取文件：{path.name}",
            detail=(proc.stderr_text or "").strip()[-2000:],
        )

    try:
        data = json.loads(proc.stdout_text or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegRunError("FFprobe 返回了无法解析的结果。", detail=str(exc)) from exc

    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise FFmpegRunError(
            f"文件中没有音频流：{path.name}",
            suggestions=["确认这是一个音频文件，而不是视频、图片或压缩包。"],
        )

    stream = streams[0]
    fmt = data.get("format", {})

    duration = _as_float(fmt.get("duration"))
    if duration <= 0:
        duration = _as_float(stream.get("duration"))
    if duration <= 0:
        # Fall back to frame count where available; leave 0.0 if truly unknown.
        nb_frames = _as_float(stream.get("nb_frames"))
        rate = _as_float(stream.get("sample_rate"))
        if nb_frames > 0 and rate > 0:
            duration = nb_frames / rate

    tags = {**fmt.get("tags", {}), **stream.get("tags", {})}

    return AudioInfo(
        duration=duration,
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        codec=stream.get("codec_name", "") or "",
        bit_rate=_as_int(fmt.get("bit_rate")),
        format_name=fmt.get("format_name", "") or "",
        title=tags.get("title", "") or "",
    )


# ffmpeg stderr banners look like:
#   Duration: 00:03:42.35, start: 0.000000, bitrate: 320 kb/s
#   Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_BITRATE_RE = re.compile(r"bitrate:\s*(\d+)\s*kb/s")
_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Audio:\s*(?P<codec>[^,\s]+)"
    r"(?P<rest>.*)$",
    # MULTILINE is essential: without it the `$` anchor only matches at the very
    # end of the whole stderr text and the stream line is never found.
    re.MULTILINE,
)
_RATE_RE = re.compile(r"(\d+)\s*Hz")


def _probe_with_ffmpeg(path: Path, ffmpeg: Path) -> AudioInfo:
    """Metadata from ffmpeg's stderr banner (used when ffprobe is missing)."""
    proc = run(
        [ffmpeg, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        timeout=300,
        check=False,
    )
    text = proc.stderr_text or ""

    # ffmpeg exits non-zero for a bare probe in some builds; the banner is what
    # matters, so only treat it as failure when there is no stream line at all.
    stream_match = _STREAM_RE.search(text)
    if stream_match is None:
        raise FFmpegRunError(
            f"无法读取音频文件：{path.name}",
            detail=text.strip()[-2000:],
        )

    duration = 0.0
    duration_match = _DURATION_RE.search(text)
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    bit_rate = 0
    bitrate_match = _BITRATE_RE.search(text)
    if bitrate_match:
        bit_rate = int(bitrate_match.group(1)) * 1000

    rest = stream_match.group("rest") or ""
    rate_match = _RATE_RE.search(rest)
    sample_rate = int(rate_match.group(1)) if rate_match else 0

    channels = 0
    lowered = rest.lower()
    if "mono" in lowered:
        channels = 1
    elif "stereo" in lowered:
        channels = 2
    else:
        # e.g. "5.1" / "7.1" layouts
        layout = re.search(r"(\d)\.(\d)", rest)
        if layout:
            channels = int(layout.group(1)) + int(layout.group(2))

    return AudioInfo(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        codec=stream_match.group("codec") or "",
        bit_rate=bit_rate,
        format_name="",
        title="",
    )


# --------------------------------------------------------------------------
# Audio operations
# --------------------------------------------------------------------------
def _pcm_codec(bits: int) -> str:
    if bits == 32:
        return "pcm_f32le"
    if bits == 24:
        return "pcm_s24le"
    return "pcm_s16le"


def decode_to_wav(
    src: str | Path,
    dst: str | Path,
    *,
    sample_rate: int = config.MODEL_SAMPLE_RATE,
    channels: int = config.MODEL_CHANNELS,
    bits: int = 32,
    extra_filters: str | None = None,
    timeout: float | None = 1800,
) -> Path:
    """Decode any input format to a WAV file.

    Defaults to 32-bit float at the model's rate/channel layout, so the whole
    processing chain can stay float without intermediate quantisation noise.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    layout = "stereo" if channels == 2 else "mono"
    filters = extra_filters or f"aformat=channel_layouts={layout}"

    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",
        "-af", filters,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", _pcm_codec(bits),
        "-f", "wav",
        str(dst),
    ]
    proc = run(args, timeout=timeout, check=False)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        raise FFmpegRunError(
            f"音频解码失败：{src.name}",
            detail=(proc.stderr_text or "").strip()[-2000:],
            suggestions=[
                "确认文件能正常播放。",
                "文件可能已损坏或受 DRM 保护，请先转换为普通 MP3/WAV。",
            ],
        )
    return dst


def build_loudness_filter(peak_dbfs: float = -0.3, enabled: bool = True) -> str:
    """Build the final master chain: optional EBU R128 + true-peak limiter.

    ``alimiter=level=disabled`` matters: without it alimiter re-normalises the
    signal to 0 dBFS, which would undo the headroom we deliberately left.
    """
    info = get_info()
    chain: list[str] = []

    if enabled and info.has_loudnorm:
        # -14 LUFS integrated is the streaming reference; TP=-1.0 dBTP stops
        # lossy encoders from overshooting on inter-sample peaks.
        chain.append("loudnorm=I=-14:TP=-1.0:LRA=11:linear=true")

    if info.has_alimiter:
        limit = 10 ** (peak_dbfs / 20.0)
        chain.append(f"alimiter=limit={limit:.6f}:level=disabled")

    if not chain:
        # Nothing fancy available - at least guarantee no wrap-around clipping.
        chain.append(f"volume=0dB,alimiter=limit=0.97" if info.has_alimiter else "volume=0dB")

    return ",".join(chain)


def encode_output(
    src_wav: str | Path,
    dst: str | Path,
    *,
    fmt: str = "wav",
    sample_rate: int = config.MODEL_SAMPLE_RATE,
    bits: int = config.DEFAULT_OUTPUT_BITS,
    mp3_bitrate: int = 320,
    loudnorm: bool = True,
    peak_dbfs: float = -0.3,
    timeout: float | None = 1800,
) -> Path:
    """Final delivery encode from an intermediate float WAV."""
    src_wav = Path(src_wav)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = build_loudness_filter(peak_dbfs=peak_dbfs, enabled=loudnorm)

    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_wav),
        "-af", filter_chain,
        "-ar", str(sample_rate),
    ]

    if fmt == "mp3":
        dst = dst.with_suffix(".mp3")
        args += ["-c:a", "libmp3lame", "-b:a", f"{mp3_bitrate}k"]
    elif fmt == "flac":
        dst = dst.with_suffix(".flac")
        args += ["-c:a", "flac"]
    else:
        dst = dst.with_suffix(".wav")
        args += ["-c:a", _pcm_codec(bits)]

    args.append(str(dst))
    proc = run(args, timeout=timeout, check=False)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise FFmpegRunError(
            f"导出失败：{dst.name}",
            detail=(proc.stderr_text or "").strip()[-2000:],
        )
    return dst


def atempo_chain(tempo: float) -> str:
    """Build an ``atempo`` filter chain for an arbitrary tempo ratio.

    Single atempo instances only cover 0.5-2.0, so ratios outside that range are
    split across several stages (2.4 -> atempo=1.2,atempo=2.0).
    """
    if tempo <= 0:
        raise ValueError("tempo must be > 0")

    remaining = float(tempo)
    segments: list[float] = []
    atempo_min, atempo_max = 0.5, 2.0

    while remaining > atempo_max:
        segments.append(atempo_max)
        remaining /= atempo_max
    while remaining < atempo_min:
        segments.append(atempo_min)
        remaining /= atempo_min
    segments.append(remaining)

    return ",".join(f"atempo={value:.10f}" for value in segments)


def time_stretch(
    src: str | Path,
    dst: str | Path,
    *,
    tempo: float,
    sample_rate: int = config.MODEL_SAMPLE_RATE,
    channels: int = config.MODEL_CHANNELS,
    timeout: float | None = 1800,
) -> Path:
    """Pitch-preserving time-stretch via FFmpeg's ``atempo``.

    Quality note: atempo is WSOLA-based.  It is perfectly usable for practice
    and shows artefacts only at extreme ratios; the Rubber Band backend
    (``app.audio.timestretch``) is preferred automatically when available.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", atempo_chain(tempo),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_f32le",
        "-f", "wav",
        str(dst),
    ]
    proc = run(args, timeout=timeout, check=False)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        raise FFmpegRunError(
            "变速处理失败。",
            detail=(proc.stderr_text or "").strip()[-2000:],
            suggestions=[
                "把目标速度调到接近原速（例如 50%–150%）再试。",
                "重新运行 install.bat 以修复 FFmpeg。",
            ],
        )
    return dst


def mix_tracks(
    tracks: list[tuple[str | Path, float]],
    dst: str | Path,
    *,
    sample_rate: int = config.MODEL_SAMPLE_RATE,
    channels: int = config.MODEL_CHANNELS,
    duration: float | None = None,
    peak_dbfs: float | None = None,
    timeout: float | None = 1800,
) -> Path:
    """Sum weighted tracks with FFmpeg.

    This is the *fallback* mixer.  The primary mixer is numpy
    (``app.audio.mix``) because it is deterministic and unit-testable; this
    exists so the pipeline still works if the numpy path is unavailable.

    ``amix`` renormalises by input count by default - ``normalize=0`` plus
    explicit ``weights`` is what preserves the requested drum level.
    """
    tracks = [(Path(path), float(weight)) for path, weight in tracks]
    if not tracks:
        raise ValueError("mix_tracks needs at least one input")

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    args = [ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
    for path, _ in tracks:
        args += ["-i", str(path)]

    weights = " ".join(f"{weight:.6f}" for _, weight in tracks)
    filter_parts = [
        f"[{index}:a]aformat=sample_fmts=fltp:channel_layouts={'stereo' if channels == 2 else 'mono'}[t{index}]"
        for index in range(len(tracks))
    ]
    inputs = "".join(f"[t{index}]" for index in range(len(tracks)))
    # dropout_transition=0 stops amix from ramping gain when an input ends,
    # which on tracks of unequal length sounds like a volume swell.
    filter_parts.append(
        f"{inputs}amix=inputs={len(tracks)}:weights='{weights}':normalize=0:"
        f"dropout_transition=0:duration=longest[mix]"
    )
    if peak_dbfs is not None:
        limit = 10 ** (peak_dbfs / 20.0)
        filter_parts.append(f"[mix]alimiter=limit={limit:.6f}:level=disabled[out]")
        out_label = "[out]"
    else:
        out_label = "[mix]"

    args += [
        "-filter_complex", ";".join(filter_parts),
        "-map", out_label,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_f32le",
        "-f", "wav",
        str(dst),
    ]
    if duration is not None:
        args = args[:-1] + ["-t", f"{duration:.6f}", str(dst)]

    proc = run(args, timeout=timeout, check=False)
    if proc.returncode != 0 or not dst.is_file():
        raise FFmpegRunError(
            "混音失败。",
            detail=(proc.stderr_text or "").strip()[-2000:],
        )
    return dst
