"""Job orchestration: the processing chain from song to practice track.

Spec section 9 chain, implemented literally:

    read song -> analyse -> AI separate -> take non-drum stems
      -> apply drum level -> time-stretch -> add metronome
      -> loudness/peak safety -> write output

Everything here is synchronous and blocking; :mod:`app.tasks` runs it on a worker
thread.  Progress is reported through a callback and cancellation is polled at
every stage boundary, so a long job stops promptly instead of running to
completion after the user clicks Cancel.
"""

from __future__ import annotations

import gc
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .audio import ffmpeg as ffmpeg_module
from .audio import io as audio_io
from .audio import loudness as loudness_module
from .audio import mix as mix_module
from .audio import timestretch as stretch_module
from .bpm import detect as bpm_module
from .errors import (
    CorruptAudioError,
    DrumPracticeError,
    InsufficientDiskSpaceError,
    OutputNotWritableError,
    TaskCancelledError,
    UnsupportedFormatError,
)
from .logging_setup import get_logger, log_job
from .metronome import MetronomeSettings, click_count, generate_click_track
from .separation.base import SeparationCancelled
from .separation.registry import get_backend
from .tools import free_disk_bytes

logger = get_logger("pipeline")

# Stage weights, used to turn per-stage progress into one overall bar.
STAGE_WEIGHTS = {
    "decode": 0.05,
    "bpm": 0.03,
    "separate": 0.65,
    "stretch": 0.12,
    "mix": 0.07,
    "encode": 0.08,
}
STAGE_LABELS = {
    "decode": "解码音频",
    "bpm": "检测 BPM",
    "separate": "AI 分离鼓轨",
    "stretch": "调整速度",
    "mix": "重新混音",
    "encode": "写出文件",
}


@dataclass
class JobConfig:
    """Everything the user chose for one practice track."""

    source: Path
    drum_volume: float = config.DEFAULT_DRUM_VOLUME   # linear 0..1 (1.0 = original)
    speed: float = config.DEFAULT_SPEED               # 1.0 = original tempo
    output_format: str = "wav"                        # wav | mp3 | flac
    output_bits: int = config.DEFAULT_OUTPUT_BITS
    mp3_bitrate: int = 320

    model: str = config.DEFAULT_MODEL
    device: str | None = None
    segment: float | None = config.DEFAULT_SEGMENT
    overlap: float = config.DEFAULT_OVERLAP
    shifts: int = config.DEFAULT_SHIFTS
    stretch_backend: str = "auto"

    # BPM
    detect_bpm: bool = True
    original_bpm: float | None = None
    target_bpm: float | None = None

    # Metronome
    metronome_enabled: bool = False
    metronome_bpm: float | None = None
    metronome_volume: float = config.DEFAULT_METRONOME_VOLUME
    time_signature: str = config.DEFAULT_TIME_SIGNATURE
    metronome_offset_ms: float = 0.0

    # Housekeeping
    keep_stems: bool = False
    keep_temp: bool = False
    loudness_normalize: bool = True
    output_basename: str | None = None

    def resolved_speed(self) -> float:
        """Speed from target BPM when given, else the explicit speed factor."""
        if self.target_bpm and self.original_bpm and self.original_bpm > 0:
            return float(self.target_bpm) / float(self.original_bpm)
        return float(self.speed)

    def effective_metronome_bpm(self, detected_bpm: float | None) -> float:
        """Tempo the click should run at.

        The output track plays at ``original * speed``, so the click must use that
        same tempo - not the detected one - or it drifts out of sync.
        """
        if self.metronome_bpm:
            return float(self.metronome_bpm)
        base = self.original_bpm or detected_bpm
        if base and base > 0:
            return base * self.resolved_speed()
        return 120.0

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "drum_volume": self.drum_volume,
            "drum_volume_pct": round(self.drum_volume * 100),
            "speed": self.resolved_speed(),
            "speed_pct": round(self.resolved_speed() * 100),
            "output_format": self.output_format,
            "output_bits": self.output_bits,
            "model": self.model,
            "device": self.device,
            "segment": self.segment,
            "overlap": self.overlap,
            "shifts": self.shifts,
            "stretch_backend": self.stretch_backend,
            "detect_bpm": self.detect_bpm,
            "original_bpm": self.original_bpm,
            "target_bpm": self.target_bpm,
            "metronome_enabled": self.metronome_enabled,
            "metronome_volume": self.metronome_volume,
            "time_signature": self.time_signature,
            "metronome_offset_ms": self.metronome_offset_ms,
            "keep_stems": self.keep_stems,
            "loudness_normalize": self.loudness_normalize,
        }


@dataclass
class JobResult:
    output: Path
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    detected_bpm: float | None = None
    original_bpm: float | None = None
    target_bpm: float | None = None
    effective_speed: float = 1.0
    drum_volume: float = 0.0
    model: str = ""
    device: str = ""
    stretch_backend: str = ""
    metronome_clicks: int = 0
    stem_paths: dict[str, str] = field(default_factory=dict)
    mix_report: dict = field(default_factory=dict)
    loudness_lufs: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output": str(self.output),
            "output_name": self.output.name,
            "duration_seconds": round(self.duration_seconds, 2),
            "duration_hms": _hms(self.duration_seconds),
            "processing_seconds": round(self.processing_seconds, 2),
            "detected_bpm": round(self.detected_bpm, 2) if self.detected_bpm else None,
            "original_bpm": round(self.original_bpm, 2) if self.original_bpm else None,
            "target_bpm": round(self.target_bpm, 2) if self.target_bpm else None,
            "effective_speed": round(self.effective_speed, 4),
            "speed_pct": round(self.effective_speed * 100),
            "drum_volume": self.drum_volume,
            "drum_volume_pct": round(self.drum_volume * 100),
            "model": self.model,
            "device": self.device,
            "stretch_backend": self.stretch_backend,
            "metronome_clicks": self.metronome_clicks,
            "stems": self.stem_paths,
            "mix_report": self.mix_report,
            "loudness_lufs": round(self.loudness_lufs, 2) if self.loudness_lufs else None,
            "notes": self.notes,
        }


def _hms(seconds: float) -> str:
    total = int(round(seconds or 0))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _sanitize_stem(name: str, max_length: int = 60) -> str:
    """Make a filename-safe stem from a song title."""
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "_" for char in name
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return (cleaned or "track")[:max_length]


def build_output_name(cfg: JobConfig, target_bpm: float | None) -> str:
    """Human-readable output name that encodes the key parameters.

    Spec section 10 wants e.g. ``Song_Drums10_BPM90.wav``.  We include the speed
    only when it is not implied by the BPM, to keep names short.
    """
    base = cfg.output_basename or Path(cfg.source).stem
    base = _sanitize_stem(base)

    drum_pct = int(round(cfg.drum_volume * 100))
    parts = [base, f"Drums{drum_pct}"]

    speed = cfg.resolved_speed()
    if target_bpm:
        parts.append(f"BPM{int(round(target_bpm))}")
    elif abs(speed - 1.0) > 1e-3:
        parts.append(f"Speed{int(round(speed * 100))}")

    if cfg.metronome_enabled:
        parts.append("Click")

    suffix = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac"}.get(cfg.output_format, ".wav")
    return "_".join(parts) + suffix


def unique_path(path: Path) -> Path:
    """Avoid silently overwriting an existing practice track."""
    if not path.exists():
        return path
    for counter in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")


class ProgressReporter:
    """Aggregates per-stage progress into one monotonic overall fraction."""

    def __init__(self, callback, should_cancel=None) -> None:
        self._callback = callback
        self._should_cancel = should_cancel
        self._stage = "decode"
        self._stage_fraction = 0.0
        self._last_overall = 0.0

    def stage(self, name: str, fraction: float = 0.0, note: str = "") -> None:
        self._stage = name
        self._stage_fraction = max(0.0, min(1.0, fraction))
        self._emit(note or STAGE_LABELS.get(name, name))

    def update(self, fraction: float, note: str = "") -> None:
        self._stage_fraction = max(0.0, min(1.0, float(fraction)))
        self._emit(note or STAGE_LABELS.get(self._stage, self._stage))

    def check_cancel(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise TaskCancelledError()

    def _overall(self) -> float:
        completed = 0.0
        for name, weight in STAGE_WEIGHTS.items():
            if name == self._stage:
                completed += weight * self._stage_fraction
                break
            completed += weight
        return min(0.999, completed)

    def _emit(self, note: str) -> None:
        overall = self._overall()
        # Monotonic: the bar must never jump backwards.
        self._last_overall = max(self._last_overall, overall)
        if self._callback:
            self._callback(
                {
                    "progress": round(self._last_overall, 4),
                    "percent": round(self._last_overall * 100, 1),
                    "stage": self._stage,
                    "stage_label": STAGE_LABELS.get(self._stage, self._stage),
                    "note": note,
                }
            )


def run_job(
    cfg: JobConfig,
    *,
    progress=None,
    should_cancel=None,
    job_id: str = "local",
) -> JobResult:
    """Execute one practice-track generation job.

    ``progress`` and ``should_cancel`` are the keyword names the task manager
    injects, so they must match :meth:`app.tasks.TaskManager.submit` exactly.

    Raises a :class:`DrumPracticeError` subclass for anything the user can act on.
    """
    started = time.perf_counter()
    config.ensure_directories()

    reporter = ProgressReporter(progress, should_cancel)
    source = Path(cfg.source)

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if not source.is_file():
        raise CorruptAudioError(source.name, detail="文件不存在")

    if source.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(source.name, config.SUPPORTED_EXTENSIONS)

    try:
        (config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        probe_file = config.OUTPUT_DIR / ".write_test"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
    except OSError as exc:
        raise OutputNotWritableError(str(config.OUTPUT_DIR), detail=str(exc)) from exc

    needed_bytes = source.stat().st_size * 6
    free_bytes = free_disk_bytes(config.TEMP_DIR)
    if free_bytes < needed_bytes:
        raise InsufficientDiskSpaceError(
            needed_bytes / (1024**3), free_bytes / (1024**3), str(config.TEMP_DIR)
        )

    job_dir = config.TEMP_SEPARATION_DIR / job_id
    mix_dir = config.TEMP_MIX_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    result = JobResult(
        output=Path(),
        drum_volume=cfg.drum_volume,
        model=cfg.model,
        device=cfg.device or "auto",
    )
    notes: list[str] = []
    detected_bpm: float | None = None
    original_bpm: float | None = cfg.original_bpm

    try:
        # --------------------------------------------------------------
        # 1. Decode
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("decode", 0.05, "读取歌曲…")

        decoded = job_dir / "input_44k.wav"
        ffmpeg_module.decode_to_wav(
            source,
            decoded,
            sample_rate=config.MODEL_SAMPLE_RATE,
            channels=config.MODEL_CHANNELS,
            bits=32,
        )
        audio, rate = audio_io.read_wav(decoded)
        duration = audio.shape[1] / rate if rate else 0.0

        if duration <= 0.5:
            raise CorruptAudioError(source.name, detail=f"解码后时长仅 {duration:.2f} 秒")

        if duration > config.MAX_DURATION_SECONDS:
            notes.append(
                f"歌曲较长（{_hms(duration)}），处理时间可能超过 {int(duration / 10)} 分钟。"
            )

        result.duration_seconds = duration
        reporter.stage("decode", 1.0, f"已解码 {_hms(duration)}")

        # --------------------------------------------------------------
        # 2. BPM (never fatal)
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("bpm", 0.2, "检测 BPM…")

        if cfg.detect_bpm:
            bpm_result = bpm_module.detect_bpm(audio, rate)
            if bpm_result.ok and bpm_result.bpm:
                detected_bpm = bpm_result.bpm
                notes.append(
                    f"自动检测 BPM：{detected_bpm:.1f}（置信度 {bpm_result.confidence:.2f}）"
                )
            else:
                notes.append(f"BPM 自动检测未成功：{bpm_result.reason}（可手动填写）")
            result.detected_bpm = detected_bpm

        if original_bpm is None:
            original_bpm = detected_bpm
        result.original_bpm = original_bpm

        effective_speed = cfg.resolved_speed()
        result.effective_speed = effective_speed
        result.target_bpm = cfg.target_bpm or (
            original_bpm * effective_speed if original_bpm else None
        )

        reporter.stage("bpm", 1.0, "BPM 检测完成")

        # Release the decoded copy; the separator re-reads from disk so we do
        # not hold a second copy of a long song in RAM through separation.
        del audio
        gc.collect()

        # --------------------------------------------------------------
        # 3. AI separation
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("separate", 0.0, "准备 AI 分离…")

        backend = get_backend(
            "demucs",
            model=cfg.model,
            device=cfg.device,
            segment=cfg.segment,
            overlap=cfg.overlap,
            shifts=cfg.shifts,
        )

        def sep_progress(fraction: float, note: str) -> None:
            if should_cancel is not None and should_cancel():
                raise SeparationCancelled()
            reporter.update(fraction, note)

        try:
            separation = backend.separate(
                decoded,
                job_dir,
                progress=sep_progress,
                should_cancel=should_cancel,
            )
        except SeparationCancelled as exc:
            raise TaskCancelledError() from exc

        result.model = separation.model
        result.device = separation.device
        notes.extend(separation.notes)

        # --------------------------------------------------------------
        # 4. Load stems
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("stretch", 0.02, "载入分离音轨…")

        drums, stem_rate = audio_io.read_wav(separation.drums)
        bed, bed_rate = audio_io.read_wav(separation.no_drums)
        if stem_rate != bed_rate:
            raise DrumPracticeError(
                "分离音轨采样率不一致，无法混音。",
                suggestions=["重试一次；若持续出现请提交日志。"],
            )
        rate = stem_rate

        frames = min(drums.shape[1], bed.shape[1])
        drums = drums[:, :frames]
        bed = bed[:, :frames]

        # --------------------------------------------------------------
        # 5. Time-stretch
        # --------------------------------------------------------------
        stretch_backend_used = "none"
        if abs(effective_speed - 1.0) > 1e-6:
            reporter.check_cancel()
            reporter.stage(
                "stretch", 0.1,
                f"变速到 {effective_speed * 100:.0f}%（音高不变）…",
            )

            stretched_drums = mix_dir / "drums_stretched.wav"
            stretched_bed = mix_dir / "bed_stretched.wav"

            info_drums = stretch_module.stretch(
                separation.drums, stretched_drums,
                tempo=effective_speed, sample_rate=rate,
                channels=drums.shape[0], backend=cfg.stretch_backend,
            )
            reporter.stage("stretch", 0.6, "变速中…（伴奏轨）")
            info_bed = stretch_module.stretch(
                separation.no_drums, stretched_bed,
                tempo=effective_speed, sample_rate=rate,
                channels=bed.shape[0], backend=cfg.stretch_backend,
            )
            stretch_backend_used = info_bed.backend

            drums, _ = audio_io.read_wav(stretched_drums)
            bed, _ = audio_io.read_wav(stretched_bed)

            frames = min(drums.shape[1], bed.shape[1])
            drums = drums[:, :frames]
            bed = bed[:, :frames]

            notes.append(
                f"变速后端：{stretch_backend_used}，{effective_speed * 100:.1f}% "
                f"（{info_drums.input_duration or separation.duration_seconds:.1f}s → "
                f"{frames / rate:.1f}s）"
            )
        else:
            notes.append("速度 100%：跳过滤变处理。")

        result.stretch_backend = stretch_backend_used
        result.duration_seconds = frames / rate if rate else duration

        # --------------------------------------------------------------
        # 6. Metronome + mix
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("mix", 0.15, "重新混音…")

        metro_bpm = cfg.effective_metronome_bpm(detected_bpm or original_bpm)
        settings = MetronomeSettings(
            enabled=cfg.metronome_enabled,
            bpm=metro_bpm,
            volume=cfg.metronome_volume,
            time_signature=cfg.time_signature,
            offset_ms=cfg.metronome_offset_ms,
        )
        clicks = generate_click_track(
            settings, rate, frames, channels=bed.shape[0]
        )
        result.metronome_clicks = click_count(settings, rate, frames) if cfg.metronome_enabled else 0
        if cfg.metronome_enabled:
            notes.append(
                f"节拍器：{cfg.time_signature} @ {metro_bpm:.1f} BPM，"
                f"音量 {cfg.metronome_volume * 100:.0f}%，共 {result.metronome_clicks} 拍"
            )

        mix_input = bed + clicks if cfg.metronome_enabled else bed

        mixed, mix_report = mix_module.mix_stems(
            mix_input, drums, cfg.drum_volume, rate,
            do_normalize=True,
            do_limit=True,
            ceiling=config.PEAK_TARGET,
        )
        result.mix_report = mix_report.to_dict()
        notes.extend(mix_report.notes)
        reporter.stage(
            "mix", 1.0,
            f"混音完成（鼓 {cfg.drum_volume * 100:.0f}%，峰值 "
            f"{mix_report.peak_after_limiter:.3f}）",
        )

        # --------------------------------------------------------------
        # 7. Encode
        # --------------------------------------------------------------
        reporter.check_cancel()
        reporter.stage("encode", 0.15, "写出文件…")

        mix_wav = mix_dir / "mix_master.wav"
        audio_io.write_wav(mix_wav, mixed, rate, bits=32)

        target_bpm_for_name = result.target_bpm
        output_name = build_output_name(cfg, target_bpm_for_name)
        output_path = unique_path(config.OUTPUT_DIR / output_name)

        final_path = ffmpeg_module.encode_output(
            mix_wav,
            output_path,
            fmt=cfg.output_format,
            sample_rate=rate,
            bits=cfg.output_bits,
            mp3_bitrate=cfg.mp3_bitrate,
            loudnorm=cfg.loudness_normalize,
        )

        result.output = final_path
        reporter.stage("encode", 0.9, "测量响度…")

        loud = loudness_module.integrated_lufs(final_path)
        result.loudness_lufs = loud

        # --------------------------------------------------------------
        # 8. Preserve stems if requested
        # --------------------------------------------------------------
        if cfg.keep_stems:
            keep_dir = config.OUTPUT_DIR / f"{final_path.stem}_stems"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for stem_name, stem_path in separation.stems.items():
                if Path(stem_path).is_file() and stem_name in ("drums", "vocals", "bass", "other"):
                    shutil.copy2(stem_path, keep_dir / f"{stem_name}.wav")
            notes.append(f"已保留分轨到 {keep_dir}")
            result.stem_paths = {
                name: str(keep_dir / f"{name}.wav")
                for name in ("drums", "vocals", "bass", "other")
                if (keep_dir / f"{name}.wav").is_file()
            }

        result.notes = notes
        result.processing_seconds = time.perf_counter() - started

        log_job(
            {
                "job_id": job_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "song": source.name,
                "duration": _hms(result.duration_seconds),
                "model": result.model,
                "gpu": _gpu_label(result.device),
                "original_bpm": _round_or_none(result.original_bpm),
                "target_bpm": _round_or_none(result.target_bpm),
                "drum_volume_pct": round(cfg.drum_volume * 100),
                "speed_pct": round(effective_speed * 100),
                "metronome": (
                    f"{cfg.time_signature} @ {metro_bpm:.1f} BPM vol {cfg.metronome_volume:.2f}"
                    if cfg.metronome_enabled
                    else "关闭"
                ),
                "stretch_backend": stretch_backend_used,
                "separation_seconds": round(separation.processing_seconds, 1),
                "processing_seconds": round(result.processing_seconds, 1),
                "realtime_factor": round(
                    result.processing_seconds / result.duration_seconds, 2
                )
                if result.duration_seconds
                else None,
                "output": final_path.name,
                "loudness_lufs": _round_or_none(loud),
                "status": "SUCCESS",
            }
        )

        reporter.stage("encode", 1.0, "完成")
        return result

    except (DrumPracticeError, TaskCancelledError):
        raise
    except Exception as exc:
        logger.error("任务异常：%s\n%s", exc, traceback.format_exc())
        log_job(
            {
                "job_id": job_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "song": source.name,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise DrumPracticeError(
            f"处理过程中出现未预期的错误：{type(exc).__name__}: {exc}",
            suggestions=[
                "重试一次。",
                "查看 logs\\drum-practice.log 获取完整堆栈。",
                "运行 diagnose.bat 检查环境。",
            ],
            detail=traceback.format_exc()[-4000:],
        ) from exc
    finally:
        # --------------------------------------------------------------
        # Temp cleanup (spec section 15)
        # --------------------------------------------------------------
        if cfg.keep_temp or cfg.keep_stems:
            logger.info("保留临时文件：%s", job_dir)
        else:
            for directory in (job_dir, mix_dir):
                try:
                    shutil.rmtree(directory, ignore_errors=True)
                except OSError:
                    pass
        gc.collect()


def _round_or_none(value: float | None, digits: int = 2):
    return round(value, digits) if value else None


def _gpu_label(device: str) -> str:
    if device != "cuda":
        return "CPU"
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA GPU"
