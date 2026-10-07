"""Logging setup.

Spec section 18 asks for a log line per job containing time, song, model, GPU,
processing time, BPM, drum volume, speed, output file and errors.  Those are
written as both a human-readable block and a machine-readable JSON line, so the
file is useful to a person and to a future "job history" screen.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

from . import APP_NAME, __version__, config

_LOGGER_NAME = "dpg"
_configured = False

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup(level: int = logging.INFO, *, console: bool = True) -> logging.Logger:
    """Configure the app logger once.  Safe to call repeatedly."""
    global _configured

    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    config.ensure_directories()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # One file per day, rotated at 5 MB, keeping a week.
    log_file = config.LOGS_DIR / "drum-practice.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # A read-only logs dir must not stop the app from starting.
        pass

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(level)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    _configured = True
    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Child logger, e.g. ``get_logger('pipeline')`` -> ``dpg.pipeline``."""
    setup()
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}" if suffix else _LOGGER_NAME)


def banner(report) -> None:
    """Log the startup environment block required by spec section 12."""
    logger = get_logger("startup")
    logger.info("=" * 68)
    logger.info("%s v%s", APP_NAME, __version__)
    logger.info("=" * 68)
    logger.info("OS            : %s %s (%s)", report.os_name, report.os_version, report.machine)
    logger.info("Python        : %s (%d-bit) %s", report.python_version, report.python_bits,
                "venv" if report.is_venv else "system")
    logger.info("Executable    : %s", report.python_executable)
    logger.info("NumPy         : %s", report.numpy_version or "未安装")

    if report.torch_installed:
        logger.info("PyTorch       : %s", report.torch_version)
        logger.info("CUDA (torch)  : %s", report.torch_cuda_build or "CPU-only build")
        logger.info("CUDA available: %s", report.torch_cuda_available)
        if report.gpu is not None:
            gpu = report.gpu
            logger.info("GPU           : %s", gpu.name)
            logger.info("Compute       : sm_%s%s", gpu.compute_capability.replace(".", ""),
                        "  (Blackwell)" if gpu.is_blackwell else "")
            logger.info("VRAM          : %d MB total, %d MB free", gpu.total_vram_mb, gpu.free_vram_mb)
        else:
            logger.info("GPU           : 未检测到")
    else:
        logger.info("PyTorch       : 未安装")

    if report.ffmpeg.get("found"):
        logger.info("FFmpeg        : %s (%s)", report.ffmpeg.get("version"),
                    report.ffmpeg.get("source"))
        logger.info("FFmpeg path   : %s", report.ffmpeg.get("path"))
    else:
        logger.info("FFmpeg        : 未找到")

    logger.info("Demucs        : %s", report.demucs_version or "未安装")
    logger.info("SoundFile     : %s", "已安装" if report.soundfile_installed else "未安装")
    logger.info("Free disk     : %.1f GB", report.free_disk_gb)
    logger.info("Default model : %s", config.DEFAULT_MODEL)

    for problem in report.problems:
        logger.warning("问题: %s", problem)
    for warning in report.warnings:
        logger.info("提示: %s", warning)
    logger.info("=" * 68)


def log_job(job: dict) -> None:
    """Write the per-job summary block (spec section 18)."""
    logger = get_logger("job")

    def _fmt(value, suffix: str = "") -> str:
        if value is None or value == "":
            return "—"
        return f"{value}{suffix}"

    logger.info("-" * 68)
    logger.info("时间        : %s", job.get("timestamp") or datetime.now().strftime(DATE_FORMAT))
    logger.info("歌曲        : %s", _fmt(job.get("song")))
    logger.info("时长        : %s", _fmt(job.get("duration")))
    logger.info("模型        : %s", _fmt(job.get("model")))
    logger.info("设备/GPU    : %s", _fmt(job.get("gpu")))
    logger.info("原始 BPM    : %s", _fmt(job.get("original_bpm")))
    logger.info("目标 BPM    : %s", _fmt(job.get("target_bpm")))
    logger.info("鼓声音量    : %s", _fmt(job.get("drum_volume_pct"), "%"))
    logger.info("速度        : %s", _fmt(job.get("speed_pct"), "%"))
    logger.info("节拍器      : %s", _fmt(job.get("metronome")))
    logger.info("变速后端    : %s", _fmt(job.get("stretch_backend")))
    logger.info("分离耗时    : %s", _fmt(job.get("separation_seconds"), " s"))
    logger.info("总处理时间  : %s", _fmt(job.get("processing_seconds"), " s"))
    logger.info("实时倍率    : %s", _fmt(job.get("realtime_factor"), "x"))
    logger.info("输出文件    : %s", _fmt(job.get("output")))
    logger.info("状态        : %s", job.get("status", "UNKNOWN"))
    if job.get("error"):
        logger.error("错误信息    : %s", job["error"])
    logger.info("-" * 68)

    # Machine-readable copy for future history/batch features.
    try:
        history = config.LOGS_DIR / "history.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_history(limit: int = 50) -> list[dict]:
    """Read the most recent job records (newest first)."""
    path = config.LOGS_DIR / "history.jsonl"
    if not path.is_file():
        return []

    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return list(reversed(records[-limit:]))


def tail(lines: int = 200) -> str:
    """Return the tail of today's log file (used by the UI's log viewer)."""
    path: Path = config.LOGS_DIR / "drum-practice.log"
    if not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])
