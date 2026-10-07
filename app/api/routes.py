"""HTTP API routes.

The UI is a thin client: every decision lives in the backend so the same
pipeline can later be driven from a CLI or a batch script.

Uploads stream to disk in chunks.  ``UploadFile.read`` is a plain synchronous
read of the underlying SpooledTemporaryFile, so a large upload does not occupy
anyio worker threads - important because songs are 5-50 MB and the default
threadpool is small.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__, config
from ..audio import ffmpeg as ffmpeg_module
from ..audio import timestretch as stretch_module
from ..bpm import detect as bpm_module
from ..env import build_report, resolve_device
from ..errors import DrumPracticeError
from ..logging_setup import get_logger, read_history, tail
from ..pipeline import JobConfig, run_job
from ..separation import models as model_store
from ..tasks import manager
from ..tools import free_disk_bytes, no_window_kwargs

logger = get_logger("api")
router = APIRouter(prefix="/api")

UPLOAD_CHUNK = 4 * 1024 * 1024


# ==========================================================================
# Request models
# ==========================================================================
class JobRequest(BaseModel):
    """Everything the Generate button sends."""

    source_id: str | None = Field(
        default=None, description="上传返回的文件 id"
    )
    source_path: str | None = Field(
        default=None, description="本机绝对路径（不想上传大文件时使用）"
    )

    drum_volume: float = Field(default=config.DEFAULT_DRUM_VOLUME, ge=0.0, le=1.0)
    speed: float = Field(default=1.0, ge=config.SPEED_MIN, le=config.SPEED_MAX)
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$")
    output_bits: int = Field(default=config.DEFAULT_OUTPUT_BITS)

    model: str = Field(default=config.DEFAULT_MODEL)
    device: str | None = None
    segment: float | None = Field(default=config.DEFAULT_SEGMENT, ge=1, le=7.8)
    overlap: float = Field(default=config.DEFAULT_OVERLAP, ge=0.0, le=0.9)
    shifts: int = Field(default=config.DEFAULT_SHIFTS, ge=0, le=10)
    stretch_backend: str = Field(default="auto", pattern="^(auto|atempo|rubberband)$")

    detect_bpm: bool = True
    original_bpm: float | None = Field(default=None, gt=0, le=400)
    target_bpm: float | None = Field(default=None, gt=0, le=400)

    metronome_enabled: bool = False
    metronome_volume: float = Field(default=config.DEFAULT_METRONOME_VOLUME, ge=0.0, le=1.0)
    time_signature: str = Field(default=config.DEFAULT_TIME_SIGNATURE)
    metronome_offset_ms: float = Field(default=0.0, ge=-5000, le=5000)

    keep_stems: bool = False
    loudness_normalize: bool = True
    output_basename: str | None = None


class ModelDownloadRequest(BaseModel):
    model: str = Field(default=config.DEFAULT_MODEL)


class AnalyzeRequest(BaseModel):
    source_id: str | None = None
    source_path: str | None = None


# ==========================================================================
# Helpers
# ==========================================================================
def _error_response(exc: DrumPracticeError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


def _upload_dir() -> Path:
    config.ensure_directories()
    return config.TEMP_UPLOAD_DIR


def _model_ready(model: str) -> bool:
    """True when a model can actually be used.

    Two caches count: the Demucs hub cache and our offline local checkpoint
    (``models/cache/<model>.th``), which is what the backend prefers because it
    avoids minutes of HuggingFace/AWS round-trips on every run.
    """
    return (
        model_store.local_checkpoint(model) is not None
        or model_store.is_installed(model)
    )


def resolve_source(source_id: str | None, source_path: str | None) -> Path:
    """Turn either an upload id or a local path into a verified file path."""
    if source_id:
        candidate = _upload_dir() / source_id
        if not candidate.is_file():
            raise DrumPracticeError(
                "找不到已上传的文件，请重新选择歌曲。",
                suggestions=["重新拖入或选择歌曲文件。"],
            )
        return candidate

    if source_path:
        candidate = Path(source_path).expanduser()
        try:
            candidate = candidate.resolve()
        except OSError:
            pass
        if not candidate.is_file():
            raise DrumPracticeError(
                f"本地路径不存在或不是文件：{candidate}",
                suggestions=[
                    "确认路径拼写正确。",
                    "路径中的反斜杠请用 \\\\ 或改用正斜杠 /。",
                    "或者改用“选择文件”按钮上传。",
                ],
            )
        if candidate.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            from ..errors import UnsupportedFormatError

            raise UnsupportedFormatError(candidate.name, config.SUPPORTED_EXTENSIONS)
        return candidate

    raise DrumPracticeError(
        "未提供歌曲。",
        suggestions=["请先选择或拖入一个音频文件。"],
    )


# ==========================================================================
# System
# ==========================================================================
@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": __version__,
        "busy": manager.is_busy(),
        "ffmpeg": ffmpeg_module.detect() is not None,
    }


@router.get("/system")
def system_info() -> dict:
    """Environment readout for the UI banner and the diagnose panel."""
    report = build_report()
    return {
        "version": __version__,
        "os": f"{report.os_name} {report.os_version}",
        "python": report.python_version,
        "python_executable": report.python_executable,
        "torch": report.torch_version,
        "torch_cuda_build": report.torch_cuda_build,
        "cuda_available": report.torch_cuda_available,
        "cuda_version": report.torch_cuda_version,
        "gpu": report.gpu.__dict__ if report.gpu else None,
        "ffmpeg": report.ffmpeg,
        "demucs": report.demucs_version,
        "numpy": report.numpy_version,
        "free_disk_gb": round(report.free_disk_gb, 1),
        "output_dir": str(config.OUTPUT_DIR),
        "temp_dir": str(config.TEMP_DIR),
        "problems": report.problems,
        "warnings": report.warnings,
        "defaults": {
            "drum_volume": config.DEFAULT_DRUM_VOLUME,
            "model": config.DEFAULT_MODEL,
            "segment": config.DEFAULT_SEGMENT,
            "overlap": config.DEFAULT_OVERLAP,
            "shifts": config.DEFAULT_SHIFTS,
            "metronome_volume": config.DEFAULT_METRONOME_VOLUME,
            "time_signature": config.DEFAULT_TIME_SIGNATURE,
            "speed": config.DEFAULT_SPEED,
            "output_bits": config.DEFAULT_OUTPUT_BITS,
        },
        "limits": {
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_duration_seconds": config.MAX_DURATION_SECONDS,
            "speed_min": config.SPEED_MIN,
            "speed_max": config.SPEED_MAX,
        },
        "time_signatures": list(config.TIME_SIGNATURES),
        "stretch_backends": stretch_module.available_backends(),
        "supported_extensions": list(config.SUPPORTED_EXTENSIONS),
    }


@router.get("/logs")
def get_logs(lines: int = Query(default=200, ge=1, le=5000)) -> dict:
    return {"log": tail(lines)}


@router.get("/history")
def get_history(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    return {"items": read_history(limit)}


@router.post("/open-folder")
def open_folder(payload: dict = Body(default={})) -> dict:
    """Reveal a file (or the output folder) in Explorer / Finder."""
    target = payload.get("path") or str(config.OUTPUT_DIR)
    path = Path(target)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{path}")

    try:
        if sys.platform == "win32":
            if path.is_file():
                subprocess.Popen(  # noqa: S603
                    ["explorer", "/select,", str(path)], **no_window_kwargs()
                )
            else:
                os.startfile(str(path))  # noqa: S606 - local-only tool, user-chosen path
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path if path.is_dir() else path.parent)])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])  # noqa: S603
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"无法打开文件夹：{exc}") from exc

    return {"ok": True, "path": str(path)}


# ==========================================================================
# Upload / analyze
# ==========================================================================
@router.post("/upload")
async def upload(file: UploadFile) -> Any:
    """Receive a song and store it in ``temp/uploads``."""
    config.ensure_directories()

    filename = Path(file.filename or "song").name
    extension = Path(filename).suffix.lower()
    if extension not in config.SUPPORTED_EXTENSIONS:
        from ..errors import UnsupportedFormatError

        try:
            raise UnsupportedFormatError(filename, config.SUPPORTED_EXTENSIONS)
        except DrumPracticeError as exc:
            return _error_response(exc)

    # Keep the original extension: ffprobe sniffs content, but a correct
    # extension avoids confusion and helps the user recognise the file.
    target = _upload_dir() / f"{uuid.uuid4().hex}{extension}"

    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    from ..errors import FileTooLargeError

                    raise FileTooLargeError(
                        written / (1024 * 1024), config.MAX_UPLOAD_BYTES // (1024 * 1024)
                    )
                # Guard disk space mid-stream so a full disk gives a clear message.
                if written % (16 * UPLOAD_CHUNK) < UPLOAD_CHUNK:
                    if free_disk_bytes(_upload_dir()) < config.MIN_FREE_DISK_BYTES:
                        handle.close()
                        target.unlink(missing_ok=True)
                        from ..errors import InsufficientDiskSpaceError

                        raise InsufficientDiskSpaceError(
                            2.0, free_disk_bytes(_upload_dir()) / (1024**3), str(_upload_dir())
                        )
                handle.write(chunk)
    except DrumPracticeError as exc:
        return _error_response(exc)
    except OSError as exc:
        target.unlink(missing_ok=True)
        return _error_response(
            DrumPracticeError(
                f"保存上传文件失败：{exc}",
                suggestions=["检查磁盘空间与杀毒软件拦截。", "重试一次。"],
            )
        )
    finally:
        await file.close()

    if written == 0:
        target.unlink(missing_ok=True)
        return _error_response(
            DrumPracticeError("上传的文件为空。", suggestions=["重新选择歌曲文件。"])
        )

    # Probe immediately so the UI can show metadata without a second round trip.
    try:
        info = ffmpeg_module.probe(target)
    except DrumPracticeError as exc:
        target.unlink(missing_ok=True)
        from ..errors import CorruptAudioError

        # Preserve the original diagnosis when it is not actually about the
        # file: "FFmpeg is missing" (503) must not be reported as "your song is
        # corrupt" (422), or the user chases the wrong problem.
        if isinstance(exc, CorruptAudioError):
            return _error_response(exc)
        if exc.code.startswith("ffmpeg_") or exc.code == "ffmpeg_missing":
            return _error_response(exc)
        return _error_response(CorruptAudioError(filename, detail=exc.detail or exc.message))

    return {
        "source_id": target.name,
        "filename": filename,
        "size_bytes": written,
        "size_mb": round(written / (1024 * 1024), 2),
        "audio": info.to_dict(),
        "extension": extension,
    }


@router.post("/analyze")
def analyze(payload: AnalyzeRequest) -> Any:
    """Detect BPM and report metadata for an already-uploaded song.

    Runs synchronously: it is a few seconds of CPU on a 90 s excerpt and the UI
    wants the answer before the user touches the sliders.
    """
    try:
        source = resolve_source(payload.source_id, payload.source_path)
    except DrumPracticeError as exc:
        return _error_response(exc)

    try:
        info = ffmpeg_module.probe(source)
    except DrumPracticeError as exc:
        return _error_response(exc)

    # Decode a short excerpt at the model rate for tempo analysis.
    config.ensure_directories()
    excerpt = config.TEMP_UPLOAD_DIR / f"{source.stem}_bpm.wav"
    try:
        ffmpeg_module.decode_to_wav(
            source,
            excerpt,
            sample_rate=config.MODEL_SAMPLE_RATE,
            channels=1,
            bits=32,
        )
        from ..audio import io as audio_io

        audio, rate = audio_io.read_wav(excerpt)
        result = bpm_module.detect_bpm(audio, rate)
    except DrumPracticeError as exc:
        return _error_response(exc)
    finally:
        excerpt.unlink(missing_ok=True)

    return {
        "filename": source.name,
        "path": str(source),
        "audio": info.to_dict(),
        "bpm": result.to_dict(),
    }


# ==========================================================================
# Models
# ==========================================================================
@router.get("/models")
def list_models() -> dict:
    return {
        "models": [status.to_dict() for status in model_store.all_statuses()],
        "cache_dir": str(model_store.models_root()),
        "cache_size_mb": round(model_store.cache_size_mb(), 1),
        "default": config.DEFAULT_MODEL,
        "licenses": model_store.license_info(),
    }


@router.post("/models/download")
def download_model(payload: ModelDownloadRequest) -> Any:
    """Download a model in the background (spec section 14).

    Never downloads implicitly: only this explicit endpoint triggers a fetch, and
    it reports the HuggingFace repository it is pulling from.
    """
    if payload.model not in config.AVAILABLE_MODELS:
        return _error_response(
            DrumPracticeError(
                f"未知模型：{payload.model}",
                suggestions=[f"可用模型：{', '.join(config.AVAILABLE_MODELS)}"],
            )
        )

    if model_store.is_installed(payload.model):
        return {"already_installed": True, "model": payload.model}

    def _download(*, progress, should_cancel, model: str):
        def cb(fraction: float, message: str) -> None:
            if should_cancel():
                from ..errors import TaskCancelledError

                raise TaskCancelledError()
            progress(
                {
                    "progress": fraction,
                    "percent": fraction * 100,
                    "stage": "download",
                    "stage_label": "下载模型",
                    "note": message,
                }
            )

        model_store.download(model, progress=cb)
        return {"model": model, "installed": model_store.is_installed(model)}

    job = manager.submit(
        "model_download", _download, label=f"下载模型 {payload.model}",
        exclusive=False, model=payload.model,
    )
    return {"job_id": job.id, "model": payload.model}


# ==========================================================================
# Jobs
# ==========================================================================
@router.post("/jobs")
def create_job(payload: JobRequest) -> Any:
    """Queue a practice-track generation job."""
    try:
        source = resolve_source(payload.source_id, payload.source_path)
    except DrumPracticeError as exc:
        return _error_response(exc)

    if payload.model not in config.AVAILABLE_MODELS:
        return _error_response(
            DrumPracticeError(
                f"未知模型：{payload.model}",
                suggestions=[f"可用模型：{', '.join(config.AVAILABLE_MODELS)}"],
            )
        )

    if payload.time_signature not in config.TIME_SIGNATURES:
        return _error_response(
            DrumPracticeError(
                f"不支持的拍号：{payload.time_signature}",
                suggestions=[f"支持：{', '.join(config.TIME_SIGNATURES)}"],
            )
        )

    # Model presence is checked up front: failing at submit time with a clear
    # "download the model" message beats failing 30 s into the job.
    if not _model_ready(payload.model):
        from ..errors import ModelNotInstalledError

        return _error_response(ModelNotInstalledError(payload.model))

    device = payload.device
    try:
        resolve_device(device)
    except DrumPracticeError as exc:
        return _error_response(exc)

    if device is None:
        device = resolve_device(None)

    cfg = JobConfig(
        source=source,
        drum_volume=payload.drum_volume,
        speed=payload.speed,
        output_format=payload.output_format,
        output_bits=payload.output_bits,
        model=payload.model,
        device=device,
        segment=payload.segment,
        overlap=payload.overlap,
        shifts=payload.shifts,
        stretch_backend=payload.stretch_backend,
        detect_bpm=payload.detect_bpm,
        original_bpm=payload.original_bpm,
        target_bpm=payload.target_bpm,
        metronome_enabled=payload.metronome_enabled,
        metronome_volume=payload.metronome_volume,
        time_signature=payload.time_signature,
        metronome_offset_ms=payload.metronome_offset_ms,
        keep_stems=payload.keep_stems,
        loudness_normalize=payload.loudness_normalize,
        output_basename=payload.output_basename,
    )

    job = manager.submit("practice_track", run_job, cfg, label=source.name)

    body = {
        "job_id": job.id,
        "config": cfg.to_dict(),
        "queued": True,
    }
    if payload.target_bpm and payload.original_bpm:
        body["implied_speed"] = bpm_module.speed_for_bpm(payload.original_bpm, payload.target_bpm)
    return body


@router.get("/jobs")
def list_jobs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"jobs": [job.to_dict() for job in manager.list(limit)]}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> Any:
    job = manager.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "unknown_job",
                    "message": "找不到该任务（可能已过期或服务已重启）。",
                    "suggestions": ["重新点击“生成练习曲”。"],
                    "detail": None,
                }
            },
        )
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Any:
    if not manager.cancel(job_id):
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "cannot_cancel",
                    "message": "该任务无法取消（不存在或已结束）。",
                    "suggestions": ["刷新状态后重试。"],
                    "detail": None,
                }
            },
        )
    return {"ok": True, "job_id": job_id, "cancel_requested": True}


# ==========================================================================
# Output
# ==========================================================================
@router.get("/output")
def list_output(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    config.ensure_directories()
    files = []
    for path in sorted(
        config.OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    ):
        if path.is_file() and path.suffix.lower() in (".wav", ".mp3", ".flac"):
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                }
            )
        if len(files) >= limit:
            break
    return {"files": files, "directory": str(config.OUTPUT_DIR)}


@router.get("/audio/{job_id}")
def stream_audio(job_id: str):
    """Stream a finished practice track for A/B preview (spec section 16)."""
    job = manager.get(job_id)
    if job is None or not job.result:
        raise HTTPException(status_code=404, detail="任务不存在或尚未完成")

    output = (job.result or {}).get("output")
    if not output:
        raise HTTPException(status_code=404, detail="该任务没有输出文件")

    path = Path(output)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="输出文件已被移动或删除")

    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
    }
    return FileResponse(
        path,
        media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
    )


@router.get("/audio-file")
def stream_output_file(path: str = Query(...)):
    """Stream an arbitrary file from the output directory (A/B comparison)."""
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        output_root = config.OUTPUT_DIR.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"路径无效：{exc}") from exc

    # Only ever serve from output/ - this endpoint takes a path from the client.
    if output_root not in resolved.parents and resolved != output_root:
        raise HTTPException(status_code=403, detail="只能访问 output 目录内的文件")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(resolved, filename=resolved.name)
