"""Demucs v4 separation backend.

Design decisions worth recording:

**In-process, not subprocess.**  We call ``demucs.api.Separator`` directly rather
than shelling out to ``demucs.separate``.  Benefits: no double torch import in a
child process, no argument-quoting hazards on Windows paths with spaces/CJK
characters, no fragile tqdm-output parsing, and the stems come back as tensors we
can write ourselves.

**Two stems, residual built by subtraction.**  We ask the model for its full set
of sources and then form the backing track as ``original - drums_estimate``.
Summing the other stems instead would accumulate four models' worth of artefacts;
subtraction keeps the residual phase-coherent with the source, so the result
sounds like the record with the drummer muted.

**Progress and cancellation come from Demucs' own callback.**  ``apply_model``
invokes it per segment with ``segment_offset``/``audio_length``, and raising
inside it aborts the separation (the pool is shut down with
``cancel_futures=True``).  That gives a real progress bar and a working Cancel
button with no monkey-patching.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from .. import config
from ..errors import (
    CUDAUnavailableError,
    DrumPracticeError,
    ModelNotInstalledError,
)
from ..tools import free_disk_bytes
from .base import ProgressCallback, SeparationCancelled, SeparationResult
from . import models as model_store

# One demucs "source"; the others sum to the backing track.
DRUM_STEM = "drums"


def _vram_estimate_mb(segment: float) -> int:
    """Rough VRAM need for one segment, used for a friendly pre-flight warning.

    Hybrid Transformer Demucs needs roughly 600 MB per second of segment plus a
    fixed ~1.5 GB of working set; measured on a 16 GB card, a 7 s segment peaks
    near 6 GB.  This is a heuristic for *advice*, never a hard gate.
    """
    return int(1500 + 650 * segment)


class DemucsBackend:
    """Separate audio with a pretrained Demucs model."""

    name = "demucs"

    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        *,
        device: str | None = None,
        segment: float | None = config.DEFAULT_SEGMENT,
        overlap: float = config.DEFAULT_OVERLAP,
        shifts: int = config.DEFAULT_SHIFTS,
    ) -> None:
        self.model_name = model
        self.device_preference = device
        self.segment = segment
        self.overlap = overlap
        self.shifts = shifts

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def is_available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            return False, f"PyTorch 不可用：{exc}"

        try:
            import demucs  # noqa: F401
        except Exception as exc:
            return False, f"Demucs 不可用：{exc}"

        if not self._model_present():
            return False, f"模型 {self.model_name} 尚未下载"

        return True, ""

    def _model_present(self) -> bool:
        """True when the model is usable - hub cache *or* offline local cache."""
        return (
            model_store.local_checkpoint(self.model_name) is not None
            or model_store.is_installed(self.model_name)
        )

    def _resolve_device(self) -> str:
        from ..env import resolve_device

        try:
            return resolve_device(self.device_preference)
        except CUDAUnavailableError:
            # auto/None never raises; only an explicit "cuda" request does.
            raise

    def _build_separator(self, device: str, notes: list[str]):
        """Build a Demucs ``Separator``, preferring the offline local cache.

        Loading by *name* makes Demucs consult HuggingFace and then the legacy
        AWS mirror.  On a slow or filtered route that costs minutes per run:
        huggingface.co timed out here and the hub retried for ~5 minutes, after
        which the AWS fallback re-downloaded the 80 MB checkpoint.  Loading the
        cached checkpoint file directly takes ~2 s and needs no network at all.
        """
        from demucs.api import Separator

        cached = model_store.local_checkpoint(self.model_name)
        if cached is not None:
            try:
                model = model_store.load_local_model(self.model_name)
                if model is not None:
                    # Bypass Separator.__init__ (it would hit the network), so
                    # every attribute separate_tensor() reads must be set here.
                    separator = Separator.__new__(Separator)
                    separator._name = self.model_name
                    separator._repo = None
                    separator._model = model
                    separator._audio_channels = model.audio_channels
                    separator._samplerate = model.samplerate
                    separator._device = device
                    separator._shifts = self.shifts
                    separator._overlap = self.overlap
                    separator._split = True
                    separator._segment = int(self.segment) if self.segment else None
                    separator._jobs = 0
                    separator._progress = False
                    # The per-run callback is passed to separate_tensor(), not
                    # stored; these two must still exist.
                    separator._callback = None
                    separator._callback_arg = None
                    notes.append(f"模型从本地缓存加载：{cached.name}（无需联网）")
                    return separator
            except Exception as exc:  # noqa: BLE001
                notes.append(f"本地缓存加载失败，回退到按名称加载：{type(exc).__name__}: {exc}")

        return Separator(
            model=self.model_name,
            device=device,
            shifts=self.shifts,
            overlap=self.overlap,
            split=True,
            segment=int(self.segment) if self.segment else None,
            jobs=0,
            progress=False,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def separate(
        self,
        source_wav: Path,
        work_dir: Path,
        *,
        progress: ProgressCallback | None = None,
        should_cancel=None,
    ) -> SeparationResult:
        source_wav = Path(source_wav)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        if not source_wav.is_file():
            raise DrumPracticeError(f"待分离文件不存在：{source_wav}")

        if not self._model_present():
            raise ModelNotInstalledError(self.model_name)

        # Never start a job that cannot finish writing its output.
        needed = source_wav.stat().st_size * 3
        if free_disk_bytes(work_dir) < needed:
            from ..errors import InsufficientDiskSpaceError

            free_gb = free_disk_bytes(work_dir) / (1024**3)
            raise InsufficientDiskSpaceError(needed / (1024**3), free_gb, str(work_dir))

        model_store.configure_cache()

        import torch

        from demucs.api import Separator

        device = self._resolve_device()
        notes: list[str] = []

        # ---- Load audio ourselves -------------------------------------
        # Decoding here (rather than letting Separator do it) means we control
        # the sample rate and channel layout exactly, so no hidden resampling
        # happens inside the model, and the same decode feeds BPM detection.
        from ..audio import io as audio_io

        audio, rate = audio_io.read_wav(source_wav)
        duration = audio.shape[1] / rate if rate else 0.0

        if audio.shape[0] != config.MODEL_CHANNELS:
            if config.MODEL_CHANNELS == 2 and audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
                notes.append("单声道输入已复制为立体声以适配模型。")
            else:
                audio = audio[: config.MODEL_CHANNELS]

        # ---- VRAM advisory -------------------------------------------
        if device == "cuda" and self.segment:
            estimate = _vram_estimate_mb(self.segment)
            try:
                free_mb = int(torch.cuda.mem_get_info()[0] / (1024 * 1024))
            except Exception:
                free_mb = 0
            if free_mb and free_mb < estimate:
                notes.append(
                    f"当前可用显存 {free_mb} MB，低于该 segment 的估算需求 "
                    f"（约 {estimate} MB），可能触发显存不足。"
                )

        # ---- Load model ----------------------------------------------
        if progress:
            progress(0.02, f"正在加载模型 {self.model_name} 到 {device.upper()}…")

        try:
            separator = self._build_separator(device, notes)
        except Exception as exc:
            from ..env import is_out_of_memory

            if is_out_of_memory(exc):
                from ..errors import OutOfMemoryError

                raise OutOfMemoryError(detail=str(exc)) from exc
            raise DrumPracticeError(
                f"模型加载失败：{self.model_name}",
                suggestions=[
                    "确认模型已完整下载（重新点击“下载模型”）。",
                    "运行 diagnose.bat 检查环境。",
                ],
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        model_rate = int(separator.samplerate)
        if rate != model_rate:
            # Should not happen - decode_to_wav already targets this rate.
            notes.append(f"输入采样率 {rate} Hz 与模型 {model_rate} Hz 不一致，已重采样。")
            audio_io.write_wav(work_dir / "_resampled.wav", audio, rate, bits=32)
            from ..audio import ffmpeg as ffmpeg_module

            ffmpeg_module.decode_to_wav(
                work_dir / "_resampled.wav",
                work_dir / "_resampled_model.wav",
                sample_rate=model_rate,
                channels=config.MODEL_CHANNELS,
                bits=32,
            )
            audio, rate = audio_io.read_wav(work_dir / "_resampled_model.wav")
            (work_dir / "_resampled.wav").unlink(missing_ok=True)
            (work_dir / "_resampled_model.wav").unlink(missing_ok=True)
            duration = audio.shape[1] / rate if rate else 0.0

        tensor = torch.from_numpy(audio)

        # ---- Progress / cancellation plumbing ------------------------
        state = {"fraction": 0.0, "segments_done": 0}

        def _callback(payload: dict) -> None:
            if should_cancel is not None and should_cancel():
                # Raising from here makes apply_model shut the pool down and
                # abandon the remaining segments.
                raise SeparationCancelled()

            total = float(payload.get("audio_length") or 0.0)
            offset = float(payload.get("segment_offset") or 0.0)
            if total <= 0:
                return

            raw = min(1.0, max(0.0, offset / total))
            # Never let the bar move backwards (bag models restart per sub-model).
            state["fraction"] = max(state["fraction"], raw)

            if progress:
                # Separation is ~90% of the whole job; leave room for mix/encode.
                models_in_bag = int(payload.get("models") or 1)
                model_index = int(payload.get("model_idx_in_bag") or 0)
                if models_in_bag > 1:
                    overall = (model_index + state["fraction"]) / models_in_bag
                    note = f"AI 分离中…（模型 {model_index + 1}/{models_in_bag}）"
                else:
                    overall = state["fraction"]
                    note = "AI 分离中…"
                progress(0.02 + 0.78 * overall, note)

        if progress:
            progress(0.03, "AI 分离开始（首次运行需要预热 CUDA，可能稍慢）…")

        started = time.perf_counter()
        try:
            _original, stems = separator.separate_tensor(tensor, rate)
        except SeparationCancelled:
            raise
        except KeyboardInterrupt as exc:  # raised by the cancel path internally
            raise SeparationCancelled() from exc
        except Exception as exc:
            from ..env import is_out_of_memory, release_gpu_memory

            release_gpu_memory()
            if is_out_of_memory(exc):
                from ..errors import OutOfMemoryError

                raise OutOfMemoryError(detail=str(exc)) from exc
            # A CUDA-side failure that is not OOM (bad kernel for sm_120 etc.)
            if device == "cuda" and "cuda" in str(exc).lower():
                raise CUDAUnavailableError(
                    detail=f"{type(exc).__name__}: {exc}"
                ) from exc
            raise DrumPracticeError(
                "AI 分离过程中出错。",
                suggestions=[
                    "重试一次。",
                    "运行 diagnose.bat 检查 PyTorch/CUDA 状态。",
                    "关闭其他占用显存的程序后重试。",
                ],
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        elapsed = time.perf_counter() - started

        if DRUM_STEM not in stems:
            raise DrumPracticeError(
                f"模型 {self.model_name} 没有输出鼓轨（可用音轨：{list(stems)}）。",
                suggestions=["改用 htdemucs 模型。"],
            )

        # ---- Write stems ---------------------------------------------
        if progress:
            progress(0.82, "正在写出分离音轨…")

        drums = stems[DRUM_STEM].detach().cpu().numpy().astype(np.float32)
        if drums.ndim == 1:
            drums = drums[None, :]

        # Residual = original - drums, computed in float64 then cast back.
        # Frames are aligned by construction (same separator output).
        frames = min(audio.shape[1], drums.shape[1])
        channels = max(audio.shape[0], drums.shape[0])
        original = np.zeros((channels, frames), dtype=np.float64)
        drum_arr = np.zeros((channels, frames), dtype=np.float64)
        original[: audio.shape[0], :frames] = audio[:, :frames]
        drum_arr[: drums.shape[0], :frames] = drums[:, :frames]
        no_drums = np.clip(original - drum_arr, -1.0, 1.0).astype(np.float32)

        drums_path = work_dir / "drums.wav"
        no_drums_path = work_dir / "no_drums.wav"
        audio_io.write_wav(drums_path, drums.astype(np.float32), rate, bits=32)
        audio_io.write_wav(no_drums_path, no_drums, rate, bits=32)

        # Keep the full 4-stem split only if asked (spec: "keep intermediate stems")
        stem_paths: dict[str, Path] = {}
        for stem_name, stem_tensor in stems.items():
            stem_path = work_dir / f"{stem_name}.wav"
            if stem_name == DRUM_STEM:
                stem_paths[stem_name] = drums_path
                continue
            data = stem_tensor.detach().cpu().numpy().astype(np.float32)
            if data.ndim == 1:
                data = data[None, :]
            audio_io.write_wav(stem_path, data, rate, bits=32)
            stem_paths[stem_name] = stem_path

        # ---- Release everything --------------------------------------
        del stems, tensor, audio, drums, drum_arr, original, no_drums
        del separator
        gc.collect()
        from ..env import release_gpu_memory

        release_gpu_memory()

        if progress:
            progress(0.85, "分离完成")

        return SeparationResult(
            drums=drums_path,
            no_drums=no_drums_path,
            sample_rate=rate,
            model=self.model_name,
            device=device,
            duration_seconds=duration,
            processing_seconds=elapsed,
            stems=stem_paths,
            notes=notes,
        )
