"""Real end-to-end test: genuine Demucs GPU inference through the full pipeline.

Not part of the offline self-test (it needs the model and a GPU).  Run with:

    .venv\\Scripts\\python.exe e2e_gpu_test.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.audio import io as audio_io  # noqa: E402
from app.separation import models as model_store  # noqa: E402

RATE = 44100
DURATION = 8.0


def build_song() -> np.ndarray:
    """A small but realistic mix: bass line + chords + drums."""
    frames = int(RATE * DURATION)
    t = np.arange(frames, dtype=np.float64) / RATE

    # Bass: 80 Hz-ish with a little movement
    bass = 0.20 * np.sin(2 * np.pi * 82.4 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    # Chords: a few partials
    chords = sum(
        0.07 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6, 440.0)
    )

    # Drums: kick on beats, snare on 2 & 4, hats on 8ths at 120 BPM
    drums = np.zeros(frames, dtype=np.float64)
    rng = np.random.default_rng(11)
    beat = 60.0 / 120.0

    def place(sample: np.ndarray, start: int) -> None:
        end = min(frames, start + sample.size)
        if end > start:
            drums[start:end] += sample[: end - start]

    klen = int(RATE * 0.13)
    kt = np.arange(klen) / RATE
    kick = (np.sin(2 * np.pi * (52 + 70 * np.exp(-kt / 0.02)) * kt)
            * np.exp(-kt / 0.06) * 0.85)
    slen = int(RATE * 0.10)
    st = np.arange(slen) / RATE
    snare = ((rng.standard_normal(slen) * 0.6 + np.sin(2 * np.pi * 185 * st) * 0.4)
             * np.exp(-st / 0.03) * 0.55)
    hlen = int(RATE * 0.03)
    ht = np.arange(hlen) / RATE
    hat = rng.standard_normal(hlen) * np.exp(-ht / 0.007)

    for index in range(int(DURATION / beat)):
        start = int(index * beat * RATE)
        if index % 4 in (0, 2):
            place(kick, start)
        if index % 4 in (1, 3):
            place(snare, start)
        place(hat * (0.26 if index % 2 == 0 else 0.12), start)
        place(hat * 0.09, start + int(beat * RATE / 2))

    mono = bass + chords + drums
    peak = float(np.max(np.abs(mono)))
    if peak > 0.85:
        mono *= 0.85 / peak
    return np.stack([mono, mono]).astype(np.float32)


def main() -> int:
    print("=" * 72)
    print("  真实端到端测试：Demucs GPU 推理 → 混音 → 变速 → 导出")
    print("=" * 72)

    import torch

    print(f"torch        : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    model_store.configure_cache()
    checkpoint = model_store.local_checkpoint("htdemucs")
    print(f"本地模型缓存 : {checkpoint}")
    if checkpoint is None:
        print("\n未找到本地模型缓存，请先在界面点击“下载模型”。")
        return 1

    work = PROJECT_ROOT / ".e2e-tmp"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    config.OUTPUT_DIR = work / "output"
    config.TEMP_DIR = work / "temp"
    config.TEMP_SEPARATION_DIR = config.TEMP_DIR / "separation"
    config.TEMP_MIX_DIR = config.TEMP_DIR / "mix"
    config.TEMP_UPLOAD_DIR = config.TEMP_DIR / "uploads"
    config.LOGS_DIR = work / "logs"
    config.ensure_directories()

    song = build_song()
    source = work / "song.wav"
    audio_io.write_wav(source, song, RATE, bits=16)
    print(f"\n输入         : {source.name}  {source.stat().st_size / 1024:.0f} KB  "
          f"{DURATION:.1f}s")

    from app.pipeline import JobConfig, run_job

    events: list[dict] = []
    cfg = JobConfig(
        source=source,
        drum_volume=0.10,
        speed=0.8,
        output_format="wav",
        output_bits=16,
        model="htdemucs",
        device="cuda",
        detect_bpm=True,
        metronome_enabled=True,
        metronome_volume=0.3,
        original_bpm=120,
    )

    print("\n--- running pipeline (real Demucs, CUDA) ---")
    started = time.time()
    result = run_job(cfg, progress=events.append, job_id="e2e")
    elapsed = time.time() - started

    print("\n--- results ---")
    print(f"输出文件     : {result.output.name}")
    print(f"文件大小     : {result.output.stat().st_size / 1024:.0f} KB")
    print(f"模型         : {result.model}")
    print(f"设备         : {result.device}")
    print(f"检测 BPM     : {result.detected_bpm}")
    print(f"目标 BPM     : {result.target_bpm}")
    print(f"时长         : {result.duration_seconds:.2f}s "
          f"(期望 ~{DURATION / 0.8:.2f}s)")
    print(f"变速后端     : {result.stretch_backend}")
    print(f"节拍器敲击   : {result.metronome_clicks}")
    print(f"处理耗时     : {elapsed:.1f}s")
    print(f"进度回调次数 : {len(events)}")

    out, _ = audio_io.read_wav(result.output)
    from app.audio import loudness as L

    info = __import__("app.audio.ffmpeg", fromlist=["x"]).probe(result.output)
    print("\n--- output validation ---")
    print(f"采样率/声道  : {info.sample_rate} Hz / {info.channels}ch  codec={info.codec}")
    print(f"峰值         : {audio_io.peak(out):.4f}  (必须 <= 1.0)")
    print(f"有限值       : {bool(np.isfinite(out).all())}")
    print(f"响度         : {L.integrated_lufs(result.output)} LUFS")

    checks = [
        ("输出文件已生成", result.output.is_file()),
        ("样本有限（无 NaN/Inf）", bool(np.isfinite(out).all())),
        ("峰值未削波", audio_io.peak(out) <= 1.0),
        ("时长约为输入/0.8", abs(result.duration_seconds - DURATION / 0.8) < 0.4),
        ("使用了 GPU", result.device == "cuda"),
        ("检测到 BPM 或已说明", result.detected_bpm is not None or bool(result.notes)),
        ("进度回调被调用", len(events) > 10),
        ("有节拍器敲击", result.metronome_clicks > 0),
    ]
    print("\n--- checks ---")
    failed = 0
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
        if not ok:
            failed += 1

    print("\n--- notes ---")
    for note in result.notes:
        print(f"  - {note}")

    if torch.cuda.is_available():
        print(f"\nGPU 显存峰值 : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    shutil.rmtree(work, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f"  {'REAL GPU END-TO-END PASSED' if failed == 0 else f'{failed} CHECK(S) FAILED'}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
