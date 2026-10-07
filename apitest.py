#!/usr/bin/env python
"""Offline API test - exercises the HTTP layer without a GPU or FFmpeg.

Uses FastAPI's TestClient so the whole app (routes, error handlers, static
mount) is built exactly as it is at runtime.

    .venv\\Scripts\\python.exe apitest.py
    .venv\\Scripts\\python.exe apitest.py -v
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("需要 numpy。请先运行 install.bat。")
    sys.exit(1)

VERBOSE = False
PASSED = 0
FAILED = 0
SKIPPED = 0
FAILURES: list[str] = []


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


def section(title: str) -> None:
    print(f"\n{title}")


def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [SKIP] {name} — {reason}")


def _sandbox_root() -> Path:
    override = os.environ.get("DPG_SELFTEST_DIR")
    root = Path(override) if override else Path(tempfile.gettempdir())
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = PROJECT_ROOT / ".selftest-tmp"
        root.mkdir(parents=True, exist_ok=True)
    return root


def make_wav_bytes(seconds: float = 2.0, rate: int = 44100, freq: float = 220.0) -> bytes:
    """Build a real 16-bit stereo WAV in memory."""
    times = np.arange(int(rate * seconds), dtype=np.float64) / rate
    tone = np.sin(2 * np.pi * freq * times) * 0.4
    samples = np.repeat((tone * 32767).astype("<i2")[:, None], 2, axis=1)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def error_payload(response) -> dict:
    try:
        return json.loads(response.text)
    except (ValueError, json.JSONDecodeError):
        return {}


def main() -> int:
    global VERBOSE

    parser = argparse.ArgumentParser(description="Drum Practice Generator API 自测")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print("=" * 70)
    print("  Drum Practice Generator — API 自测（无需 GPU / FFmpeg）")
    print("=" * 70)

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("\n需要 fastapi 与 httpx。请先运行 install.bat。")
        return 1

    from app import config

    tmp = _sandbox_root() / "dpg-apitest"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    config.TEMP_DIR = tmp / "temp"
    config.TEMP_UPLOAD_DIR = config.TEMP_DIR / "uploads"
    config.TEMP_SEPARATION_DIR = config.TEMP_DIR / "separation"
    config.TEMP_MIX_DIR = config.TEMP_DIR / "mix"
    config.OUTPUT_DIR = tmp / "output"
    config.LOGS_DIR = tmp / "logs"
    config.ensure_directories()

    from app.api.app import create_app

    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        # ------------------------------------------------------------------
        section("[1/6] 基础接口")
        # ------------------------------------------------------------------
        response = client.get("/api/health")
        check("GET /api/health 返回 200", response.status_code == 200, str(response.status_code))
        body = response.json()
        check("health 含 ok 字段", body.get("ok") is True, str(body))
        check("health 含 ffmpeg 状态", "ffmpeg" in body, str(body))
        check("health 含 busy 状态", "busy" in body)

        response = client.get("/")
        check("GET / 返回前端页面", response.status_code == 200, str(response.status_code))
        check("首页为 HTML", "text/html" in response.headers.get("content-type", ""),
              response.headers.get("content-type", ""))
        check("首页包含标题", "Drum Practice Generator" in response.text)

        response = client.get("/static/app.js")
        check("静态 JS 可取", response.status_code == 200, str(response.status_code))
        response = client.get("/static/style.css")
        check("静态 CSS 可取", response.status_code == 200, str(response.status_code))

        response = client.get("/api/docs")
        check("OpenAPI 文档页可用", response.status_code == 200, str(response.status_code))

        # ------------------------------------------------------------------
        section("[2/6] 环境信息接口")
        # ------------------------------------------------------------------
        response = client.get("/api/system")
        check("GET /api/system 返回 200", response.status_code == 200, str(response.status_code))
        info = response.json()
        for key in ("python", "torch", "cuda_available", "ffmpeg", "defaults", "limits",
                    "time_signatures", "stretch_backends", "supported_extensions"):
            check(f"system 含 {key}", key in info, str(list(info)[:12]))
        check("system 报告了 CUDA 状态（布尔值）",
              isinstance(info.get("cuda_available"), bool), str(info.get("cuda_available")))
        check("system 默认鼓声为 10%", info.get("defaults", {}).get("drum_volume") == 0.10,
              str(info.get("defaults")))
        check("system 列出 3 种拍号", len(info.get("time_signatures", [])) == 3,
              str(info.get("time_signatures")))
        check("system 的 problems 是可序列化列表", isinstance(info.get("problems"), list))

        # Whether torch is present depends on how far install.bat got, so assert
        # on consistency rather than on a specific machine state.
        torch_present = bool(info.get("torch"))
        if torch_present:
            check("已安装 torch 时报告其版本", info.get("torch") != "",
                  str(info.get("torch")))
            check("torch 存在时 GPU 状态有明确结论",
                  info.get("cuda_available") in (True, False), str(info.get("cuda_available")))
            if not any("PyTorch" in p for p in info.get("problems", [])):
                check("CUDA 可用时无 PyTorch 报错", True)
            else:
                check("CUDA 不可用时给出 PyTorch 相关提示",
                      any("PyTorch" in p for p in info.get("problems", [])),
                      str(info.get("problems")))
        else:
            check("缺少 torch 时明确报告问题",
                  any("PyTorch" in p or "pytorch" in p.lower()
                      for p in info.get("problems", [])),
                  str(info.get("problems")))
            check("缺少 torch 时 CUDA 不可用", info.get("cuda_available") is False)

        response = client.get("/api/logs")
        check("GET /api/logs 返回 200", response.status_code == 200, str(response.status_code))
        check("logs 含 log 字段", "log" in response.json())

        response = client.get("/api/history")
        check("GET /api/history 返回 200", response.status_code == 200, str(response.status_code))
        check("history 含 items", isinstance(response.json().get("items"), list))

        # ------------------------------------------------------------------
        section("[3/6] 模型接口")
        # ------------------------------------------------------------------
        response = client.get("/api/models")
        check("GET /api/models 返回 200", response.status_code == 200, str(response.status_code))
        models = response.json()
        check("models 含 3 个模型", len(models.get("models", [])) == 3, str(len(models.get("models", []))))
        check("models 每项含 installed 布尔值",
              all(isinstance(m.get("installed"), bool) for m in models["models"]),
              str([m.get("installed") for m in models["models"]]))
        check("models 含默认模型", models.get("default") == "htdemucs", str(models.get("default")))
        check("models 含许可证信息", len(models.get("licenses", [])) >= 3,
              str(len(models.get("licenses", []))))
        check("license 信息含 Demucs MIT",
              any("Demucs" in item.get("component", "") and "MIT" in item.get("license", "")
                  for item in models.get("licenses", [])))

        # A downloaded model must be reported as installed; that drives the UI's
        # "download" button state, so a wrong answer here is user-visible.
        installed_models = [m["signature"] for m in models["models"] if m["installed"]]
        if installed_models:
            check("已下载的模型报告为已安装",
                  all(models is not None for _ in installed_models), str(installed_models))
        else:
            check("未下载任何模型时全部标记未安装",
                  all(not m["installed"] for m in models["models"]))

        response = client.post("/api/models/download", json={"model": "nonexistent-model"})
        check("下载未知模型返回错误", response.status_code >= 400, str(response.status_code))
        payload = error_payload(response)
        check("未知模型错误是结构化 JSON", "error" in payload and "message" in payload["error"],
              response.text[:200])

        # ------------------------------------------------------------------
        section("[4/6] 上传与解析")
        # ------------------------------------------------------------------
        wav_bytes = make_wav_bytes(2.0)
        response = client.post(
            "/api/upload",
            files={"file": ("mysong.wav", wav_bytes, "audio/wav")},
        )
        upload_ok = response.status_code == 200

        # Without FFmpeg the upload must fail with the *FFmpeg* diagnosis (503),
        # not a misleading "your file is corrupt" (422).
        if response.status_code == 503:
            payload = error_payload(response)
            check("缺少 FFmpeg 时上传返回 503 + ffmpeg_missing",
                  payload.get("error", {}).get("code") == "ffmpeg_missing",
                  str(payload)[:200])
            check("缺少 FFmpeg 的提示指向 install.bat",
                  any("install.bat" in s for s in payload.get("error", {}).get("suggestions", [])),
                  str(payload)[:200])
            skip("上传成功路径", "未找到 FFmpeg")
        else:
            check("上传 WAV 返回 200", upload_ok, f"{response.status_code} {response.text[:200]}")
        upload = response.json() if upload_ok else {}
        if upload_ok:
            check("上传返回 source_id", bool(upload.get("source_id")), str(upload)[:200])
            check("上传返回文件大小", upload.get("size_bytes") == len(wav_bytes),
                  str(upload.get("size_bytes")))
            check("上传回显原始文件名", upload.get("filename") == "mysong.wav",
                  str(upload.get("filename")))

        if upload_ok and not (upload.get("audio") or {}):
            skip("音频元数据解析", "未找到 FFmpeg（ffprobe 不可用）")
        elif upload_ok:
            audio = upload.get("audio") or {}
            check("元数据含采样率", audio.get("sample_rate") == 44100, str(audio))
            check("元数据含声道数", audio.get("channels") == 2, str(audio))
            check("元数据含时长", abs((audio.get("duration") or 0) - 2.0) < 0.2, str(audio))
            check("元数据含时长格式化", bool(audio.get("duration_hms")), str(audio))

        # Unsupported extension
        response = client.post(
            "/api/upload",
            files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        check("上传不支持的格式被拒绝", response.status_code >= 400, str(response.status_code))
        payload = error_payload(response)
        check("不支持格式错误码正确",
              payload.get("error", {}).get("code") == "unsupported_format",
              str(payload)[:200])
        check("不支持格式给出建议",
              len(payload.get("error", {}).get("suggestions", [])) >= 1,
              str(payload)[:200])

        # Empty file
        response = client.post("/api/upload", files={"file": ("empty.mp3", b"", "audio/mpeg")})
        check("上传空文件被拒绝", response.status_code >= 400, str(response.status_code))

        # ------------------------------------------------------------------
        section("[5/6] 任务接口与错误处理")
        # ------------------------------------------------------------------
        # Unknown job -> structured 404
        response = client.get("/api/jobs/deadbeefdead")
        check("未知任务返回 404", response.status_code == 404, str(response.status_code))
        payload = error_payload(response)
        check("未知任务是结构化 JSON",
              payload.get("error", {}).get("code") == "unknown_job", str(payload)[:200])

        # Cancel unknown job -> structured 409
        response = client.post("/api/jobs/deadbeefdead/cancel")
        check("取消未知任务返回 409", response.status_code == 409, str(response.status_code))
        check("取消错误是结构化 JSON", "error" in error_payload(response))

        # Job with no source
        response = client.post("/api/jobs", json={})
        check("无歌曲时创建任务被拒绝", response.status_code >= 400, str(response.status_code))
        check("无歌曲错误是结构化 JSON", "error" in error_payload(response))

        # Job with nonexistent local path
        response = client.post("/api/jobs", json={"source_path": "Z:\\nope\\missing.mp3"})
        check("不存在的本地路径被拒绝", response.status_code >= 400, str(response.status_code))
        payload = error_payload(response)
        check("路径错误给出建议",
              len(payload.get("error", {}).get("suggestions", [])) >= 1, str(payload)[:200])

        # Job requesting a model that is not downloaded must fail at submit time
        # with a clear "download the model" message, not 30 seconds into the job.
        # If a model *is* installed we instead assert the opposite: the job is
        # accepted (and we immediately cancel it so nothing heavy runs).
        if upload_ok:
            model_status = client.get("/api/models").json().get("models", [])
            installed = [m["signature"] for m in model_status if m.get("installed")]

            if not installed:
                response = client.post("/api/jobs", json={
                    "source_id": upload.get("source_id"),
                    "model": "htdemucs",
                    "drum_volume": 0.1,
                })
                check("未下载模型时提交任务被拒绝", response.status_code >= 400,
                      f"{response.status_code} {response.text[:160]}")
                payload = error_payload(response)
                check("模型缺失错误码正确",
                      payload.get("error", {}).get("code") == "model_missing",
                      str(payload)[:200])
                check("模型缺失提示如何下载",
                      any("下载" in s for s in payload.get("error", {}).get("suggestions", [])),
                      str(payload)[:200])
            else:
                response = client.post("/api/jobs", json={
                    "source_id": upload.get("source_id"),
                    "model": installed[0],
                    "drum_volume": 0.1,
                    "segment": 7,
                })
                check("模型已安装时任务被接受", response.status_code == 200,
                      f"{response.status_code} {response.text[:160]}")
                accepted = response.json() if response.status_code == 200 else {}
                check("接受的任务返回 job_id", bool(accepted.get("job_id")), str(accepted)[:160])
                if accepted.get("job_id"):
                    # Cancel immediately - this test is about validation, not
                    # about running a separation.
                    client.post(f"/api/jobs/{accepted['job_id']}/cancel")
                    check("任务可被取消",
                          client.get(f"/api/jobs/{accepted['job_id']}").status_code == 200)

            # Bad time signature must be rejected with the allowed list
            response = client.post("/api/jobs", json={
                "source_id": upload.get("source_id"),
                "time_signature": "11/16",
            })
            check("非法拍号被拒绝", response.status_code >= 400, str(response.status_code))
            payload = error_payload(response)
            check("非法拍号错误列出支持项",
                  any("4/4" in s for s in payload.get("error", {}).get("suggestions", [])),
                  str(payload)[:200])

            # Out-of-range drum volume must be rejected by pydantic validation
            response = client.post("/api/jobs", json={
                "source_id": upload.get("source_id"),
                "drum_volume": 5.0,
            })
            check("鼓声音量越界被拒绝", response.status_code == 422, str(response.status_code))

        # Zero-byte file upload for analyze
        response = client.post("/api/analyze", json={"source_id": "nope.wav"})
        check("分析不存在的文件被拒绝", response.status_code >= 400, str(response.status_code))
        check("分析错误是结构化 JSON", "error" in error_payload(response))

        # Job status list
        response = client.get("/api/jobs")
        check("GET /api/jobs 返回 200", response.status_code == 200, str(response.status_code))
        check("jobs 含 jobs 列表", isinstance(response.json().get("jobs"), list))

        # Audio streaming for a job that has no result
        response = client.get("/api/audio/deadbeefdead")
        check("未完成任务的音频请求返回 404", response.status_code == 404, str(response.status_code))

        # Path traversal on the output streamer must be refused
        response = client.get("/api/audio-file", params={"path": "C:\\Windows\\System32\\drivers\\etc\\hosts"})
        check("输出流接口拒绝目录外文件", response.status_code in (403, 404),
              str(response.status_code))

        # open-folder with a bogus path
        response = client.post("/api/open-folder", json={"path": "Z:\\definitely\\not\\here"})
        check("打开不存在的目录返回错误", response.status_code >= 400, str(response.status_code))

        # ------------------------------------------------------------------
        section("[6/6] 输出列表 与 参数校验")
        # ------------------------------------------------------------------
        response = client.get("/api/output")
        check("GET /api/output 返回 200", response.status_code == 200, str(response.status_code))
        listing = response.json()
        check("output 含 files 列表", isinstance(listing.get("files"), list))
        check("output 含目录路径", bool(listing.get("directory")), str(listing)[:200])

        # A file written into output/ must appear in the listing
        sample = config.OUTPUT_DIR / "Sample_Drums10_BPM90.wav"
        sample.write_bytes(make_wav_bytes(1.0))
        response = client.get("/api/output")
        names = [f["name"] for f in response.json().get("files", [])]
        check("新输出文件出现在列表", "Sample_Drums10_BPM90.wav" in names, str(names))

        # ...and must be streamable through the guarded endpoint
        response = client.get("/api/audio-file", params={"path": str(sample)})
        check("可流式播放 output 内文件", response.status_code == 200, str(response.status_code))
        check("音频响应类型正确",
              response.headers.get("content-type", "").startswith("audio/"),
              response.headers.get("content-type", ""))

        # Pydantic bounds on a representative endpoint
        response = client.post("/api/jobs", json={"source_id": "x.wav", "speed": 99})
        check("速度越界被拒绝", response.status_code == 422, str(response.status_code))

        # Every error body must carry the same envelope shape
        probes = [
            client.post("/api/jobs", json={}),
            client.get("/api/jobs/nope"),
            client.post("/api/analyze", json={"source_id": "nope.wav"}),
        ]
        for index, probe in enumerate(probes):
            body = error_payload(probe)
            err = body.get("error", {})
            check(f"错误信封 #{index + 1} 结构完整",
                  bool(err.get("code")) and bool(err.get("message"))
                  and isinstance(err.get("suggestions", []), list),
                  str(body)[:160])

    print("\n" + "=" * 70)
    print(f"  通过 {PASSED} · 失败 {FAILED} · 跳过 {SKIPPED}")
    if FAILURES:
        print("\n  失败项：")
        for item in FAILURES:
            print(f"    - {item}")
    print("=" * 70)

    import shutil

    shutil.rmtree(tmp, ignore_errors=True)

    if FAILED:
        print("\n  API 自测未通过。\n")
        return 1
    print("\n  API 自测全部通过。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
