"""Environment diagnosis (``diagnose.bat`` / ``python -m app.diagnose``).

Prints exactly what spec section 26 asks for - OS, Python, PyTorch, CUDA, GPU,
VRAM, FFmpeg, models - plus the concrete next step for anything that is wrong.

Exit code 0 = ready to run, 1 = something critical is missing.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import APP_NAME, __version__, config


def _line(char: str = "-", width: int = 70) -> str:
    return char * width


def collect() -> dict:
    """Structured diagnosis, also used by the ``--json`` flag."""
    from .audio import timestretch as stretch_module
    from .env import build_report
    from .separation import models as model_store

    report = build_report()
    data = report.to_dict()
    data["version"] = __version__
    data["models"] = [status.to_dict() for status in model_store.all_statuses()]
    data["model_cache_dir"] = str(model_store.models_root())
    data["model_cache_size_mb"] = round(model_store.cache_size_mb(), 1)
    data["stretch_backends"] = stretch_module.available_backends()
    data["output_dir"] = str(config.OUTPUT_DIR)
    data["temp_dir"] = str(config.TEMP_DIR)
    return data


def _check(label: str, ok: bool, detail: str = "") -> str:
    mark = "OK  " if ok else "FAIL"
    return f"  [{mark}] {label}{(' — ' + detail) if detail else ''}"


def render(data: dict) -> str:
    out: list[str] = []
    add = out.append

    add(_line("="))
    add(f"  {APP_NAME} v{__version__} — 环境诊断")
    add(_line("="))
    add("")

    # ---- System ------------------------------------------------------
    add("[系统]")
    add(_check("Windows", data["os_name"] == "Windows", f"{data['os_name']} {data['os_version']}"))
    add(_check("64 位", data["python_bits"] == 64, f"{data['machine']}"))
    add("")

    # ---- Python ------------------------------------------------------
    add("[Python]")
    py_ok = data["python_version"].startswith(("3.10", "3.11", "3.12"))
    add(_check("版本", py_ok, data["python_version"]))
    add(_check("虚拟环境", data["is_venv"], data["python_executable"]))
    add(_check("NumPy", bool(data["numpy_version"]), data["numpy_version"] or "未安装"))
    add("")

    # ---- PyTorch / CUDA ---------------------------------------------
    add("[PyTorch / CUDA]")
    add(_check("PyTorch 已安装", data["torch_installed"], data["torch_version"] or "未安装"))
    cuda_build = data.get("torch_cuda_build") or ""
    add(
        _check(
            "CUDA 版本",
            bool(cuda_build),
            f"{cuda_build}（版本号带 +cuXXX 才是 GPU 版）" if cuda_build else "CPU 版 PyTorch",
        )
    )
    add(_check("CUDA 可用", data["torch_cuda_available"], data.get("cuda_version") or ""))
    if data.get("cudnn_version"):
        add(f"          cuDNN: {data['cudnn_version']}")
    add("")

    # ---- GPU ---------------------------------------------------------
    add("[GPU]")
    gpu = data.get("gpu")
    if gpu:
        add(f"          {gpu['name']}")
        add(f"          显存: {gpu['total_vram_mb']} MB 总量 / {gpu['free_vram_mb']} MB 可用")
        add(f"          计算能力: sm_{str(gpu['compute_capability']).replace('.', '')}")
        if gpu.get("is_blackwell"):
            add("          架构: Blackwell (RTX 50 系列)，需要 CUDA 12.8+ 的 PyTorch")
        device = "CUDA"
    else:
        add("          未检测到可用 CUDA GPU（将使用 CPU 模式，速度慢很多）")
        device = "CPU"
    add("")

    # ---- FFmpeg ------------------------------------------------------
    add("[FFmpeg]")
    ffmpeg = data.get("ffmpeg") or {}
    add(_check("已找到", bool(ffmpeg.get("found")), ffmpeg.get("version") or "未找到"))
    if ffmpeg.get("found"):
        add(f"          路径: {ffmpeg.get('path')}")
        add(f"          来源: {ffmpeg.get('source')}")
        add(f"          ffprobe: {ffmpeg.get('ffprobe') or '未找到'}")
        filters = ffmpeg.get("filters") or {}
        add(_check("atempo 变速滤镜", bool(filters.get("atempo"))))
        add(_check("loudnorm 响度滤镜", bool(filters.get("loudnorm"))))
        add(_check("alimiter 限幅滤镜", bool(filters.get("alimiter"))))
    add("")

    # ---- AI runtime --------------------------------------------------
    add("[AI 运行时]")
    add(_check("Demucs", data["demucs_installed"], data["demucs_version"] or "未安装"))
    add(_check("sphn（音频读取）", data["sphn_installed"]))
    backends = data.get("stretch_backends") or {}
    add(
        _check(
            "变速后端",
            bool(backends.get("atempo") or backends.get("rubberband")),
            f"atempo={backends.get('atempo')}, rubberband={backends.get('rubberband')}",
        )
    )
    add("")

    # ---- Models ------------------------------------------------------
    add("[模型]")
    add(f"          缓存目录: {data['model_cache_dir']}")
    add(f"          占用: {data['model_cache_size_mb']} MB")
    for model in data.get("models", []):
        add(
            _check(
                f"{model['signature']} ({model['display_name']})",
                model["installed"],
                f"约 {model['size_mb']} MB"
                + (f"，已缓存 {model['present_files']} 个权重文件" if model["installed"] else "，尚未下载"),
            )
        )
    add("")

    # ---- Storage -----------------------------------------------------
    add("[存储]")
    add(_check("磁盘可用空间", data["free_disk_gb"] > 2.0, f"{data['free_disk_gb']:.1f} GB"))
    add(f"          输出目录: {data['output_dir']}")
    add(f"          临时目录: {data['temp_dir']}")
    add("")

    # ---- Verdict -----------------------------------------------------
    add(_line("="))
    critical: list[str] = []
    if not ffmpeg.get("found"):
        critical.append("FFmpeg 未安装 → 运行 install.bat")
    if not data["torch_installed"]:
        critical.append("PyTorch 未安装 → 运行 install.bat")
    if data["torch_installed"] and not cuda_build:
        critical.append("安装的是 CPU 版 PyTorch → 重新运行 install.bat")
    if not data["demucs_installed"]:
        critical.append("Demucs 未安装 → 运行 install.bat")
    if not any(m["installed"] for m in data.get("models", [])):
        critical.append("尚未下载任何模型 → 在网页界面点击“下载模型”")

    if critical:
        add("  结论: 尚未就绪")
        for item in critical:
            add(f"    * {item}")
    else:
        add(f"  结论: 就绪 — AI 分离将使用 {device} 模式")

    if data.get("problems"):
        add("")
        add("  检测到的问题:")
        for problem in data["problems"]:
            add(f"    * {problem}")
    if data.get("warnings"):
        add("")
        add("  提示:")
        for warning in data["warnings"]:
            add(f"    * {warning}")

    if device == "CPU" and data["torch_installed"]:
        add("")
        add("  CUDA GPU unavailable. The program will fall back to CPU mode.")
        add("  CUDA GPU 不可用，程序将回退到 CPU 模式。")

    add(_line("="))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="环境诊断工具")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    # Make sure Chinese output does not explode on a CP936 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        data = collect()
    except Exception as exc:  # pragma: no cover - diagnosis must not crash
        print(f"诊断过程出错：{type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render(data))

    ffmpeg_ok = bool((data.get("ffmpeg") or {}).get("found"))
    ready = bool(
        data["torch_installed"]
        and ffmpeg_ok
        and data["demucs_installed"]
        and (data.get("torch_cuda_build") or data.get("torch_cuda_available"))
    )
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
