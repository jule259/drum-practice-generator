"""Drum Practice Generator - entry point.

Run (normally via start.bat, which activates the venv first):

    python -m app.main
    python -m app.main --cpu --port 8766

The server binds to 127.0.0.1 only.  This is a local tool that reads the user's
own music files; it must never be reachable from the network.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser

from . import APP_NAME, __version__, config


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so Chinese output is never garbled or fatal.

    The launcher sets ``PYTHONIOENCODING``, but running ``python -m app.main``
    directly bypasses that.  Without this, Python encodes with the ANSI code page
    (GBK on Chinese Windows) and the banner arrives as mojibake; under a legacy
    code page it can also raise UnicodeEncodeError mid-startup.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            continue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drum-practice-generator",
        description=f"{APP_NAME} - 本地 AI 电子鼓练习曲生成器",
    )
    parser.add_argument("--host", default=config.HOST, help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=config.PORT, help="监听端口")
    parser.add_argument(
        "--cpu", action="store_true", help="强制使用 CPU（调试用，速度很慢）"
    )
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument(
        "--diagnose", action="store_true", help="打印环境报告后退出（等同 diagnose.bat）"
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = parse_args(argv)

    if args.diagnose:
        from .diagnose import main as diagnose_main

        return diagnose_main([])

    if args.cpu:
        import os

        os.environ["DPG_DEVICE"] = "cpu"

    config.ensure_directories()

    from .env import build_report
    from .logging_setup import banner, get_logger

    logger = get_logger("main")
    report = build_report()
    banner(report)

    # Fail fast with an actionable message rather than 500ing on first upload.
    if not report.ffmpeg.get("found"):
        logger.error("FFmpeg 未找到。请先运行 install.bat。")
        print("\n" + "=" * 68)
        print("  启动中止：未找到 FFmpeg")
        print("  请先双击运行 install.bat，它会自动下载 FFmpeg 到 tools\\ffmpeg。")
        print("=" * 68 + "\n")
        return 2

    if not report.torch_installed:
        logger.error("PyTorch 未安装。AI 分离功能不可用。")
        print("\n" + "=" * 68)
        print("  警告：未安装 PyTorch，AI 分离功能将不可用。")
        print("  请先双击运行 install.bat。")
        print("=" * 68 + "\n")

    if not report.torch_cuda_available and report.torch_installed:
        logger.warning(
            "CUDA GPU unavailable. The program will fall back to CPU mode."
        )
        print("\n" + "-" * 68)
        print("  CUDA GPU unavailable. The program will fall back to CPU mode.")
        print("  CUDA GPU 不可用，将回退到 CPU 模式，处理时间会显著变长。")
        print("  运行 diagnose.bat 查看详细原因。")
        print("-" * 68 + "\n")

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  {APP_NAME} v{__version__}")
    print(f"  界面地址: {url}")
    print(f"  输出目录: {config.OUTPUT_DIR}")
    print("  按 Ctrl+C 停止服务\n")

    if not args.no_browser and config.OPEN_BROWSER:
        def _open() -> None:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    from .api.app import create_app

    application = create_app()
    try:
        uvicorn.run(
            application,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
