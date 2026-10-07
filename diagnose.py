#!/usr/bin/env python
"""Environment diagnostic tool (spec section 26).

Usage:

    python diagnose.py
    python diagnose.py --json
    diagnose.bat

Prints OS, Python, PyTorch, CUDA, GPU, VRAM, FFmpeg and model status, plus the
concrete next step for anything missing.  Exit code 0 = ready to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (double-click / `python diagnose.py`) by
# making sure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        from app.diagnose import main as diagnose_main
    except ImportError as exc:
        print("无法导入 app 包，请确认在项目根目录运行本脚本。")
        print(f"错误：{exc}")
        print(f"\n当前目录: {Path.cwd()}")
        print(f"脚本目录: {PROJECT_ROOT}")
        return 1

    return diagnose_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
