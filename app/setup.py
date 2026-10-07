"""Installer backend, invoked by install.bat.

Why the real work lives here rather than in batch:

* A ``.bat`` file is decoded by ``cmd.exe`` using the OEM code page, so UTF-8
  Chinese text inside it gets garbled and can corrupt the script.  Python lets us
  control the encoding exactly.
* Long pip/HTTP operations need real error handling, per-step verification and a
  log file - all painful in batch and straightforward here.

``install.bat`` stays ASCII-only and just bootstraps a Python interpreter.

Usage:
    python app/setup.py --project-dir <dir> [--skip-torch] [--cpu-only-torch]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Console output.  Windows consoles are not always UTF-8, and this script's job
# includes telling the user what went wrong, so never die on an encoding error.
# --------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

TORCH_VERSION = "2.9.1"

# Two CUDA builds are supported.  Both cover Blackwell / sm_120 (RTX 50 series):
#
#   cu128  -- Primary.  CUDA 12.8 runtime runs on any driver >= 12.8, including
#             the 13.x driver on the reference machine.  Crucially, this is the
#             build that *domestic mirrors actually carry*, and mirror throughput
#             measured 4.95-17.85 MB/s versus 1.87-3.41 MB/s from
#             download.pytorch.org (a ~5x difference on a ~2.9 GB download).
#   cu130  -- Matches a CUDA 13.x driver's major version, used when the official
#             index is chosen explicitly.
TORCH_VARIANTS = {
    "cu128": {
        "official": "https://download.pytorch.org/whl/cu128",
        "mirror": "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128",
        "label": "CUDA 12.8",
    },
    "cu130": {
        "official": "https://download.pytorch.org/whl/cu130",
        "mirror": None,  # no domestic mirror carries cu130 yet
        "label": "CUDA 13.0",
    },
}
DEFAULT_VARIANT = "cu128"

# PyPI mirrors.  TUNA measured 2.74 MB/s against a very slow default route.
PYPI_MIRRORS = {
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "official": None,
}
DEFAULT_PYPI_MIRROR = "tuna"

_STEPS = 6
_log_handle = None


def log(message: str = "") -> None:
    """Print to the console and append to logs/install.log."""
    print(message, flush=True)
    if _log_handle is not None:
        try:
            _log_handle.write(message + "\n")
            _log_handle.flush()
        except OSError:
            pass


def section(step: int, title: str) -> None:
    log()
    log("=" * 75)
    log(f"  [{step}/{_STEPS}] {title}")
    log("=" * 75)


def run(args: list[str], *, timeout: float | None = None) -> int:
    """Run a subprocess, streaming output straight to our console and log."""
    log(f"  $ {' '.join(args)}")
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        log(f"  [错误] 找不到程序：{args[0]}")
        return 127
    except subprocess.TimeoutExpired:
        log(f"  [错误] 命令超时：{' '.join(args[:3])}")
        return 124

    for line in (proc.stdout or "").splitlines():
        log("    " + line)
    return proc.returncode


def probe_speed_mbps(url: str, *, seconds: float = 6.0, max_bytes: int = 8 * 1024 * 1024) -> float:
    """Download a small ranged chunk and return MB/s.  0.0 means unreachable.

    Used to pick between the official PyTorch index and a domestic mirror: the
    difference is minutes versus hours on a ~2.9 GB wheel, so measuring beats
    guessing.  Never raises - a failure just returns 0.0.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "Mozilla/5.0"}
    )
    started = time.time()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            while True:
                chunk = response.read(262144)
                if not chunk:
                    break
                total += len(chunk)
                if total >= max_bytes or (time.time() - started) >= seconds:
                    break
    except (urllib.error.URLError, OSError, ValueError):
        return 0.0

    elapsed = max(time.time() - started, 0.001)
    if total <= 0:
        return 0.0
    return (total / (1024 * 1024)) / elapsed


def wheel_tag() -> str:
    """PEP 425 interpreter tag for the running Python, e.g. ``cp311``."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def torch_wheel_url(index_url: str, variant: str, *, tag: str | None = None) -> str:
    """Best-effort direct URL of the torch wheel on a CUDA index.

    Only used for speed probing, so a wrong guess is harmless: the probe returns
    0.0 and the caller falls back to reachability-agnostic behaviour.
    """
    tag = tag or wheel_tag()
    filename = f"torch-{TORCH_VERSION}%2B{variant}-{tag}-{tag}-win_amd64.whl"
    return f"{index_url.rstrip('/')}/{filename}"


def choose_torch_source(variant: str, requested: str) -> tuple[str, str]:
    """Decide between the official index and the mirror.

    ``requested`` is ``auto`` (measure and pick), ``mirror`` or ``official``.
    Returns ``(index_url, description)``.
    """
    info = TORCH_VARIANTS[variant]
    official = info["official"]
    mirror = info["mirror"]

    if requested == "official":
        return official, "官方源（按要求）"
    if requested == "mirror":
        if not mirror:
            log(f"  [提示] {variant} 没有可用的国内镜像，改用官方源。")
            return official, "官方源（无镜像）"
        return mirror, "国内镜像（按要求）"
    if not mirror:
        # e.g. cu130: no mirror carries it.
        if requested == "auto":
            log(f"  [提示] {variant} 暂无国内镜像，使用官方源。")
        return official, "官方源（该版本无镜像）"

    # auto: measure both and take the faster one.
    log("  正在测速：官方源 vs 国内镜像（各约 6 秒）...")
    official_speed = probe_speed_mbps(torch_wheel_url(official, variant))
    mirror_speed = probe_speed_mbps(torch_wheel_url(mirror, variant))

    log(f"    官方源  : {official_speed:6.2f} MB/s")
    log(f"    国内镜像: {mirror_speed:6.2f} MB/s")

    if mirror_speed > official_speed and mirror_speed > 0.3:
        return mirror, f"国内镜像（实测 {mirror_speed:.1f} MB/s）"
    if official_speed > 0:
        return official, f"官方源（实测 {official_speed:.1f} MB/s）"
    if mirror_speed > 0:
        return mirror, f"国内镜像（实测 {mirror_speed:.1f} MB/s）"
    # Both probes failed (offline, blocked, or an unexpected wheel name).  Prefer
    # the mirror, and let pip fall back to the official index on its own.
    log("  [提示] 测速均未成功；优先尝试国内镜像，失败会自动回退官方源。")
    return mirror, "国内镜像（测速未成功，优先尝试）"


def human_error(step: str, detail: str, fixes: list[str]) -> None:
    log()
    log("!" * 75)
    log(f"  [失败] {step}")
    log("!" * 75)
    if detail:
        log(f"  原因：{detail}")
    log("  建议：")
    for index, fix in enumerate(fixes, 1):
        log(f"    {index}. {fix}")
    log("!" * 75)


# ==========================================================================
def detect_venv_python(project_dir: Path) -> Path | None:
    candidate = project_dir / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else None


def step_venv(project_dir: Path, python_exe: str) -> Path | None:
    section(1, "创建虚拟环境 (.venv)")
    venv_python = detect_venv_python(project_dir)
    if venv_python is not None:
        log(f"  已存在，跳过创建：{venv_python}")
    else:
        code = run([python_exe, "-m", "venv", str(project_dir / ".venv")])
        if code != 0:
            human_error(
                "创建虚拟环境失败",
                f"python -m venv 返回 {code}",
                [
                    "确认 Python 安装完整（需要 venv 模块）。",
                    "如果使用 Microsoft Store 版 Python，建议改装 python.org 版本。",
                    "删除 .venv 目录后重试。",
                ],
            )
            return None
        venv_python = detect_venv_python(project_dir)
        if venv_python is None:
            human_error("创建虚拟环境失败", "未生成 .venv\\Scripts\\python.exe", [])
            return None
        log(f"  创建完成：{venv_python}")

    log("  升级 pip / setuptools / wheel ...")
    run([str(venv_python), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel", "--quiet", "--disable-pip-version-check"])
    return venv_python


def build_torch_pip_args(
    venv_python: Path, *, variant: str, source: str, pypi_mirror: str,
) -> tuple[list[str], str, str]:
    """Build the pip argv for the PyTorch install.

    Pure function (no I/O, no network) so the argument construction can be unit
    tested - a bug here previously crashed the installer with a NameError before
    pip ever ran.

    Returns ``(args, index_url, description)``.
    """
    index_url, description = choose_torch_source(variant, source)
    official = TORCH_VARIANTS[variant]["official"]

    args = [
        str(venv_python), "-m", "pip", "install",
        f"torch=={TORCH_VERSION}",
        "--index-url", index_url,
        "--disable-pip-version-check",
        # A 2.9 GB wheel over a flaky route needs generous retries; pip resumes
        # partial downloads, so an interrupted run is not wasted work.
        "--retries", "10",
        "--timeout", "120",
    ]

    # Torch's own dependencies (filelock, sympy, jinja2, nvidia-* ...) come from
    # PyPI, which may be slow or unreachable on the default route.
    mirror_url = PYPI_MIRRORS.get(pypi_mirror)
    if mirror_url:
        args += ["--extra-index-url", mirror_url]

    # When the CUDA index is not the primary source, add it as a fallback so a
    # mirror that lacks this exact wheel does not dead-end the install.
    if index_url != official:
        args += ["--extra-index-url", official]

    return args, index_url, description


def step_torch(venv_python: Path, skip: bool, *, variant: str, source: str,
               pypi_mirror: str) -> bool:
    section(2, f"安装 PyTorch {TORCH_VERSION} ({TORCH_VARIANTS[variant]['label']})")
    if skip:
        log("  --skip-torch 已指定，跳过。")
        return True

    log("  说明：必须安装 CUDA 版，否则无法使用 GPU 加速。")
    log("  这一步会下载约 2-3 GB —— 使用国内镜像通常只需几分钟。")
    log()

    args, index_url, description = build_torch_pip_args(
        venv_python, variant=variant, source=source, pypi_mirror=pypi_mirror
    )
    log(f"  下载源：{description}")
    log(f"          {index_url}")
    log()

    code = run(args, timeout=7200)
    if code != 0:
        other = "cu130" if variant == "cu128" else "cu128"
        human_error(
            "PyTorch 安装失败",
            f"pip 返回 {code}",
            [
                "检查网络连接；中断后重跑 install.bat 会断点续传已下载的部分。",
                "确认磁盘剩余空间大于 8 GB（CUDA 运行时依赖也占空间）。",
                f"换用另一个 CUDA 版本重试："
                f"install.bat --variant {other}",
                "若公司网络需要代理：先运行 set HTTPS_PROXY=http://主机:端口 再装。",
                "也可以强制走官方源：install.bat --source official",
            ],
        )
        return False

    # Verify it is actually the CUDA build, not silently the CPU wheel.
    log()
    log("  验证 PyTorch ...")
    code = run([
        str(venv_python), "-c",
        "import torch;"
        "print('torch', torch.__version__);"
        "print('cuda build', torch.version.cuda);"
        "print('cuda available', torch.cuda.is_available());"
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU');",
    ])
    if code != 0:
        human_error("PyTorch 导入失败", "安装后无法 import torch", ["重跑 install.bat。"])
        return False
    return True


def step_requirements(project_dir: Path, venv_python: Path, *, pypi_mirror: str) -> bool:
    section(3, "安装其余依赖 (Demucs / FastAPI 等)")
    req = project_dir / "requirements.txt"
    if not req.is_file():
        human_error("缺少 requirements.txt", str(req), ["确认项目文件完整。"])
        return False

    args = [
        str(venv_python), "-m", "pip", "install", "-r", str(req),
        "--disable-pip-version-check", "--retries", "10", "--timeout", "120",
    ]
    mirror_url = PYPI_MIRRORS.get(pypi_mirror)
    if mirror_url:
        args += ["--index-url", mirror_url]
        log(f"  使用 PyPI 镜像：{mirror_url}")
    code = run(args)
    if code != 0:
        human_error(
            "依赖安装失败",
            f"pip 返回 {code}",
            [
                "检查网络连接。",
                "重跑 install.bat（已下载的部分会被复用）。",
                "试试官方源：install.bat --pypi-mirror official",
                "若卡在 demucs 的依赖上，可单独执行：."
                "venv\\Scripts\\python.exe -m pip install demucs==4.1.0",
            ],
        )
        return False
    return True


def step_ffmpeg(project_dir: Path, venv_python: Path, *, pypi_mirror: str) -> bool:
    section(4, "准备 FFmpeg")

    tools_bin = project_dir / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if tools_bin.is_file():
        log(f"  已存在，跳过：{tools_bin}")
        return True

    # Preferred route: the pip wheel, which bundles a complete static FFmpeg.
    # Measured here: a direct download from gyan.dev ran at 0.02 MB/s while pip
    # fetched the same build in a fraction of the time, and PyPI gives us
    # mirrors and resume for free.
    log("  正在通过 pip 获取内置 FFmpeg (imageio-ffmpeg) ...")
    log("  说明：该包内含完整的静态 FFmpeg，无需单独下载。")
    log()

    pip_args = [
        str(venv_python), "-m", "pip", "install",
        "--disable-pip-version-check", "imageio-ffmpeg",
    ]
    mirror_url = PYPI_MIRRORS.get(pypi_mirror)
    if mirror_url:
        pip_args += ["--index-url", mirror_url]
    code = run(pip_args)
    if code != 0:
        human_error(
            "FFmpeg 获取失败",
            f"pip 返回 {code}",
            [
                "检查网络连接后重跑 install.bat。",
                "或手动下载 https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
                "解压后把里面的 bin 目录复制到 " + str(project_dir / "tools" / "ffmpeg" / "bin"),
                "或安装 FFmpeg 并加入系统 PATH。",
            ],
        )
        return False

    # Locate it and copy into tools/ffmpeg/bin so the layout is predictable.
    # NOTE: the lookup must run *inside the venv*, because imageio_ffmpeg is
    # installed there - this installer process has its own sys.path and would
    # not see it.
    try:
        proc = subprocess.run(
            [str(venv_python), "-c",
             "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        exe_output = (proc.stdout or "").strip()
        if proc.returncode != 0 or not exe_output:
            raise FileNotFoundError((proc.stderr or "").strip()[:300] or "定位失败")

        exe_path = Path(exe_output)
        if not exe_path.is_file():
            raise FileNotFoundError(f"路径不存在：{exe_path}")

        log(f"  找到 FFmpeg：{exe_path}")
        tools_bin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe_path, tools_bin)
        log(f"  已复制到：{tools_bin}")
        log("  注意：该来源不含 ffprobe，程序会自动改用 ffmpeg 读取音频元数据。")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"  复制到 tools 目录失败：{exc}")
        log("  程序仍会直接使用 pip 包内的 FFmpeg，功能不受影响。")
        return True


def step_models(project_dir: Path) -> None:
    section(5, "AI 模型")
    try:
        # Absolute import on purpose: install.bat runs this file as a script
        # (``python app\\setup.py``), so there is no parent package and a
        # relative import would raise "attempted relative import with no known
        # parent package".
        from app.separation import models as model_store

        model_store.configure_cache()
        statuses = model_store.all_statuses()
    except Exception as exc:  # noqa: BLE001
        log(f"  无法检查模型状态：{type(exc).__name__}: {exc}")
        return

    installed = [s for s in statuses if s.installed]

    if installed:
        log("  已就绪的模型：")
        for status in installed:
            log(f"    [OK] {status.signature}  （{status.display_name}）")
        log()
        log("  模型已缓存，后续运行无需联网。")
        missing = [s for s in statuses if not s.installed]
        if missing:
            log()
            log("  可选的其他模型（需要时在网页界面点击「下载模型」）：")
            for status in missing:
                log(f"    {status.signature}  {status.description}  约 {status.size_mb} MB")
    else:
        log("  尚未下载任何模型。")
        log()
        log("  首次使用时，在网页界面展开「高级设置」并点击「下载模型」")
        log("  （默认 htdemucs，约 80 MB，来自 HuggingFace 官方仓库 adefossez/HTDemucs）。")

    log()
    log(f"  模型目录：{project_dir / 'models'}")
    log(f"  当前占用：{model_store.cache_size_mb():.1f} MB")


def step_verify(project_dir: Path, venv_python: Path) -> bool:
    section(6, "验证安装")
    checks: list[tuple[str, bool, str]] = []

    def probe(label: str, code: str) -> tuple[str, bool, str]:
        proc = subprocess.run(
            [str(venv_python), "-c", code],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        return label, proc.returncode == 0, output.splitlines()[0] if output else ""

    results = [
        probe("Python", "import sys;print(sys.version.split()[0])"),
        probe(
            "PyTorch",
            "import torch;print(torch.__version__ + ' cuda=' + str(torch.version.cuda))",
        ),
        probe(
            "CUDA 可用",
            "import torch;"
            "print('True ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'False')",
        ),
        probe("Demucs", "import demucs;print('已安装')"),
        probe("FastAPI", "import fastapi;print(fastapi.__version__)"),
        probe("NumPy", "import numpy;print(numpy.__version__)"),
    ]

    log()
    for label, ok, detail in results:
        mark = "OK  " if ok else "FAIL"
        log(f"  [{mark}] {label:<12} {detail}")

    # FFmpeg / model status through the app's own detection logic.
    log()
    proc = subprocess.run(
        [str(venv_python), "-c",
         "import sys;sys.path.insert(0, r'{}');"
         "from app.audio import ffmpeg as f;i=f.detect();"
         "print((i.version + '  source=' + i.source + '  ffprobe=' + str(i.ffprobe is not None)) "
         "if i else 'NOT FOUND')".format(str(project_dir))],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    ffmpeg_out = (proc.stdout or "").strip()
    ffmpeg_ok = proc.returncode == 0 and "NOT FOUND" not in ffmpeg_out
    log(f"  [{'OK  ' if ffmpeg_ok else 'FAIL'}] {'FFmpeg':<12} {ffmpeg_out}")

    critical_ok = all(ok for _, ok, _ in results) and ffmpeg_ok
    if not critical_ok:
        human_error(
            "安装后验证未通过",
            "有组件未正确安装",
            [
                "重跑 install.bat。",
                "运行 diagnose.bat 查看详细报告。",
                "把 logs\\install.log 发给开发者。",
            ],
        )
    return critical_ok


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    global _log_handle

    parser = argparse.ArgumentParser(description="Drum Practice Generator 安装程序")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--skip-torch", action="store_true", help="跳过 PyTorch（调试用）")
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, choices=sorted(TORCH_VARIANTS),
        help=f"CUDA 版本，默认 {DEFAULT_VARIANT}（有国内镜像加速）",
    )
    parser.add_argument(
        "--source", default="auto", choices=("auto", "mirror", "official"),
        help="PyTorch 下载源：auto 会实测速度后自动选择",
    )
    parser.add_argument(
        "--pypi-mirror", default=DEFAULT_PYPI_MIRROR, choices=sorted(PYPI_MIRRORS),
        help=f"其他依赖使用的 PyPI 镜像，默认 {DEFAULT_PYPI_MIRROR}",
    )
    args = parser.parse_args(argv)

    project_dir = Path(args.project_dir).resolve() if args.project_dir else Path(__file__).resolve().parent.parent

    # install.bat invokes this file as a script, so the project root is not on
    # sys.path and ``import app.*`` would fail.  Add it explicitly.
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    try:
        _log_handle = (project_dir / "logs" / "install.log").open("w", encoding="utf-8")
    except OSError:
        _log_handle = None

    log("=" * 75)
    log("  Drum Practice Generator - 安装程序")
    log("=" * 75)
    log(f"  项目目录 : {project_dir}")
    log(f"  Python   : {sys.version.split()[0]}  ({sys.executable})")
    log(f"  目标版本 : Python 3.11 + PyTorch {TORCH_VERSION} "
        f"({TORCH_VARIANTS[args.variant]['label']})")
    log(f"  下载源   : PyTorch={args.source}  PyPI={args.pypi_mirror}")
    log(f"  日志文件 : {project_dir / 'logs' / 'install.log'}")
    log("=" * 75)

    started = time.time()

    venv_python = step_venv(project_dir, sys.executable)
    if venv_python is None:
        return 1

    if not step_torch(
        venv_python, args.skip_torch,
        variant=args.variant, source=args.source, pypi_mirror=args.pypi_mirror,
    ):
        return 2

    if not step_requirements(project_dir, venv_python, pypi_mirror=args.pypi_mirror):
        return 3

    if not step_ffmpeg(project_dir, venv_python, pypi_mirror=args.pypi_mirror):
        return 4

    step_models(project_dir)

    ok = step_verify(project_dir, venv_python)

    elapsed = time.time() - started
    log()
    log("=" * 75)
    log(f"  安装{'完成' if ok else '未通过'}，用时 {elapsed / 60:.1f} 分钟")
    log("=" * 75)

    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
