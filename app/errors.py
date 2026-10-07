"""User-facing error types.

The whole point of this module is rule 17 of the spec: never show the user a
bare "Error".  Every failure that can realistically happen gets a dedicated
exception carrying a *code*, a human-readable message, and concrete suggestions
the UI can display as a bullet list.

The API layer serialises these into ``{"error": {...}}`` responses.
"""

from __future__ import annotations

from typing import Iterable


class DrumPracticeError(Exception):
    """Base class for every expected failure.

    ``message``  - what went wrong, in the user's language, no jargon.
    ``suggestions`` - actionable next steps shown as a list.
    ``code``     - stable machine-readable identifier for the frontend.
    ``detail``   - raw technical text, shown in a collapsible panel only.
    """

    code = "internal_error"
    http_status = 500

    def __init__(
        self,
        message: str,
        suggestions: Iterable[str] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.suggestions = list(suggestions or [])
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "suggestions": self.suggestions,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# Environment / installation problems
# --------------------------------------------------------------------------
class FFmpegNotFoundError(DrumPracticeError):
    code = "ffmpeg_missing"
    http_status = 503

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            "FFmpeg 未找到。FFmpeg 是音频解码、变速和导出的必需组件。",
            suggestions=[
                "运行 install.bat，它会自动下载 FFmpeg 到项目的 tools\\ffmpeg 目录。",
                "或者手动安装 FFmpeg 并将其 bin 目录加入系统 PATH。",
                "安装后重新运行 diagnose.bat 确认状态。",
            ],
            detail=detail,
        )


class FFmpegRunError(DrumPracticeError):
    code = "ffmpeg_failed"
    http_status = 500

    def __init__(self, message: str, detail: str | None = None, suggestions=None) -> None:
        super().__init__(
            message,
            suggestions=suggestions
            or [
                "确认输入文件可以正常播放。",
                "查看 logs\\ 目录下最新日志的 detail 字段获取 FFmpeg 原始输出。",
            ],
            detail=detail,
        )


class ModelNotInstalledError(DrumPracticeError):
    code = "model_missing"
    http_status = 409

    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            f"AI 模型 “{model}” 尚未下载。",
            suggestions=[
                "在界面点击“下载模型”按钮（约 80 MB，只需一次）。",
                "模型来自官方仓库：HuggingFace (Kyutai/Demucs) 或 Meta 官方 AWS 镜像。",
                "确认网络可访问 huggingface.co；国内网络可能需要代理。",
            ],
            detail=detail,
        )


class ModelDownloadError(DrumPracticeError):
    code = "model_download_failed"
    http_status = 502

    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            f"模型 “{model}” 下载失败。",
            suggestions=[
                "检查网络连接 / 代理设置。",
                "重试一次；HuggingFace 偶发超时属正常现象。",
                "若持续失败，可手动下载后放入 models\\demucs 目录。",
            ],
            detail=detail,
        )


class CUDAUnavailableError(DrumPracticeError):
    code = "cuda_unavailable"
    http_status = 503

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            "CUDA GPU 不可用，程序将回退到 CPU 模式（速度会慢很多）。",
            suggestions=[
                "确认安装了 CUDA 版 PyTorch：运行 diagnose.bat 查看 PyTorch 版本是否带 +cuXXX。",
                "如果显示 CPU 版 PyTorch，请重新运行 install.bat。",
                "更新 NVIDIA 显卡驱动后重启。",
            ],
            detail=detail,
        )


class OutOfMemoryError(DrumPracticeError):  # noqa: A001 - clearer than GPUOOMError
    code = "gpu_oom"
    http_status = 507

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            "GPU 显存不足。",
            suggestions=[
                "在高级设置中降低 Segment（例如从 7 改为 5）。",
                "改用较小的模型 htdemucs（不要用 htdemucs_ft）。",
                "关闭其他占用显存的程序（浏览器、游戏、其他 AI 工具）。",
                "把 Shifts 设为 1。",
            ],
            detail=detail,
        )


# --------------------------------------------------------------------------
# Input problems
# --------------------------------------------------------------------------
class UnsupportedFormatError(DrumPracticeError):
    code = "unsupported_format"
    http_status = 415

    def __init__(self, filename: str, supported: Iterable[str]) -> None:
        super().__init__(
            f"不支持的文件格式：{filename}",
            suggestions=[
                "支持的格式：" + "、".join(sorted(supported)),
                "如果文件确实是上述格式之一，可能是扩展名与实际编码不符，可用 FFmpeg 转成 WAV 再试。",
            ],
        )


class CorruptAudioError(DrumPracticeError):
    code = "corrupt_audio"
    http_status = 422

    def __init__(self, filename: str, detail: str | None = None) -> None:
        super().__init__(
            f"无法解码音频文件：{filename}",
            suggestions=[
                "先用播放器确认该文件能正常播放。",
                "文件可能未下载完整或已损坏，请重新获取。",
                "若是 DRM 保护的 M4A，请先转换为普通 MP3/WAV。",
            ],
            detail=detail,
        )


class FileTooLargeError(DrumPracticeError):
    code = "file_too_large"
    http_status = 413

    def __init__(self, size_mb: float, limit_mb: int) -> None:
        super().__init__(
            f"文件过大：{size_mb:.0f} MB（上限 {limit_mb} MB）。",
            suggestions=[
                "先用 FFmpeg 压缩为 320 kbps MP3 再导入。",
                "或直接截取需要练习的片段。",
            ],
        )


class InsufficientDiskSpaceError(DrumPracticeError):
    code = "disk_full"
    http_status = 507

    def __init__(self, required_gb: float, free_gb: float, path: str) -> None:
        super().__init__(
            f"磁盘空间不足：{path} 需要约 {required_gb:.1f} GB，当前可用 {free_gb:.1f} GB。",
            suggestions=[
                "清理磁盘后重试。",
                "删除 output\\ 和 temp\\ 中的历史文件。",
                "通过环境变量 DPG_OUTPUT_DIR 把输出目录改到大容量磁盘。",
            ],
        )


class OutputNotWritableError(DrumPracticeError):
    code = "output_not_writable"
    http_status = 500

    def __init__(self, path: str, detail: str | None = None) -> None:
        super().__init__(
            f"输出目录不可写：{path}",
            suggestions=[
                "不要把项目放在受保护的系统目录（如 C:\\Program Files）。",
                "检查杀毒软件是否拦截了写入。",
                "删除该目录下的只读属性后重试。",
            ],
            detail=detail,
        )


# --------------------------------------------------------------------------
# Task lifecycle
# --------------------------------------------------------------------------
class TaskCancelledError(DrumPracticeError):
    code = "cancelled"
    http_status = 409

    def __init__(self) -> None:
        super().__init__("任务已被取消。", suggestions=["可以重新点击生成。"])
