"""Background job manager.

Design notes:

* **One heavy job at a time.**  A global lock serialises pipeline runs.  Two
  concurrent Demucs inferences on a 16 GB card is how you get an OOM, and the
  spec explicitly asks us to avoid VRAM blow-ups.
* **Threads, not processes.**  The pipeline is I/O and GPU bound; a worker thread
  keeps the model resident and avoids pickling.
* **Cancellation is cooperative.**  ``should_cancel`` is polled at every stage
  boundary by the pipeline, so Cancel stops the job within a second or two.
* **Errors are structured.**  :class:`DrumPracticeError` carries code, message
  and suggestions; anything else is wrapped so the UI never shows a bare
  "Error".
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import DrumPracticeError, TaskCancelledError
from .logging_setup import get_logger

logger = get_logger("tasks")

# Keep the last N finished jobs in memory for the UI's state polling.
MAX_HISTORY = 30

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

TERMINAL = {STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED}


@dataclass
class Job:
    id: str
    kind: str
    label: str = ""
    status: str = STATUS_QUEUED
    progress: float = 0.0
    percent: float = 0.0
    stage: str = ""
    stage_label: str = ""
    note: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: dict | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "progress": round(self.progress, 4),
            "percent": round(self.percent, 1),
            "stage": self.stage,
            "stage_label": self.stage_label,
            "note": self.note,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
            ),
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }


class TaskManager:
    """Registry and runner for background jobs."""

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._heavy_lock = threading.Lock()
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in self._order[-limit:] if jid in self._jobs]

    def active(self) -> Job | None:
        with self._lock:
            for jid in reversed(self._order):
                job = self._jobs.get(jid)
                if job and job.status in (STATUS_QUEUED, STATUS_RUNNING):
                    return job
        return None

    def is_busy(self) -> bool:
        return self.active() is not None

    # ------------------------------------------------------------------
    # Mutating
    # ------------------------------------------------------------------
    def cancel(self, job_id: str) -> bool:
        """Request cancellation.  Returns False for unknown/finished jobs."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                return False
            job.cancel_requested = True
            job.note = "正在取消…"
            logger.info("请求取消任务 %s", job_id)
            return True

    def submit(
        self,
        kind: str,
        func: Callable[..., Any],
        *args,
        label: str = "",
        exclusive: bool = True,
        **kwargs,
    ) -> Job:
        """Run ``func(*args, should_cancel=..., progress=...)`` on a worker thread.

        ``func`` must accept the keyword arguments ``progress`` and
        ``should_cancel``.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, kind=kind, label=label)

        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._trim_locked()

        thread = threading.Thread(
            target=self._run,
            args=(job, func, args, kwargs, exclusive),
            name=f"dpg-{kind}-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def _trim_locked(self) -> None:
        if len(self._order) <= self._max_history:
            return
        for jid in list(self._order[: -self._max_history]):
            existing = self._jobs.get(jid)
            if existing and existing.status in TERMINAL:
                self._order.remove(jid)
                self._jobs.pop(jid, None)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(
        self,
        job: Job,
        func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        exclusive: bool,
    ) -> None:
        def progress(payload: dict) -> None:
            with self._lock:
                job.progress = float(payload.get("progress", job.progress))
                job.percent = float(payload.get("percent", job.progress * 100))
                job.stage = payload.get("stage", job.stage)
                job.stage_label = payload.get("stage_label", job.stage_label)
                note = payload.get("note")
                if note:
                    job.note = str(note)

        def should_cancel() -> bool:
            return job.cancel_requested

        acquired = False
        if exclusive:
            # Block (rather than fail) so a queued job still runs afterwards.
            self._heavy_lock.acquire()
            acquired = True

        try:
            with self._lock:
                if job.cancel_requested:
                    job.status = STATUS_CANCELLED
                    job.finished_at = time.time()
                    job.note = "已取消"
                    return
                job.status = STATUS_RUNNING
                job.started_at = time.time()
                job.note = "开始处理…"

            logger.info("任务 %s (%s) 开始：%s", job.id, job.kind, job.label)
            outcome = func(*args, progress=progress, should_cancel=should_cancel, **kwargs)

            with self._lock:
                if job.cancel_requested:
                    job.status = STATUS_CANCELLED
                    job.note = "已取消"
                else:
                    job.status = STATUS_DONE
                    job.progress = 1.0
                    job.percent = 100.0
                    job.note = "完成"
                job.result = _jsonable(outcome)
                job.finished_at = time.time()

            logger.info(
                "任务 %s 结束：%s（%.1fs）",
                job.id, job.status, (job.finished_at or 0) - (job.started_at or 0),
            )

        except TaskCancelledError as exc:
            with self._lock:
                job.status = STATUS_CANCELLED
                job.note = "已取消"
                job.error = exc.to_dict()
                job.finished_at = time.time()
            logger.info("任务 %s 已取消", job.id)

        except DrumPracticeError as exc:
            with self._lock:
                job.status = STATUS_ERROR
                job.error = exc.to_dict()
                job.note = exc.message
                job.finished_at = time.time()
            logger.error("任务 %s 失败：%s", job.id, exc.message)

        except BaseException as exc:  # noqa: BLE001 - must never kill the worker silently
            detail = traceback.format_exc()
            with self._lock:
                job.status = STATUS_ERROR
                job.error = {
                    "code": "unexpected",
                    "message": f"未预期的错误：{type(exc).__name__}: {exc}",
                    "suggestions": [
                        "重试一次。",
                        "查看 logs\\drum-practice.log 的完整堆栈。",
                        "运行 diagnose.bat 检查环境。",
                    ],
                    "detail": detail[-4000:],
                }
                job.note = job.error["message"]
                job.finished_at = time.time()
            logger.error("任务 %s 未预期失败：%s\n%s", job.id, exc, detail)

        finally:
            if acquired:
                self._heavy_lock.release()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a result object for the API."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


# Module-level singleton used by the API layer.
manager = TaskManager()
