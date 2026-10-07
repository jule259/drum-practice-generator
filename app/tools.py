"""Thin, well-behaved wrapper around running external executables.

Two things matter here on Windows:

1. **No console flash.**  Every ``subprocess`` call passes ``CREATE_NO_WINDOW``
   so launching FFmpeg from the web server never pops a black window.

2. **Encoding is a lottery.**  FFmpeg and Demucs write status text that may or
   may not be valid UTF-8/CP936 depending on the machine's code page.  We decode
   with ``errors="replace"`` so a stray byte can never crash a task with a
   ``UnicodeDecodeError``.
"""

from __future__ import annotations

import locale
import os
import shutil
import subprocess
import sys
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def no_window_kwargs() -> dict:
    """subprocess kwargs that suppress the console window on Windows."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def decode_output(raw: bytes | None) -> str:
    """Best-effort decode of a subprocess stream."""
    if not raw:
        return ""
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp936", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def which(executable: str, extra_dirs: list[Path] | None = None) -> Path | None:
    """Locate an executable on PATH, then in ``extra_dirs``.

    ``extra_dirs`` lets us prefer a project-bundled copy over a system one.
    """
    for directory in extra_dirs or []:
        try:
            candidate = Path(directory) / (
                executable + ".exe" if sys.platform == "win32" else executable
            )
            if candidate.is_file():
                return candidate
        except OSError:
            continue

    found = shutil.which(executable)
    if found:
        return Path(found)

    # Windows quirk: some tools live in the Microsoft Store alias dir, which
    # shutil.which resolves but which is an App Execution Alias stub that
    # fails when run.  Callers verify by actually executing the binary.
    return None


def run(
    args: list[str | Path],
    *,
    timeout: float | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command and return the completed process.

    Raises ``FileNotFoundError`` if the executable is missing - callers turn
    that into a specific user-facing error.
    """
    argv = [str(a) for a in args]
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None
    proc = subprocess.run(  # noqa: S603 - argv is built by us, never a shell string
        argv,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        shell=False,
        **no_window_kwargs(),
    )
    if capture:
        # Re-attach decoded text so callers do not repeat the dance.
        proc.stdout_text = decode_output(proc.stdout)  # type: ignore[attr-defined]
        proc.stderr_text = decode_output(proc.stderr)  # type: ignore[attr-defined]
    else:
        proc.stdout_text = ""  # type: ignore[attr-defined]
        proc.stderr_text = ""  # type: ignore[attr-defined]

    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, argv, output=proc.stdout, stderr=proc.stderr
        )
    return proc


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the volume holding ``path`` (walks up to an existing dir)."""
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return 0


def env_with_tools(extra_dirs: list[Path]) -> dict[str, str]:
    """Return os.environ with ``extra_dirs`` prepended to PATH."""
    env = dict(os.environ)
    parts = [str(d) for d in extra_dirs if Path(d).is_dir()]
    if parts:
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env
