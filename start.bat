@echo off
REM ===========================================================================
REM  Drum Practice Generator - launcher
REM
REM  ASCII-only with CRLF line endings on purpose: cmd.exe mis-tokenizes
REM  LF-only batch files, and non-ASCII bytes in a .bat are decoded with the OEM
REM  code page.  All Chinese messages come from the Python program itself.
REM ===========================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Force Python's stdout/stderr to UTF-8 so Chinese messages survive the
REM console codepage. Without this Python encodes with the ANSI code page (GBK
REM on Chinese Windows) and the text arrives garbled.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM %~dp0 ends with a backslash; strip it so paths below have a single separator.
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "FFMPEG_BIN=%PROJECT_DIR%\tools\ffmpeg\bin"

title Drum Practice Generator

echo.
echo ===========================================================================
echo   Drum Practice Generator
echo ===========================================================================
echo.

if not exist "%VENV_PY%" (
    echo   [ERROR] Virtual environment not found:
    echo           %VENV_PY%
    echo.
    echo   Please run install.bat first.
    echo.
    pause
    exit /b 1
)

REM Put the bundled FFmpeg on PATH so Demucs can find a bare "ffmpeg".
if exist "%FFMPEG_BIN%\ffmpeg.exe" (
    set "PATH=%FFMPEG_BIN%;%PATH%"
    set "FFMPEG_BINARY=%FFMPEG_BIN%\ffmpeg.exe"
)

REM Fail fast with a readable message instead of a Python traceback.
"%VENV_PY%" -c "import torch, demucs, fastapi, numpy" >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Dependencies are incomplete: torch / demucs / fastapi / numpy.
    echo.
    echo   Please run install.bat again.
    echo.
    pause
    exit /b 1
)

REM Detect the GPU name.  Two robustness notes:
REM  * done via a temp file rather than for/f, because the inline Python contains
REM    quotes, colons and parentheses that cmd.exe parses unreliably inside a
REM    for/f command string;
REM  * the temp file lives in the project directory, not %TEMP%, because some
REM    sandboxed/restricted environments deny child-process writes to %TEMP%.
set "GPUFILE=%PROJECT_DIR%\.gpu_probe.tmp"
set "GPU_NAME=CPU"
if exist "%GPUFILE%" del /q "%GPUFILE%" >nul 2>&1
"%VENV_PY%" -c "import torch,pathlib;pathlib.Path(r'%GPUFILE%').write_text(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', encoding='utf-8')" >nul 2>&1
if exist "%GPUFILE%" set /p GPU_NAME=<"%GPUFILE%"
if exist "%GPUFILE%" del /q "%GPUFILE%" >nul 2>&1
if not defined GPU_NAME set "GPU_NAME=CPU"

if /I "%GPU_NAME%"=="CPU" (
    echo   [WARNING] CUDA GPU unavailable - falling back to CPU mode.
    echo             Processing will be much slower.
    echo             Run diagnose.bat to see why.
    echo.
) else (
    echo   GPU: %GPU_NAME%
    echo.
)

echo   Starting the local server - your browser will open automatically.
echo   Press Ctrl+C in this window, or close it, to stop.
echo.

"%VENV_PY%" -m app.main %*
set "RC=%ERRORLEVEL%"

echo.
echo   Server stopped (exit code %RC%).
echo.
pause
exit /b %RC%
