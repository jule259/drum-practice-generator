@echo off
REM ===========================================================================
REM  Drum Practice Generator - environment diagnostics
REM
REM  ASCII-only with CRLF line endings on purpose: cmd.exe mis-tokenizes
REM  LF-only batch files, and non-ASCII bytes in a .bat are decoded with the OEM
REM  code page.  The report itself is printed by Python (app\diagnose.py).
REM ===========================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Force Python's output to UTF-8 so the Chinese report is not garbled by the
REM console codepage.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM %~dp0 ends with a backslash; strip it so paths below have a single separator.
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "FFMPEG_BIN=%PROJECT_DIR%\tools\ffmpeg\bin"

title Drum Practice Generator - Diagnostics

if exist "%FFMPEG_BIN%\ffmpeg.exe" set "PATH=%FFMPEG_BIN%;%PATH%"

if not exist "%VENV_PY%" (
    echo.
    echo   [ERROR] Virtual environment not found:
    echo           %VENV_PY%
    echo.
    echo   Run install.bat first.
    echo.
    echo   ---- falling back to a limited system-Python report ----
    echo.
    py -3 "%~dp0diagnose.py" %*
    echo.
    pause
    exit /b 1
)

"%VENV_PY%" -m app.diagnose %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Result: environment is READY - you can run start.bat
) else (
    echo   Result: environment is NOT ready - see the guidance above.
    echo           Usually this just means you need to run install.bat.
)
echo.
pause
exit /b %RC%
