@echo off
REM ===========================================================================
REM  Drum Practice Generator - installer
REM
REM  IMPORTANT: this file is intentionally ASCII-only with CRLF line endings.
REM  Windows cmd.exe mis-tokenizes batch files that use LF-only endings, and
REM  non-ASCII text in a .bat is decoded using the OEM code page (not UTF-8),
REM  which garbles the script.  Chinese messages live in app\setup.py instead,
REM  where Python controls the encoding.
REM ===========================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Force Python's output to UTF-8 so the Chinese installer messages are not
REM garbled by the console codepage.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM %~dp0 always ends with a backslash.  Strip it, otherwise "%PROJECT_DIR%"
REM expands to "...\generator\" and the \" escapes the closing quote, which leaks
REM a literal quote into any program that receives the path as an argument.
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "LOG=%PROJECT_DIR%\logs\install.log"
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs" >nul 2>&1

title Drum Practice Generator - Install

echo.
echo ===========================================================================
echo   Drum Practice Generator - Installer
echo ===========================================================================
echo   Project : %PROJECT_DIR%
echo   Log     : %LOG%
echo ===========================================================================
echo.

REM ---------------------------------------------------------------------------
REM  Find a usable Python (skip the Microsoft Store alias stub)
REM ---------------------------------------------------------------------------
set "PY_CMD="

REM Prefer 3.11 (the tested combination), then 3.12 / 3.10.
for %%V in (3.11 3.12 3.10) do (
    if not defined PY_CMD (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -%%V"
    )
)

if not defined PY_CMD (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python312\python.exe"
    ) do (
        if not defined PY_CMD (
            if exist %%P set "PY_CMD=%%~P"
        )
    )
)

if not defined PY_CMD (
    for /f "delims=" %%p in ('where python 2^>nul') do (
        if not defined PY_CMD (
            echo %%p | findstr /I "WindowsApps" >nul
            if errorlevel 1 set "PY_CMD=%%p"
        )
    )
)

REM 1d. Registry lookup: covers installs whose PATH entry is missing or stale,
REM     which happens when the user did not tick "Add python.exe to PATH".
if not defined PY_CMD (
    for /f "tokens=2,*" %%a in ('reg query "HKLM\SOFTWARE\Python\PythonCore\3.11\InstallPath" /ve 2^>nul ^| findstr /I "REG_SZ"') do (
        if not defined PY_CMD if exist "%%b\python.exe" set "PY_CMD=%%b\python.exe"
    )
)
if not defined PY_CMD (
    for /f "tokens=2,*" %%a in ('reg query "HKCU\SOFTWARE\Python\PythonCore\3.11\InstallPath" /ve 2^>nul ^| findstr /I "REG_SZ"') do (
        if not defined PY_CMD if exist "%%b\python.exe" set "PY_CMD=%%b\python.exe"
    )
)

if defined PY_CMD goto :have_python

echo   Python was not found on this machine.
echo.
echo   Attempting to install Python 3.11 with winget...
echo   (a UAC prompt may appear - please approve it)
echo.
where winget >nul 2>&1
if errorlevel 1 goto :no_winget

winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :winget_failed

echo.
echo   Python 3.11 installed.
echo   Please CLOSE this window and run install.bat again.
echo.
pause
exit /b 0

:no_winget
echo   [ERROR] winget is not available, so Python cannot be installed automatically.
echo.
echo   Please install Python 3.11 manually:
echo     1. Open https://www.python.org/downloads/windows/
echo     2. Download "Windows installer (64-bit)" and run it
echo     3. IMPORTANT: tick "Add python.exe to PATH" during setup
echo     4. Run install.bat again
echo.
pause
exit /b 1

:winget_failed
echo.
echo   [ERROR] winget could not install Python.
echo.
echo   Please install Python 3.11 manually:
echo     https://www.python.org/downloads/windows/
echo   Tick "Add python.exe to PATH" during setup, then run install.bat again.
echo.
pause
exit /b 1

:have_python
echo   Using Python: %PY_CMD%
echo.
echo ---------------------------------------------------------------------------
echo   Handing over to the installer (all output is also saved to the log file)
echo ---------------------------------------------------------------------------
echo.

%PY_CMD% "%PROJECT_DIR%\app\setup.py" --project-dir "%PROJECT_DIR%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo ===========================================================================
    echo   INSTALL COMPLETED SUCCESSFULLY
    echo ===========================================================================
    echo.
    echo   Next steps:
    echo     1. Double-click start.bat   (opens the web UI in your browser)
    echo     2. In the web UI click "Download model" (one time, about 80 MB)
    echo     3. Drag in a song and click Generate
    echo.
    echo   If the GPU is not used, run diagnose.bat for a full report.
    echo ===========================================================================
) else (
    echo ===========================================================================
    echo   INSTALL FAILED  (exit code %RC%)
    echo ===========================================================================
    echo.
    echo   See the messages above, and the full log at:
    echo     %LOG%
    echo.
    echo   Please send that log file when reporting the problem.
    echo ===========================================================================
)
echo.
pause
exit /b %RC%
