@echo off
setlocal
cd /d "%~dp0"
set "MINI_PY="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "MINI_PY=py -3.12"
    ) else (
        set "MINI_PY=py -3"
    )
) else (
    set "MINI_PY=python"
)
echo [Inkling Mini] Checking Windows Firewall for LAN access...
%MINI_PY% mini_firewall.py
if errorlevel 1 (
    echo.
    echo Firewall setup failed or UAC was cancelled. LAN server not started.
    echo Please review Windows Firewall administrator policy.
    pause
    exit /b 1
)
%MINI_PY% mini_platform.py
if errorlevel 1 (
    echo.
    echo Python 3.10+ and Windows x64 are required.
    pause
    exit /b 1
)
