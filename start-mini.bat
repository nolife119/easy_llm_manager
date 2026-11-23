@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    py -3.12 mini_platform.py
    goto end
  )
  py -3 mini_platform.py
  goto end
)
python mini_platform.py
:end
if errorlevel 1 (
  echo.
  echo Python 3.10 or later is required. Verify PATH / py launcher.
  pause
)
