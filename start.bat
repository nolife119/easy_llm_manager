@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    py -3.12 manager.py
    goto done
  )
)
python manager.py
:done
pause
