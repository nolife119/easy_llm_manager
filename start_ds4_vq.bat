@echo off
cd /d "%~dp0"
py -3 ds4_oneclick.py
if errorlevel 1 (
  echo.
  echo DS4 preflight or startup did not succeed. Read the message above.
  pause
)
