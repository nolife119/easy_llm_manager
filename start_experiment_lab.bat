@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 experiment_lab.py) else (python experiment_lab.py)
if errorlevel 1 (
 echo The experiment dashboard did not start. Check that Python 3.10+ is installed.
 pause
)
