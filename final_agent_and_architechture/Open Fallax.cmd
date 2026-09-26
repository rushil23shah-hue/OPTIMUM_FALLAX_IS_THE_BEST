@echo off
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -B dashboard.py
) else (
  python -B dashboard.py
)
if errorlevel 1 pause
