@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -Command "try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:7860/'; Start-Process 'http://127.0.0.1:7860/'; exit 0 } catch { exit 1 }"
if not errorlevel 1 exit /b 0
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B app.py
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -B app.py
  ) else (
    python -B app.py
  )
)
if errorlevel 1 pause
