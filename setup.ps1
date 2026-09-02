param(
  [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDirectory

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
  & $PythonCommand -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".\.venv\Scripts\python.exe" -m pytest

Write-Host "Environment ready. Start the read-only result viewer with:"
Write-Host "  .\.venv\Scripts\python.exe app.py"
Write-Host "Then open http://127.0.0.1:7860/"
