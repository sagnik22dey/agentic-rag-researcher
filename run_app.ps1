$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found! Please create .venv first."
    exit 1
}

& ".\.venv\Scripts\Activate.ps1"
& ".\.venv\Scripts\streamlit.exe" run app.py
