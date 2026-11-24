$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
} catch {
    Write-Host "[INFO] Ollama server is not running. Starting Ollama in the background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found! Please create .venv first."
    exit 1
}

& ".\.venv\Scripts\Activate.ps1"
& ".\.venv\Scripts\streamlit.exe" run app.py
