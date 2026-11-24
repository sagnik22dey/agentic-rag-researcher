@echo off
cd /d "%~dp0"

curl.exe -s -m 2 http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama server is not running. Starting Ollama in the background...
    start "" /b ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
)

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run app.py
pause
