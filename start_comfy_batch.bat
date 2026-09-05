@echo off
setlocal
REM ComfyBatchTool launcher (Windows). Double-click to start the WebUI.
REM Detects a usable Python 3.8+ automatically. No hardcoded private paths.
cd /d "%~dp0"
set "PYTHON="

REM 0) optional explicit override
if defined PYTHON_OVERRIDE if exist "%PYTHON_OVERRIDE%" set "PYTHON=%PYTHON_OVERRIDE%"

REM 1) python on PATH, optional, skipped silently if where is unavailable
if not defined PYTHON (
    for %%E in (python python3 py) do (
        if not defined PYTHON (
            for /f "delims=" %%P in ('where %%E 2^>nul') do (
                if not defined PYTHON set "PYTHON=%%P"
            )
        )
    )
)

REM 2) common install locations, exact path check, works without System32 in PATH
if not defined PYTHON (
    for %%D in (
        "%LOCALAPPDATA%\Programs\Python\Python313"
        "%LOCALAPPDATA%\Programs\Python\Python312"
        "%LOCALAPPDATA%\Programs\Python\Python311"
        "%LOCALAPPDATA%\Programs\Python\Python310"
        "%LOCALAPPDATA%\Programs\Python\Python39"
        "%LOCALAPPDATA%\Programs\Python\Python38"
        "C:\Python313"
        "C:\Python312"
        "C:\Python311"
        "C:\Python310"
        "C:\Python39"
        "C:\Program Files\Python313"
        "C:\Program Files\Python312"
        "C:\Program Files (x86)\Python312"
    ) do (
        if not defined PYTHON if exist "%%~D\python.exe" set "PYTHON=%%~D\python.exe"
    )
)

REM 3) Windows py launcher
if not defined PYTHON (
    if exist "%SystemRoot%\py.exe" set "PYTHON=%SystemRoot%\py.exe"
)
if not defined PYTHON (
    if exist "C:\Windows\py.exe" set "PYTHON=C:\Windows\py.exe"
)

if not defined PYTHON (
    echo [ERROR] Python 3.8+ not found.
    echo Install Python from https://www.python.org/downloads/ and tick Add python.exe to PATH,
    echo or set PYTHON_OVERRIDE to the full path of your python.exe, then re-run.
    pause
    exit /b 1
)

if not exist "comfy_batch_tool.py" (
    echo [ERROR] comfy_batch_tool.py not found in this folder.
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
echo Starting ComfyBatchTool WebUI ...
echo Open http://127.0.0.1:8091 in your browser after it boots.
echo Close this window to stop the service.
echo.
"%PYTHON%" comfy_batch_tool.py --port 8091
pause
endlocal
