@echo off
setlocal

rem Always run from the repo root, regardless of where this was launched from.
cd /d "%~dp0"

echo ============================================
echo  Starting fantasyfootball app
echo  http://localhost:8501
echo ============================================
echo.
echo Close this window (or press Ctrl+C) to stop the app.
echo.

rem Streamlit's real first-run behavior (confirmed live, not assumed): with
rem no ~/.streamlit/credentials.toml yet, it prints a "Welcome to Streamlit /
rem enter your email" prompt and blocks on stdin forever -- no visible error,
rem the process just sits there. STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
rem does NOT suppress this specific prompt (tested; only affects telemetry
rem after the fact). Pre-creating an empty credentials.toml is the real,
rem one-time fix -- once it exists, every future run (from this script or
rem anywhere else) skips the prompt permanently.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    (
        echo [general]
        echo email = ""
    ) > "%USERPROFILE%\.streamlit\credentials.toml"
)
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

rem Laptop-only: bind to localhost, not 0.0.0.0 -- no LAN/network exposure.
rem --server.headless false makes the browser tab open automatically.
uv run streamlit run src\ffapp\app\streamlit_app.py --server.address localhost --server.port 8501 --server.headless false

if errorlevel 1 (
    echo.
    echo ============================================
    echo  FAILED to start the app ^(exit code %errorlevel%^)
    echo  Common causes: uv not on PATH, port 8501 already
    echo  in use, or a real error in the app itself -- see
    echo  the output above.
    echo ============================================
    pause
    exit /b 1
)

endlocal
