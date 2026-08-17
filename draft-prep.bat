@echo off
setlocal

rem Always run from the repo root, regardless of where this was launched from.
cd /d "%~dp0"

echo ============================================
echo  Draft-day prep: rogan-radinator-league
echo ============================================
echo.

echo [1/2] Refreshing rankings/ADP sources (live)...
echo.
uv run ffapp ingest rankings --league rogan-radinator-league --no-offline
if errorlevel 1 (
    echo.
    echo ==== FAILED: refreshing rankings ^(exit code %errorlevel%^) ====
    goto :end
)

echo.
echo [2/2] Regenerating the draft board...
echo.
uv run ffapp draft board --league rogan-radinator-league
if errorlevel 1 (
    echo.
    echo ==== FAILED: regenerating draft board ^(exit code %errorlevel%^) ====
    goto :end
)

echo.
echo ============================================
echo  Draft prep complete. Board CSV path is
echo  printed above ^("Wrote N players to ..."^).
echo ============================================

:end
echo.
pause
endlocal
