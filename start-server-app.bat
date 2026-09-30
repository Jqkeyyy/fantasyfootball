@echo off
setlocal
cd /d "%~dp0"

rem The server address lives in .env (FFAPP_SERVER_HOST) so it stays out of the repo.
set SERVER_HOST=%FFAPP_SERVER_HOST%
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if /I "%%a"=="FFAPP_SERVER_HOST" set SERVER_HOST=%%b
    )
)
if "%SERVER_HOST%"=="" (
    echo Set FFAPP_SERVER_HOST in .env to the home server's address first.
    pause
    exit /b 1
)

echo ============================================
echo  Connecting to the fantasyfootball server
echo  Dashboard: http://localhost:8502
echo ============================================
echo.
echo Keep this window open while using the dashboard.
echo Press Ctrl+C or close it to disconnect.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:8502'"
ssh -N -o ExitOnForwardFailure=yes -L 8502:127.0.0.1:8501 %SERVER_HOST%

if errorlevel 1 (
    echo.
    echo Could not connect to the home server at %SERVER_HOST%.
    pause
    exit /b 1
)

endlocal
