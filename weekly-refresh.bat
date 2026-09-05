@echo off
setlocal
rem Weekly in-season refresh, driven by Windows Task Scheduler.
rem Usage: weekly-refresh.bat [tuesday|thursday|sunday]
rem
rem Tuesday   : rebuild feature pipeline, backfill last week's actuals,
rem             warm caches, refresh rankings, project week N, build
rem             rest-of-season projections + ROS rankings, log snapshot.
rem Thursday  : refresh rankings, project week N, log snapshot.
rem Sunday    : refresh rankings, project week N, log snapshot.
rem
rem Current NFL week is resolved live from Sleeper's public state endpoint
rem (https://api.sleeper.app/v1/state/nfl) -- no manual week bookkeeping.
rem Every step logs to data\outputs\logs\.

cd /d "%~dp0"

set RUN_LABEL=%1
if "%RUN_LABEL%"=="" (
    echo Usage: weekly-refresh.bat [tuesday^|thursday^|sunday]
    exit /b 1
)

set SEASON=2026
set LEAGUES=rogan-radinator-league bdff-chopped

set LOG_DIR=data\outputs\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_FILE=%LOG_DIR%\weekly_%RUN_LABEL%_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%.log
set LOG_FILE=%LOG_FILE: =0%

echo ============================================ > "%LOG_FILE%"
echo  Weekly refresh: %RUN_LABEL% (%date% %time%) >> "%LOG_FILE%"
echo ============================================ >> "%LOG_FILE%"

for /f "usebackq delims=" %%w in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get-current-week.ps1"`) do set WEEK=%%w
if "%WEEK%"=="" (
    echo FAILED: could not resolve current NFL week from Sleeper's state endpoint >> "%LOG_FILE%"
    goto :end
)
echo Current NFL week: %WEEK% >> "%LOG_FILE%"
set /a PREV_WEEK=%WEEK%-1

if /I "%RUN_LABEL%"=="tuesday" (
    echo. >> "%LOG_FILE%"
    echo [Tuesday] rebuilding feature pipeline... >> "%LOG_FILE%"
    uv run python notebooks\build_features_pipeline.py >> "%LOG_FILE%" 2>&1
    if errorlevel 1 echo   WARNING: feature pipeline rebuild failed, continuing anyway >> "%LOG_FILE%"

    echo. >> "%LOG_FILE%"
    echo [Tuesday] warming Sleeper cache for all leagues... >> "%LOG_FILE%"
    uv run ffapp cache warm --season %SEASON% --all-leagues --no-offline >> "%LOG_FILE%" 2>&1
)

for %%L in (%LEAGUES%) do (
    echo. >> "%LOG_FILE%"
    echo [%RUN_LABEL%] %%L: refreshing rankings/ADP... >> "%LOG_FILE%"
    uv run ffapp ingest rankings --league %%L --season %SEASON% --no-offline >> "%LOG_FILE%" 2>&1

    echo [%RUN_LABEL%] %%L: projecting week %WEEK%... >> "%LOG_FILE%"
    uv run ffapp project --week %WEEK% --season %SEASON% --league %%L --no-offline >> "%LOG_FILE%" 2>&1

    if /I "%RUN_LABEL%"=="tuesday" (
        if %PREV_WEEK% GEQ 1 (
            echo [Tuesday] %%L: backfilling week %PREV_WEEK% actuals... >> "%LOG_FILE%"
            uv run ffapp log backfill --week %PREV_WEEK% --season %SEASON% --league %%L >> "%LOG_FILE%" 2>&1
        )

        echo [Tuesday] %%L: rest-of-season projections... >> "%LOG_FILE%"
        uv run ffapp project --week %WEEK% --from-week %WEEK% --through-week 18 --season %SEASON% --league %%L --no-offline >> "%LOG_FILE%" 2>&1

        echo [Tuesday] %%L: rest-of-season rankings... >> "%LOG_FILE%"
        uv run ffapp rankings ros --season %SEASON% --league %%L --no-offline >> "%LOG_FILE%" 2>&1
    )

    echo [%RUN_LABEL%] %%L: logging week %WEEK% snapshot, label=%RUN_LABEL%... >> "%LOG_FILE%"
    uv run ffapp log week --week %WEEK% --run-label %RUN_LABEL% --season %SEASON% --league %%L --no-offline >> "%LOG_FILE%" 2>&1
)

echo. >> "%LOG_FILE%"
echo Weekly refresh (%RUN_LABEL%) finished %date% %time% >> "%LOG_FILE%"

:end
endlocal
