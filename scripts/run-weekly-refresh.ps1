param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("tuesday", "thursday", "sunday")]
    [string]$RunLabel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDirectory = Join-Path $ProjectRoot "data\outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory "scheduled-refresh-$RunLabel.log"

function Send-LocalMessage {
    param([string]$Message)
    $Messenger = Get-Command msg.exe -ErrorAction SilentlyContinue
    if ($null -eq $Messenger) {
        Add-Content -LiteralPath $LogPath -Value "Notification unavailable: $Message"
        return
    }
    try {
        & $Messenger.Source $env:USERNAME $Message 2>> $LogPath
    }
    catch {
        Add-Content -LiteralPath $LogPath -Value "Notification failed: $($_.Exception.Message)"
    }
}

Push-Location $ProjectRoot
try {
    & uv run ffapp refresh weekly --all-leagues --run-label $RunLabel --no-offline *>> $LogPath
    $ExitCode = $LASTEXITCODE
    $Unhealthy = @()
    foreach ($Manifest in Get-ChildItem -Path "data\outputs\*\refresh_runs\latest.json") {
        $Result = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
        if ($Result.status -ne "healthy") {
            $Unhealthy += "$($Result.league_slug): $($Result.status)"
        }
    }
    if ($Unhealthy.Count -gt 0) {
        Send-LocalMessage "FFApp $RunLabel refresh issue(s): $($Unhealthy -join ', '). See $LogPath."
    }

    $NewAlerts = @()
    foreach ($AlertFile in Get-ChildItem -Path "data\outputs\*\alerts\latest.json") {
        $AlertResult = Get-Content -LiteralPath $AlertFile.FullName -Raw | ConvertFrom-Json
        if ($AlertResult.alerts.Count -gt 0) {
            $NewAlerts += $AlertResult.alerts
        }
    }
    if ($NewAlerts.Count -gt 0) {
        $Preview = ($NewAlerts | Select-Object -First 3 | ForEach-Object { $_.message }) -join " | "
        Send-LocalMessage "FFApp has $($NewAlerts.Count) new decision alert(s): $Preview"
    }
    exit $ExitCode
}
finally {
    Pop-Location
}
