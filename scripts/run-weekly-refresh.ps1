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

Push-Location $ProjectRoot
try {
    & uv run ffapp refresh weekly --run-label $RunLabel --no-offline *>> $LogPath
    $ExitCode = $LASTEXITCODE
    $Manifest = Get-ChildItem -Path "data\outputs\*\refresh_runs\latest.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $Manifest) {
        $Result = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
        if ($Result.status -ne "healthy") {
            & msg.exe $env:USERNAME "FFApp $RunLabel refresh: $($Result.status). See $($Manifest.FullName)."
        }
    }
    $AlertFile = Get-ChildItem -Path "data\outputs\*\alerts\latest.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $AlertFile) {
        $AlertResult = Get-Content -LiteralPath $AlertFile.FullName -Raw | ConvertFrom-Json
        if ($AlertResult.alerts.Count -gt 0) {
            $Preview = ($AlertResult.alerts | Select-Object -First 3 | ForEach-Object { $_.message }) -join " | "
            & msg.exe $env:USERNAME "FFApp has $($AlertResult.alerts.Count) new decision alert(s): $Preview"
        }
    }
    exit $ExitCode
}
finally {
    Pop-Location
}
