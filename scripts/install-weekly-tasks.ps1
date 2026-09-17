$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run-weekly-refresh.ps1"
$TaskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$CurrentPrincipal = New-Object System.Security.Principal.WindowsPrincipal(
    [System.Security.Principal.WindowsIdentity]::GetCurrent()
)
$IsAdministrator = $CurrentPrincipal.IsInRole(
    [System.Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $IsAdministrator) {
    throw "Installing S4U tasks requires an elevated PowerShell. Re-run this script as Administrator."
}

$Schedules = @(
    @{ Name = "FFApp Weekly Tuesday"; Day = "Tuesday"; Time = "07:00"; Label = "tuesday" },
    @{ Name = "FFApp Weekly Thursday"; Day = "Thursday"; Time = "07:00"; Label = "thursday" },
    @{ Name = "FFApp Weekly Sunday"; Day = "Sunday"; Time = "08:00"; Label = "sunday" }
)

foreach ($Schedule in $Schedules) {
    $Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Runner`" -RunLabel $($Schedule.Label)"
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arguments
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Schedule.Day -At $Schedule.Time
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $Principal = New-ScheduledTaskPrincipal `
        -UserId $TaskUser `
        -LogonType S4U `
        -RunLevel Limited
    Register-ScheduledTask `
        -TaskName $Schedule.Name `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Refresh fantasy-football recommendations and alert on degraded runs." `
        -Force `
        -ErrorAction Stop | Out-Null
    Write-Host "Installed $($Schedule.Name) at $($Schedule.Time)."
}
