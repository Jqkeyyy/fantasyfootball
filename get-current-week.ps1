# Prints the current NFL week from Sleeper's public state endpoint.
# Called by weekly-refresh.bat -- kept as its own file so the batch script's
# `for /f` clause doesn't have to parse nested parentheses from an inline
# PowerShell command.
(Invoke-RestMethod -Uri 'https://api.sleeper.app/v1/state/nfl').week
