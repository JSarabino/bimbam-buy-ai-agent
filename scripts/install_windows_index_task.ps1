param(
    [string]$TaskName = "BimBam Assistant - Index Maintenance",
    [ValidatePattern("^([01]\d|2[0-3]):[0-5]\d$")]
    [string]$DailyAt = "03:00"
)

$ErrorActionPreference = "Stop"

$RunnerPath = Join-Path `
    $PSScriptRoot `
    "run_index_maintenance.ps1"

if (-not (Test-Path $RunnerPath)) {
    throw "No se encontró el ejecutor: $RunnerPath"
}

$TaskCommand = (
    'powershell.exe -NoProfile -ExecutionPolicy Bypass ' +
    '-File "' + $RunnerPath + '"'
)

& schtasks.exe `
    /Create `
    /SC DAILY `
    /ST $DailyAt `
    /TN $TaskName `
    /TR $TaskCommand `
    /F

if ($LASTEXITCODE -ne 0) {
    throw "No fue posible crear la tarea programada."
}

Write-Host ""
Write-Host "Tarea creada correctamente."
Write-Host "Nombre: $TaskName"
Write-Host "Horario diario: $DailyAt"
Write-Host ""
Write-Host "Para ejecutarla ahora:"
Write-Host "schtasks.exe /Run /TN `"$TaskName`""
Write-Host ""
Write-Host "Para eliminarla:"
Write-Host "schtasks.exe /Delete /TN `"$TaskName`" /F"
