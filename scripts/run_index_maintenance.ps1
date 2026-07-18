param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\python.exe"
$IndexScript = Join-Path $ProjectRoot "scripts\index_documents.py"
$LogDirectory = Join-Path $ProjectRoot "storage\maintenance"
$LogPath = Join-Path `
    $LogDirectory `
    ("index-maintenance-{0}.log" -f (Get-Date -Format "yyyyMMdd"))

if (-not (Test-Path $PythonPath)) {
    throw "No se encontró Python en: $PythonPath"
}

if (-not (Test-Path $IndexScript)) {
    throw "No se encontró el script de indexación: $IndexScript"
}

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force | Out-Null

$Arguments = @(
    $IndexScript
)

if ($Force) {
    $Arguments += "--force"
}

Push-Location $ProjectRoot

$ExitCode = 1

try {
    "[$(Get-Date -Format o)] Inicio del mantenimiento." |
        Tee-Object -FilePath $LogPath -Append

    # Windows PowerShell 5.1 convierte la salida STDERR de procesos
    # nativos en NativeCommandError. Python logging escribe en STDERR
    # por defecto incluso para mensajes INFO. Se permite temporalmente
    # esa salida y se transforma en texto normal antes de registrarla.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $PythonPath @Arguments 2>&1 |
            ForEach-Object {
                $_.ToString()
            } |
            Tee-Object -FilePath $LogPath -Append

        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    "[$(Get-Date -Format o)] Fin del mantenimiento. Código: $ExitCode" |
        Tee-Object -FilePath $LogPath -Append
}
catch {
    $_.ToString() |
        Tee-Object -FilePath $LogPath -Append

    $ExitCode = 1
}
finally {
    Pop-Location
}

exit $ExitCode
