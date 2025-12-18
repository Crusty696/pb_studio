#Requires -Version 5.1
<#
.SYNOPSIS
    PB Studio - PowerShell Starter

.DESCRIPTION
    Robuster Starter fuer PB Studio.
    Unterstuetzt: .venv, Poetry, System-Python

.EXAMPLE
    .\Start-PBStudio.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "PB Studio - Starter"

# Ins Projektverzeichnis wechseln
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  PB Studio - Precision Beat Video Studio" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Log-Verzeichnis erstellen
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$exitCode = 0

# ----------------------------------------------------------------------------
# 1. Pruefe .venv (bevorzugt)
# ----------------------------------------------------------------------------
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[OK] Virtuelle Umgebung gefunden: .venv" -ForegroundColor Green
    Write-Host "     Aktiviere .venv..."

    & ".venv\Scripts\Activate.ps1"

    Write-Host "     Starte PB Studio..."
    Write-Host ""

    python start_app.py
    $exitCode = $LASTEXITCODE
}
# ----------------------------------------------------------------------------
# 2. Pruefe Poetry
# ----------------------------------------------------------------------------
elseif (Get-Command poetry -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Poetry gefunden" -ForegroundColor Green
    Write-Host "     Starte mit Poetry..."
    Write-Host ""

    poetry run python start_app.py
    $exitCode = $LASTEXITCODE
}
elseif (python -m poetry --version 2>$null) {
    Write-Host "[OK] Poetry (via Python) gefunden" -ForegroundColor Green
    Write-Host "     Starte mit Poetry..."
    Write-Host ""

    python -m poetry run python start_app.py
    $exitCode = $LASTEXITCODE
}
# ----------------------------------------------------------------------------
# 3. Fallback: System-Python direkt
# ----------------------------------------------------------------------------
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "[WARNUNG] Weder .venv noch Poetry gefunden" -ForegroundColor Yellow
    Write-Host "          Versuche Python Launcher (py)..."
    Write-Host ""

    py -3 start_app.py
    $exitCode = $LASTEXITCODE
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "[WARNUNG] Weder .venv noch Poetry gefunden" -ForegroundColor Yellow
    Write-Host "          Versuche System-Python..."
    Write-Host ""

    python start_app.py
    $exitCode = $LASTEXITCODE
}
# ----------------------------------------------------------------------------
# Fehler: Kein Python gefunden
# ----------------------------------------------------------------------------
else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  FEHLER: Python nicht gefunden!" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Bitte installiere Python 3.10+:"
    Write-Host "  https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "  Oder erstelle eine virtuelle Umgebung:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""

    Read-Host "Druecke Enter zum Schliessen"
    exit 1
}

# ----------------------------------------------------------------------------
# Ende
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "  [OK] PB Studio beendet" -ForegroundColor Green
} else {
    Write-Host "  [FEHLER] Exit-Code: $exitCode" -ForegroundColor Red
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($exitCode -ne 0) {
    Read-Host "Druecke Enter zum Schliessen"
}

exit $exitCode
