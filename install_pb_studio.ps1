# PB_studio Installer
# Vollautomatische Installation mit Dependency-Management
# Version: 1.0.0

#Requires -Version 5.1

param(
    [switch]$SkipFFmpeg,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Farben für Output
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"
$ColorInfo = "Cyan"

# Installer-Konfiguration
$INSTALLER_VERSION = "1.0.0"
$MIN_PYTHON_VERSION = "3.10"
$PYTHON_VERSION = "3.10"
$REQUIRED_DISK_SPACE_GB = 2

# Pfade
$INSTALL_DIR = $PSScriptRoot
$LOG_FILE = Join-Path $INSTALL_DIR "install.log"
$VENV_DIR = Join-Path $INSTALL_DIR ".venv"

# URLs für Downloads
$PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
$POETRY_INSTALL_URL = "https://install.python-poetry.org"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LOG_FILE -Value $logMessage -ErrorAction SilentlyContinue
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n[*] $Message" -ForegroundColor $ColorInfo
    Write-Log $Message "STEP"
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor $ColorSuccess
    Write-Log $Message "SUCCESS"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor $ColorWarning
    Write-Log $Message "WARNING"
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor $ColorError
    Write-Log $Message "ERROR"
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-DiskSpace {
    $drive = (Get-Item $INSTALL_DIR).PSDrive.Name
    $freeSpace = (Get-PSDrive $drive).Free / 1GB
    return $freeSpace -gt $REQUIRED_DISK_SPACE_GB
}

function Test-PythonVersion {
    param([string]$PythonPath)

    try {
        $versionOutput = & $PythonPath --version 2>&1
        if ($versionOutput -match "Python (\d+\.\d+)") {
            $version = [Version]$matches[1]
            $minVersion = [Version]$MIN_PYTHON_VERSION
            return $version -ge $minVersion
        }
    }
    catch {
        return $false
    }
    return $false
}

# ============================================================================
# INSTALLATION FUNCTIONS
# ============================================================================

function Install-Python {
    Write-Step "Python $PYTHON_VERSION wird installiert..."

    # Versuche zuerst winget
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  Versuche Installation via winget..."
        try {
            winget install Python.Python.$PYTHON_VERSION --silent --accept-package-agreements --accept-source-agreements
            Write-Success "Python via winget installiert"
            return $true
        }
        catch {
            Write-Warning "winget Installation fehlgeschlagen, versuche direkten Download..."
        }
    }

    # Direkter Download als Fallback
    Write-Host "  Lade Python Installer herunter..."
    $installerPath = Join-Path $env:TEMP "python_installer.exe"

    try {
        Invoke-WebRequest -Uri $PYTHON_INSTALLER_URL -OutFile $installerPath -UseBasicParsing
        Write-Host "  Fuehre Python Installer aus..."
        Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0" -Wait
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        Write-Success "Python installiert"
        return $true
    }
    catch {
        Write-ErrorMsg "Python Installation fehlgeschlagen: $_"
        return $false
    }
}

function Install-Poetry {
    Write-Step "Poetry wird installiert..."

    try {
        $installScript = Invoke-WebRequest -Uri $POETRY_INSTALL_URL -UseBasicParsing
        $installScript.Content | python -

        # Fuege Poetry zu PATH hinzu
        $poetryPath = Join-Path $env:APPDATA "Python\Scripts"
        if ($env:PATH -notlike "*$poetryPath*") {
            $env:PATH = "$poetryPath;$env:PATH"
            [Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")
        }

        Write-Success "Poetry installiert"
        return $true
    }
    catch {
        Write-ErrorMsg "Poetry Installation fehlgeschlagen: $_"
        return $false
    }
}

function Install-FFmpeg {
    Write-Step "FFmpeg wird geprueft/installiert..."

    # Pruefe ob FFmpeg bereits vorhanden
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Success "FFmpeg ist bereits installiert"
        return $true
    }

    # Versuche winget Installation
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  Installiere FFmpeg via winget..."
        try {
            winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
            Write-Success "FFmpeg via winget installiert"
            return $true
        }
        catch {
            Write-Warning "winget Installation fehlgeschlagen"
        }
    }

    # Manuelle Anleitung als Fallback
    Write-Warning "FFmpeg konnte nicht automatisch installiert werden"
    Write-Host "`nBitte installiere FFmpeg manuell:"
    Write-Host "1. Besuche: https://www.gyan.dev/ffmpeg/builds/"
    Write-Host "2. Download 'ffmpeg-release-essentials.zip'"
    Write-Host "3. Entpacke und fuege 'bin' Ordner zu PATH hinzu"
    Write-Host "`nOder fuehre aus: winget install Gyan.FFmpeg"
    return $false
}

function Install-Dependencies {
    Write-Step "Installiere Python-Dependencies..."

    try {
        # Wechsle in Install-Verzeichnis
        Push-Location $INSTALL_DIR

        # Poetry install
        Write-Host "  Fuehre 'poetry install' aus (kann einige Minuten dauern)..."
        poetry install --no-interaction

        Pop-Location
        Write-Success "Dependencies installiert"
        return $true
    }
    catch {
        Pop-Location
        Write-ErrorMsg "Dependency Installation fehlgeschlagen: $_"
        return $false
    }
}

function New-DesktopShortcut {
    Write-Step "Erstelle Desktop-Verknuepfung..."

    try {
        $WshShell = New-Object -ComObject WScript.Shell
        $desktopPath = [Environment]::GetFolderPath("Desktop")
        $shortcutPath = Join-Path $desktopPath "PB_studio.lnk"
        $startScriptPath = Join-Path $INSTALL_DIR "START_PB_STUDIO.bat"

        $Shortcut = $WshShell.CreateShortcut($shortcutPath)
        $Shortcut.TargetPath = $startScriptPath
        $Shortcut.WorkingDirectory = $INSTALL_DIR
        $Shortcut.Description = "PB_studio - Precision Beat Video Studio"

        # Icon setzen falls vorhanden
        $iconPath = Join-Path $INSTALL_DIR "icon.ico"
        if (Test-Path $iconPath) {
            $Shortcut.IconLocation = $iconPath
        }

        $Shortcut.Save()
        Write-Success "Desktop-Verknuepfung erstellt"
        return $true
    }
    catch {
        Write-Warning "Desktop-Verknuepfung konnte nicht erstellt werden: $_"
        return $false
    }
}

function New-StartMenuEntry {
    Write-Step "Erstelle Startmenue-Eintrag..."

    try {
        $WshShell = New-Object -ComObject WScript.Shell
        $startMenuPath = Join-Path ([Environment]::GetFolderPath("Programs")) "PB_studio"

        if (-not (Test-Path $startMenuPath)) {
            New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
        }

        $shortcutPath = Join-Path $startMenuPath "PB_studio.lnk"
        $startScriptPath = Join-Path $INSTALL_DIR "START_PB_STUDIO.bat"

        $Shortcut = $WshShell.CreateShortcut($shortcutPath)
        $Shortcut.TargetPath = $startScriptPath
        $Shortcut.WorkingDirectory = $INSTALL_DIR
        $Shortcut.Description = "PB_studio - Precision Beat Video Studio"
        $Shortcut.Save()

        Write-Success "Startmenue-Eintrag erstellt"
        return $true
    }
    catch {
        Write-Warning "Startmenue-Eintrag konnte nicht erstellt werden: $_"
        return $false
    }
}

# ============================================================================
# MAIN INSTALLATION
# ============================================================================

function Start-Installation {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor $ColorInfo
    Write-Host "  PB_studio Installer v$INSTALLER_VERSION" -ForegroundColor $ColorInfo
    Write-Host "  Precision Beat Video Studio" -ForegroundColor $ColorInfo
    Write-Host "============================================================" -ForegroundColor $ColorInfo
    Write-Host ""

    # Initialisiere Log
    "Installation gestartet: $(Get-Date)" | Out-File $LOG_FILE -Force
    Write-Log "Installer Version: $INSTALLER_VERSION" "INFO"

    # Pruefe System-Requirements
    Write-Step "Pruefe System-Anforderungen..."

    if (-not (Test-DiskSpace)) {
        Write-ErrorMsg "Nicht genug Speicherplatz! Mindestens $REQUIRED_DISK_SPACE_GB GB erforderlich."
        return $false
    }
    Write-Success "Speicherplatz ausreichend"

    # Python pruefen/installieren
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Path

    if ($pythonPath -and (Test-PythonVersion $pythonPath)) {
        Write-Success "Python $MIN_PYTHON_VERSION+ ist bereits installiert"
    }
    else {
        Write-Warning "Python $MIN_PYTHON_VERSION+ nicht gefunden"
        if (-not (Install-Python)) {
            Write-ErrorMsg "Installation kann ohne Python nicht fortgesetzt werden"
            return $false
        }

        # PATH neu laden
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    }

    # Poetry pruefen/installieren
    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        if (-not (Install-Poetry)) {
            Write-ErrorMsg "Installation kann ohne Poetry nicht fortgesetzt werden"
            return $false
        }
    }
    else {
        Write-Success "Poetry ist bereits installiert"
    }

    # FFmpeg pruefen/installieren (optional)
    if (-not $SkipFFmpeg) {
        Install-FFmpeg | Out-Null
    }

    # Dependencies installieren
    if (-not (Install-Dependencies)) {
        Write-ErrorMsg "Dependency Installation fehlgeschlagen"
        return $false
    }

    # Shortcuts erstellen
    New-DesktopShortcut | Out-Null
    New-StartMenuEntry | Out-Null

    return $true
}

# ============================================================================
# EXECUTION
# ============================================================================

try {
    $success = Start-Installation

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor $ColorInfo

    if ($success) {
        Write-Host "  Installation erfolgreich abgeschlossen!" -ForegroundColor $ColorSuccess
        Write-Host "============================================================" -ForegroundColor $ColorInfo
        Write-Host ""
        Write-Host "PB_studio wurde installiert!" -ForegroundColor $ColorSuccess
        Write-Host ""
        Write-Host "Starte die App:"
        Write-Host "  - Desktop-Icon: 'PB_studio'" -ForegroundColor $ColorInfo
        Write-Host "  - Oder: START_PB_STUDIO.bat ausfuehren" -ForegroundColor $ColorInfo
        Write-Host ""
        Write-Host "Dokumentation: README_INSTALLATION.md"
        Write-Host "Bei Problemen: TROUBLESHOOTING.md"
        Write-Log "Installation erfolgreich abgeschlossen" "SUCCESS"
    }
    else {
        Write-Host "  Installation fehlgeschlagen!" -ForegroundColor $ColorError
        Write-Host "============================================================" -ForegroundColor $ColorInfo
        Write-Host ""
        Write-Host "Bitte pruefe die Log-Datei: $LOG_FILE" -ForegroundColor $ColorWarning
        Write-Host "Oder siehe: TROUBLESHOOTING.md" -ForegroundColor $ColorWarning
        Write-Log "Installation fehlgeschlagen" "ERROR"
    }

    Write-Host ""
    Read-Host "Druecke Enter zum Beenden"
}
catch {
    Write-ErrorMsg "Unerwarteter Fehler: $_"
    Write-Log "Kritischer Fehler: $_" "CRITICAL"
    Write-Host ""
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}
