@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set "PYTHONUTF8=1"

:: ============================================================================
:: PB Studio - Robuster Starter
:: Unterstützt: .venv, Poetry, System-Python
:: ============================================================================

title PB Studio - Starter
cd /d "%~dp0"

echo.
echo ============================================
echo   PB Studio - Precision Beat Video Studio
echo ============================================
echo.

:: Log-Verzeichnis
if not exist logs mkdir logs

:: ----------------------------------------------------------------------------
:: 1. Prüfe .venv (bevorzugt)
:: ----------------------------------------------------------------------------
if exist ".venv\Scripts\activate.bat" (
    echo [OK] Virtuelle Umgebung gefunden: .venv
    echo      Aktiviere .venv...
    call .venv\Scripts\activate.bat

    echo     :: Starte die App via main.py (robuster Entry Point)
echo      Starte PB Studio...
    python main.py
    if errorlevel 1 goto error
    goto end"EXITCODE=!ERRORLEVEL!"
    goto :done
)

:: ----------------------------------------------------------------------------
:: 2. Prüfe Poetry
:: ----------------------------------------------------------------------------
echo [INFO] Kein .venv gefunden, prüfe Poetry...

where poetry >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Poetry gefunden
    echo      Starte mit Poetry...
    echo.
    poetry run python start_app.py
    set "EXITCODE=!ERRORLEVEL!"
    goto :done
)

:: Prüfe python -m poetry
python -m poetry --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Poetry (via Python) gefunden
    echo      Starte mit Poetry...
    echo.
    python -m poetry run python main.py
    set "EXITCODE=!ERRORLEVEL!"
    goto :done
)

:: ----------------------------------------------------------------------------
:: 3. Fallback: System-Python direkt
:: ----------------------------------------------------------------------------
echo [WARNUNG] Weder .venv noch Poetry gefunden
echo           Versuche System-Python...
echo.

:: Prüfe py -3
py -3 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python Launcher (py -3) gefunden
    py -3 main.py
    set "EXITCODE=!ERRORLEVEL!"
    goto :done
)

:: Prüfe python
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python gefunden
    python start_app.py
    set "EXITCODE=!ERRORLEVEL!"
    goto :done
)

:: ----------------------------------------------------------------------------
:: Fehler: Kein Python gefunden
:: ----------------------------------------------------------------------------
echo.
echo ============================================
echo   FEHLER: Python nicht gefunden!
echo ============================================
echo.
echo   Bitte installiere Python 3.10+:
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1

:: ----------------------------------------------------------------------------
:: Ende
:: ----------------------------------------------------------------------------
:done
echo.
echo ============================================
if %EXITCODE% EQU 0 (
    echo   [OK] PB Studio beendet
) else (
    echo   [FEHLER] Exit-Code: %EXITCODE%
)
echo ============================================
echo.

if %EXITCODE% NEQ 0 pause
exit /b %EXITCODE%