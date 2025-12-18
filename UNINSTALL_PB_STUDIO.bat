@echo off
REM =========================================================================
REM PB_studio Uninstaller (Windows)
REM =========================================================================
REM
REM Entfernt PB_studio komplett vom System:
REM - Virtual Environment
REM - Desktop-Verknuepfung
REM - Startmenue-Eintrag
REM
REM HINWEIS: Projektdateien und Videos bleiben erhalten!
REM
REM =========================================================================

echo.
echo ========================================================================
echo PB_studio Deinstallation
echo ========================================================================
echo.
echo WARNUNG: Dies wird PB_studio vollstaendig entfernen!
echo.
echo Die folgenden Dinge werden geloescht:
echo   - Python Virtual Environment (.venv)
echo   - Desktop-Verknuepfung
echo   - Startmenue-Eintrag
echo.
echo BLEIBEN ERHALTEN:
echo   - Projektdateien
echo   - Gerenderte Videos
echo   - Test-Clips
echo   - Einstellungen
echo.
set /p CONFIRM="Moechtest du fortfahren? (J/N): "

if /i not "%CONFIRM%"=="J" (
    echo.
    echo Deinstallation abgebrochen.
    pause
    exit /b 0
)

echo.
echo ========================================================================
echo Starte Deinstallation...
echo ========================================================================
echo.

REM Wechsle in Projektverzeichnis
cd /d "%~dp0"

REM Entferne Virtual Environment
echo [1/3] Entferne Virtual Environment...
if exist ".venv" (
    rmdir /s /q ".venv"
    echo   [OK] Virtual Environment entfernt
) else (
    echo   [INFO] Virtual Environment nicht gefunden
)

REM Entferne Desktop-Verknuepfung
echo.
echo [2/3] Entferne Desktop-Verknuepfung...
set DESKTOP=%USERPROFILE%\Desktop
if exist "%DESKTOP%\PB_studio.lnk" (
    del /q "%DESKTOP%\PB_studio.lnk"
    echo   [OK] Desktop-Verknuepfung entfernt
) else (
    echo   [INFO] Desktop-Verknuepfung nicht gefunden
)

REM Entferne Startmenue-Eintrag
echo.
echo [3/3] Entferne Startmenue-Eintrag...
set STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\PB_studio
if exist "%STARTMENU%" (
    rmdir /s /q "%STARTMENU%"
    echo   [OK] Startmenue-Eintrag entfernt
) else (
    echo   [INFO] Startmenue-Eintrag nicht gefunden
)

echo.
echo ========================================================================
echo Deinstallation abgeschlossen!
echo ========================================================================
echo.
echo PB_studio wurde entfernt.
echo.
echo Projektdateien und Videos bleiben erhalten.
echo Um diese zu loeschen, loesche manuell:
echo   - %CD%
echo.
pause
