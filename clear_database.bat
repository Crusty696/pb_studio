@echo off
REM ============================================================
REM PB_studio - Datenbank komplett bereinigen (FIXED VERSION)
REM ============================================================
REM WICHTIG: App MUSS geschlossen sein!
REM ============================================================

echo.
echo ============================================================
echo PB_studio - Datenbank bereinigen (FIXED)
echo ============================================================
echo.
echo WICHTIG: Stelle sicher dass die App GESCHLOSSEN ist!
echo.
echo Diese Version loescht:
echo  1. Alle Video-Clips aus der Datenbank
echo  2. WAL/SHM Cache-Dateien (verhindert alte Daten)
echo  3. VACUUM (komprimiert Datenbank)
echo.
pause

echo.
echo Bereinige Datenbank...
echo.

REM Verwende das neue, funktionierende Skript
python scripts\clear_database_now.py

echo.
echo ============================================================
echo Fertig!
echo ============================================================
echo.
echo Wenn Clips immernoch sichtbar sind:
echo  1. Schliesse die App KOMPLETT
echo  2. Fuehre dieses Script nochmal aus
echo  3. Starte die App neu
echo.
pause
