# Migration Rollback - Quick Reference

## Kommandozeile

### Backup-Info anzeigen
```bash
cd src/pb_studio/database/migrations
python test_rollback.py info
```

### Alle Backups auflisten
```bash
python test_rollback.py list
```

### Backup verifizieren
```bash
python test_rollback.py verify path/to/backup.db
```

### Rollback durchführen (v2 → v1)
```bash
# Mit interaktiver Bestätigung
python test_rollback.py rollback

# Direkter Rollback
python v2_video_analysis.py rollback
```

## Python API

### Automatischer Rollback
```python
from pb_studio.database.migrations import rollback_v2_to_v1

# Rollback v2 → v1
success = rollback_v2_to_v1()
```

### Backup-Informationen
```python
from pb_studio.database.migrations import (
    get_schema_version,
    list_backups,
    print_backup_info
)

# Aktuelle Version
version = get_schema_version("path/to/db.db")

# Alle Backups
backups = list_backups("path/to/db.db")
for path, version, timestamp in backups:
    print(f"v{version}: {timestamp}")

# Detaillierte Info
print_backup_info("path/to/db.db")
```

### Backup verifizieren
```python
from pb_studio.database.migrations import verify_backup

if verify_backup("path/to/backup.db"):
    print("Backup OK")
```

### Manueller Rollback
```python
from pb_studio.database.migrations import (
    rollback_to_backup,
    find_backup_for_version
)

# Finde v1 Backup
backup = find_backup_for_version("path/to/db.db", target_version=1)

# Rollback durchführen
if backup:
    success = rollback_to_backup("path/to/db.db", backup)
```

## Workflow

### Standard-Rollback-Prozess

1. **Status prüfen**
   ```bash
   python test_rollback.py info
   ```

2. **Backups prüfen**
   ```bash
   python test_rollback.py list
   ```

3. **Rollback durchführen**
   ```bash
   python test_rollback.py rollback
   ```

4. **Ergebnis prüfen**
   ```bash
   python test_rollback.py info
   ```

### Sicherheits-Workflow

Vor kritischer Operation:
```python
from pb_studio.database.migrations import create_backup

# Backup erstellen
backup_path = create_backup(db_path)

# ... Operation durchführen ...

# Bei Fehler: Rollback
if operation_failed:
    rollback_to_backup(db_path, backup_path)
```

## Fehlerbehebung

### Problem: "Kein Backup gefunden"
```bash
# Prüfe ob Backups existieren
ls -lh *.db | grep backup

# Falls keine: Migration neu durchführen
python v2_video_analysis.py
```

### Problem: "Backup beschädigt"
```bash
# Alle Backups verifizieren
for backup in *_backup_*.db; do
    python test_rollback.py verify "$backup"
done
```

### Problem: "Rollback schlägt fehl"
```bash
# 1. Log aktivieren
export PB_STUDIO_LOG_LEVEL=DEBUG

# 2. Rollback erneut versuchen
python test_rollback.py rollback

# 3. Manuelle Wiederherstellung
cp backup_file.db pb_studio.db
```

## Wichtige Hinweise

**Vor Rollback:**
- PB Studio beenden
- Aktuelle DB-Verbindungen schließen
- Backup-Existenz prüfen

**Während Rollback:**
- Nicht unterbrechen
- Automatisches Safety-Backup wird erstellt

**Nach Rollback:**
- Version prüfen
- Integrität prüfen
- Anwendung testen

## Support-Informationen sammeln

```bash
# System-Info
python --version
sqlite3 --version

# DB-Info
python test_rollback.py info

# Backup-Liste
python test_rollback.py list

# Logs
tail -n 100 pb_studio.log
```
