"""
Test-Skript fuer Migration Rollback

Verwendung:
    python test_rollback.py info          # Zeigt Backup-Info
    python test_rollback.py list          # Listet Backups
    python test_rollback.py verify <path> # Verifiziert Backup
    python test_rollback.py rollback      # Fuehrt Rollback durch
"""

import sys
from pathlib import Path

from ..connection import get_db_manager
from .rollback import (
    get_schema_version,
    list_backups,
    print_backup_info,
    verify_backup,
)
from .v2_video_analysis import rollback_v2_to_v1


def show_info():
    """Zeige Datenbank-Info."""
    db_manager = get_db_manager()
    db_path = str(db_manager.db_path)

    print("\n=== Datenbank Info ===")
    print(f"DB-Pfad: {db_path}")
    print(f"Existiert: {Path(db_path).exists()}")

    if Path(db_path).exists():
        version = get_schema_version(db_path)
        print(f"Schema-Version: {version}")

    print_backup_info(db_path)


def list_all_backups():
    """Liste alle Backups."""
    db_manager = get_db_manager()
    db_path = str(db_manager.db_path)

    backups = list_backups(db_path)

    if not backups:
        print("\nKeine Backups gefunden.")
        return

    print(f"\n=== Gefundene Backups ({len(backups)}) ===")
    for i, (backup_path, version, timestamp) in enumerate(backups, 1):
        backup_file = Path(backup_path)
        size_mb = backup_file.stat().st_size / (1024 * 1024)

        print(f"\n{i}. Backup:")
        print(f"   Pfad:     {backup_path}")
        print(f"   Version:  {version}")
        print(f"   Datum:    {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Groesse:  {size_mb:.2f} MB")


def verify_specific_backup(backup_path: str):
    """Verifiziere spezifisches Backup."""
    print("\n=== Backup-Verifikation ===")
    print(f"Backup: {backup_path}")

    if not Path(backup_path).exists():
        print("FEHLER: Backup-Datei existiert nicht!")
        return False

    print("Pruefe Integritaet...")
    is_valid = verify_backup(backup_path)

    if is_valid:
        print("✓ Backup ist gueltig")

        # Zeige Version
        version = get_schema_version(backup_path)
        print(f"  Schema-Version: {version}")

    else:
        print("✗ Backup ist ungueltig oder beschaedigt")

    return is_valid


def perform_rollback():
    """Fuehre Rollback durch."""
    db_manager = get_db_manager()
    db_path = str(db_manager.db_path)

    current_version = get_schema_version(db_path)

    print("\n=== Rollback ===")
    print(f"Aktuelle Version: {current_version}")

    if current_version != 2:
        print(f"FEHLER: Kann nur v2 zurueckrollen (aktuelle Version: {current_version})")
        return False

    # Warnung
    print("\nWARNUNG: Dieser Vorgang stellt die Datenbank aus einem Backup wieder her.")
    print("         Alle Aenderungen seit dem Backup gehen verloren!")
    print("\nBackups verfuegbar:")

    backups = list_backups(db_path)
    v1_backups = [(p, v, t) for p, v, t in backups if v == 1]

    if not v1_backups:
        print("  Keine v1 Backups gefunden!")
        return False

    for backup_path, version, timestamp in v1_backups[:3]:  # Zeige max 3
        print(f"  - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Bestaetigung
    response = input("\nRollback durchfuehren? (ja/nein): ").strip().lower()

    if response not in ["ja", "j", "yes", "y"]:
        print("Abgebrochen.")
        return False

    print("\nFuehre Rollback durch...")
    success = rollback_v2_to_v1()

    if success:
        print("\n✓ Rollback erfolgreich!")

        # Pruefe Version
        new_version = get_schema_version(db_path)
        print(f"  Neue Version: {new_version}")

    else:
        print("\n✗ Rollback fehlgeschlagen!")

    return success


def main():
    """Hauptfunktion."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "info":
        show_info()

    elif command == "list":
        list_all_backups()

    elif command == "verify":
        if len(sys.argv) < 3:
            print("FEHLER: Backup-Pfad erforderlich")
            print("Verwendung: python test_rollback.py verify <backup_path>")
            sys.exit(1)

        backup_path = sys.argv[2]
        success = verify_specific_backup(backup_path)
        sys.exit(0 if success else 1)

    elif command == "rollback":
        success = perform_rollback()
        sys.exit(0 if success else 1)

    else:
        print(f"FEHLER: Unbekannter Befehl '{command}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
