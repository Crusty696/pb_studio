"""
Database Migrations Package fuer PB_studio

Verfuegbare Funktionen:
- migrate_to_v2(): Migration auf Schema v2
- rollback_v2_to_v1(): Rollback von v2 zu v1
- get_schema_version(): Aktuelle Schema-Version
- check_migration_needed(): Prueft ob Migration noetig

Rollback-Utilities:
- rollback_to_backup(): Manueller Rollback
- list_backups(): Verfuegbare Backups auflisten
- verify_backup(): Backup-Integritaet pruefen
- print_backup_info(): Backup-Info anzeigen

Siehe ROLLBACK_GUIDE.md fuer Details.
"""

from .rollback import (
    find_backup_for_version,
    list_backups,
    print_backup_info,
    rollback_to_backup,
    verify_backup,
)
from .v2_video_analysis import (
    SCHEMA_VERSION,
    check_migration_needed,
    create_backup,
    get_schema_version,
    get_unanalyzed_clip_count,
    migrate_to_v2,
    rollback_v2_to_v1,
)

__all__ = [
    # Migration
    "migrate_to_v2",
    "rollback_v2_to_v1",
    "get_schema_version",
    "check_migration_needed",
    "create_backup",
    "get_unanalyzed_clip_count",
    "SCHEMA_VERSION",
    # Rollback utilities
    "rollback_to_backup",
    "list_backups",
    "find_backup_for_version",
    "verify_backup",
    "print_backup_info",
]
