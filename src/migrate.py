from pathlib import Path
from db import get_db_connection
from logger import get_logger

log = get_logger("migrate")
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def run_migrations() -> None:
    if not MIGRATIONS_DIR.exists():
        return
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute("SELECT filename FROM schema_migrations;")
            applied = {r[0] for r in cur.fetchall()}

            for fpath in sql_files:
                if fpath.name in applied:
                    continue
                log.info(f"Migration uygulanıyor: {fpath.name}")
                cur.execute(fpath.read_text(encoding="utf-8"))
                cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s);", (fpath.name,))
        conn.commit()


if __name__ == "__main__":
    run_migrations()
    print("[DB] Tüm migration dosyaları başarıyla uygulandı.")
