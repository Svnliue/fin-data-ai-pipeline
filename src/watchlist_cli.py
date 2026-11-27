import sys
from db import get_db_connection
from logger import get_logger

log = get_logger("watchlist_cli")


def list_symbols() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, asset_type, is_active, added_at::date FROM watchlist ORDER BY is_active DESC, symbol;")
            rows = cur.fetchall()
    print("\n--- KULLANICI İZLEME LİSTESİ (WATCHLIST) ---")
    for sym, atype, active, added in rows:
        status = "AKTİF" if active else "PASİF"
        print(f"  {sym:<8} | Tür: {atype:<6} | Durum: {status:<6} | Eklenme: {added}")
    print("-" * 44)


def add_symbol(symbol: str, asset_type: str = "STOCK") -> None:
    sym = symbol.strip().upper()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO watchlist (symbol, asset_type, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (symbol) DO UPDATE SET is_active = TRUE, asset_type = EXCLUDED.asset_type;
                """,
                (sym, asset_type.upper()),
            )
        conn.commit()
    log.info(f"Watchlist güncellendi: {sym} eklendi/aktifleştirildi.")


def remove_symbol(symbol: str) -> None:
    sym = symbol.strip().upper()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE watchlist SET is_active = FALSE WHERE symbol = %s;", (sym,))
        conn.commit()
    log.info(f"Watchlist güncellendi: {sym} pasife alındı.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        list_symbols()
    elif sys.argv[1] == "add" and len(sys.argv) >= 3:
        atype = sys.argv[3] if len(sys.argv) >= 4 else "STOCK"
        add_symbol(sys.argv[2], atype)
        list_symbols()
    elif sys.argv[1] == "remove" and len(sys.argv) >= 3:
        remove_symbol(sys.argv[2])
        list_symbols()
    else:
        print("Kullanım: python src/watchlist_cli.py [list | add <SEMBOL> [STOCK|ETF] | remove <SEMBOL>]")
