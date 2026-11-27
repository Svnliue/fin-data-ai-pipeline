import sys
import yfinance as yf
from db import get_db_connection
from logger import get_logger

log = get_logger("watchlist_cli")

LEVERAGED_KEYWORDS = (
    "2X", "3X", "-2X", "-3X", "ULTRA", "ULTRAPRO",
    "BULL", "BEAR", "SHORT", "INVERSE", "LEVERAGED", "DAILY"
)
COMMODITY_KEYWORDS = (
    "GOLD", "SILVER", "OIL", "CRUDE", "NATURAL GAS",
    "COPPER", "PLATINUM", "COMMODITY", "COMMODITIES", "BULLION", "MINERS"
)


def detect_asset_type(symbol: str) -> tuple[str, str]:
    """
    Yahoo Finance meta verisinden varlik sinifini otomatik tespit eder.
    Donus: (asset_type, tespit_aciklamasi)
    """
    sym = symbol.strip().upper()
    try:
        info = yf.Ticker(sym).info or {}
        quote_type = str(info.get("quoteType", "")).upper()
        name = str(info.get("longName") or info.get("shortName") or "").upper()
        category = str(info.get("category") or "").upper()
        combined = f"{name} {category}"

        if quote_type == "EQUITY":
            return "STOCK", f"Yahoo Finance EQUITY ({info.get('shortName', sym)})"

        if quote_type in ("ETF", "MUTUALFUND", "FUTURE"):
            if any(kw in combined for kw in LEVERAGED_KEYWORDS):
                return "LEVERAGED", f"Kaldıraçlı/Ters ETF ({info.get('shortName', sym)})"
            if quote_type == "FUTURE" or any(kw in combined for kw in COMMODITY_KEYWORDS):
                return "COMMODITY", f"Emtia/Maden Varlığı ({info.get('shortName', sym)})"
            return "SECTOR_ETF", f"Sektör/Endeks ETF ({info.get('shortName', sym)})"

    except Exception as exc:
        log.warning(f"{sym} için Yahoo meta verisi alınamadı, varsayılan STOCK atandı: {exc}")

    return "STOCK", "Varsayılan Profil (STOCK)"


def list_symbols() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, asset_type, is_active, added_at::date FROM watchlist ORDER BY is_active DESC, symbol;")
            rows = cur.fetchall()
    print("\n--- KULLANICI İZLEME LİSTESİ (WATCHLIST) ---")
    for sym, atype, active, added in rows:
        status = "AKTİF" if active else "PASİF"
        print(f"  {sym:<8} | Tür: {atype:<11} | Durum: {status:<6} | Eklenme: {added}")
    print("-" * 52)


def add_symbol(symbol: str, asset_type: str | None = None) -> None:
    sym = symbol.strip().upper()
    if asset_type:
        resolved_type = asset_type.strip().upper()
        reason = "Kullanıcı manuel seçimi"
    else:
        resolved_type, reason = detect_asset_type(sym)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO watchlist (symbol, asset_type, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (symbol) DO UPDATE SET is_active = TRUE, asset_type = EXCLUDED.asset_type;
                """,
                (sym, resolved_type),
            )
        conn.commit()
    log.info(f"Watchlist güncellendi: {sym} -> {resolved_type} ({reason})")
    print(f"[OTOMATİK SINIFLANDIRMA] {sym} -> {resolved_type} | Kaynak: {reason}")


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
        manual_type = sys.argv[3] if len(sys.argv) >= 4 else None
        add_symbol(sys.argv[2], manual_type)
        list_symbols()
    elif sys.argv[1] == "remove" and len(sys.argv) >= 3:
        remove_symbol(sys.argv[2])
        list_symbols()
    else:
        print("Kullanım: python src/watchlist_cli.py [list | add <SEMBOL> [MANUEL_TÜR] | remove <SEMBOL>]")
