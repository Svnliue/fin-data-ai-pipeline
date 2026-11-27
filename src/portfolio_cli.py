import sys
from db import get_db_connection
from logger import get_logger
from watchlist_cli import add_symbol

log = get_logger("portfolio_cli")


def list_positions() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.symbol, p.shares::float, p.entry_price::float, p.entry_date,
                       COALESCE(w.asset_type, 'STOCK') AS asset_type
                FROM portfolio_positions p
                LEFT JOIN watchlist w ON p.symbol = w.symbol
                ORDER BY p.symbol;
                """
            )
            rows = cur.fetchall()

    print("\n--- AÇIK PORTFÖY POZİSYONLARIM (ELİMDEKİ VARLIKLAR) ---")
    if not rows:
        print("  Portföyde şu an açık pozisyon bulunmuyor (Tamamen Nakitte).")
    else:
        for sym, shares, entry_px, edate, atype in rows:
            cost = shares * entry_px
            print(f"  {sym:<8} | Tür: {atype:<11} | Adet: {shares:<8.2f} | Maliyet: ${entry_px:<9.2f} | Toplam: ${cost:<9.2f} | Tarih: {edate}")
    print("-" * 68)


def buy_position(symbol: str, shares: float, entry_price: float) -> None:
    sym = symbol.strip().upper()
    if shares <= 0 or entry_price <= 0:
        raise ValueError("Adet ve alış fiyatı 0'dan büyük olmalıdır.")

    # İzleme listesinde yoksa otomatik sınıflandırarak ekle ki veri çekimi devam etsin
    add_symbol(sym)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT shares::float, entry_price::float FROM portfolio_positions WHERE symbol = %s;", (sym,))
            existing = cur.fetchone()
            if existing:
                old_shares, old_px = existing
                new_shares = old_shares + shares
                avg_px = ((old_shares * old_px) + (shares * entry_price)) / new_shares
                cur.execute(
                    """
                    UPDATE portfolio_positions
                    SET shares = %s, entry_price = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = %s;
                    """,
                    (new_shares, round(avg_px, 4), sym),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO portfolio_positions (symbol, shares, entry_price)
                    VALUES (%s, %s, %s);
                    """,
                    (sym, shares, entry_price),
                )
        conn.commit()
    log.info(f"Portföy güncellendi: {sym} pozisyonu kaydedildi.")


def sell_position(symbol: str, shares: float | None = None) -> None:
    sym = symbol.strip().upper()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT shares::float FROM portfolio_positions WHERE symbol = %s;", (sym,))
            existing = cur.fetchone()
            if not existing:
                print(f"[UYARI] Portföyde {sym} pozisyonu bulunamadı.")
                return
            old_shares = existing[0]
            if shares is None or shares >= old_shares:
                cur.execute("DELETE FROM portfolio_positions WHERE symbol = %s;", (sym,))
                log.info(f"Portföy güncellendi: {sym} pozisyonu tamamen kapatıldı.")
            else:
                cur.execute(
                    "UPDATE portfolio_positions SET shares = %s, updated_at = CURRENT_TIMESTAMP WHERE symbol = %s;",
                    (old_shares - shares, sym),
                )
                log.info(f"Portföy güncellendi: {sym} pozisyonundan {shares} adet satıldı.")
        conn.commit()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        list_positions()
    elif sys.argv[1] == "buy" and len(sys.argv) >= 5:
        buy_position(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
        list_positions()
    elif sys.argv[1] == "sell" and len(sys.argv) >= 3:
        qty = float(sys.argv[3]) if len(sys.argv) >= 4 else None
        sell_position(sys.argv[2], qty)
        list_positions()
    else:
        print("Kullanım: python src/portfolio_cli.py [list | buy <SEMBOL> <ADET> <ALIŞ_FİYATI> | sell <SEMBOL> [ADET]]")
