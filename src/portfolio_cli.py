import sys
import polars as pl
from db import get_db_connection
from logger import get_logger
from config import ASSET_PROFILES, get_active_asset_map
from watchlist_cli import add_symbol

log = get_logger("portfolio_cli")


def _get_latest_atr(symbol: str, fallback_price: float) -> float:
    query = """
        SELECT high_price::float AS high, low_price::float AS low, close_price::float AS close
        FROM market_prices
        WHERE symbol = %s
        ORDER BY price_date DESC
        LIMIT 20;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (symbol,))
                rows = cur.fetchall()
        if len(rows) >= 15:
            rows = list(reversed(rows))
            df = pl.DataFrame(rows, schema=["high", "low", "close"], orient="row")
            df = df.with_columns(
                pl.max_horizontal([
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ]).alias("tr")
            )
            atr = df["tr"].tail(14).mean()
            if atr and atr > 0:
                return round(float(atr), 4)
    except Exception:
        pass
    return round(fallback_price * 0.02, 4)


def list_positions() -> list[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.symbol, p.shares::float, p.entry_price::float,
                       COALESCE(p.initial_atr::float, 0), COALESCE(p.stop_loss::float, 0),
                       COALESCE(p.tp1::float, 0), COALESCE(p.tp2::float, 0),
                       COALESCE(p.scaled_out, FALSE), p.entry_date,
                       COALESCE(w.asset_type, 'STOCK') AS asset_type
                FROM portfolio_positions p
                LEFT JOIN watchlist w ON p.symbol = w.symbol
                ORDER BY p.symbol;
                """
            )
            rows = cur.fetchall()

    print("\n--- AÇIK PORTFÖY POZİSYONLARIM (ELİMDEKİ VARLIKLAR) ---")
    result = []
    if not rows:
        print("  Portföyde şu an açık pozisyon bulunmuyor (Tamamen Nakitte).")
    else:
        for sym, shares, entry_px, i_atr, sl, tp1, tp2, scaled, edate, atype in rows:
            cost = shares * entry_px
            print(
                f"  {sym:<8} | {atype:<10} | Adet: {shares:<7.2f} | Maliyet: ${entry_px:<8.2f} | "
                f"SL: ${sl:<8.2f} | TP1: ${tp1:<8.2f} | TP2: ${tp2:<8.2f} | KısmiSatış: {scaled}"
            )
            result.append({
                "symbol": sym,
                "asset_type": atype,
                "shares": shares,
                "entry_price": entry_px,
                "initial_atr": i_atr,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
                "scaled_out": scaled,
                "entry_date": str(edate),
                "cost_usd": round(cost, 2),
            })
    print("-" * 92)
    return result


def buy_position(symbol: str, shares: float, entry_price: float) -> None:
    sym = symbol.strip().upper()
    if shares <= 0 or entry_price <= 0:
        raise ValueError("Adet ve alış fiyatı 0'dan büyük olmalıdır.")

    add_symbol(sym)
    atype = get_active_asset_map().get(sym, "STOCK")
    profile = ASSET_PROFILES.get(atype, ASSET_PROFILES["STOCK"])
    atr = _get_latest_atr(sym, entry_price)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT shares::float, entry_price::float FROM portfolio_positions WHERE symbol = %s;", (sym,))
            existing = cur.fetchone()
            if existing:
                old_shares, old_px = existing
                new_shares = old_shares + shares
                avg_px = ((old_shares * old_px) + (shares * entry_price)) / new_shares
                sl = round(max(0.01, avg_px - (profile["sl_atr"] * atr)), 4)
                tp1 = round(avg_px + (profile["tp1_atr"] * atr), 4)
                tp2 = round(avg_px + (profile["tp2_atr"] * atr), 4)
                cur.execute(
                    """
                    UPDATE portfolio_positions
                    SET shares = %s, entry_price = %s, initial_atr = %s,
                        stop_loss = %s, tp1 = %s, tp2 = %s, scaled_out = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = %s;
                    """,
                    (new_shares, round(avg_px, 4), atr, sl, tp1, tp2, sym),
                )
            else:
                sl = round(max(0.01, entry_price - (profile["sl_atr"] * atr)), 4)
                tp1 = round(entry_price + (profile["tp1_atr"] * atr), 4)
                tp2 = round(entry_price + (profile["tp2_atr"] * atr), 4)
                cur.execute(
                    """
                    INSERT INTO portfolio_positions
                        (symbol, shares, entry_price, initial_atr, stop_loss, tp1, tp2, scaled_out)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE);
                    """,
                    (sym, shares, entry_price, atr, sl, tp1, tp2),
                )
        conn.commit()
    log.info(f"Portföy güncellendi: {sym} kaydedildi (ATR: {atr:.2f}, SL: {sl:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}).")


def sell_position(symbol: str, shares: float | None = None, exit_price: float | None = None, reason: str = "MANUAL_SELL") -> None:
    sym = symbol.strip().upper()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT shares::float, entry_price::float, entry_date FROM portfolio_positions WHERE symbol = %s;",
                (sym,),
            )
            existing = cur.fetchone()
            if not existing:
                print(f"[UYARI] Portföyde {sym} pozisyonu bulunamadı.")
                return
            old_shares, entry_px, opened_at = existing

            if exit_price is None:
                cur.execute("SELECT close_price::float FROM market_prices WHERE symbol = %s ORDER BY price_date DESC LIMIT 1;", (sym,))
                px_row = cur.fetchone()
                exit_price = px_row[0] if px_row else entry_px

            sold_shares = old_shares if (shares is None or shares >= old_shares) else shares
            realized_pnl = round((exit_price - entry_px) * sold_shares, 4)

            cur.execute(
                """
                INSERT INTO trades (symbol, shares, entry_price, exit_price, realized_pnl, exit_reason, opened_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (sym, sold_shares, entry_px, exit_price, realized_pnl, reason, opened_at),
            )

            if sold_shares >= old_shares:
                cur.execute("DELETE FROM portfolio_positions WHERE symbol = %s;", (sym,))
                log.info(f"{sym} pozisyonu kapatıldı | Realized PnL: ${realized_pnl:+.2f}")
            else:
                cur.execute(
                    """
                    UPDATE portfolio_positions
                    SET shares = %s, scaled_out = TRUE, stop_loss = GREATEST(COALESCE(stop_loss, 0), %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = %s;
                    """,
                    (old_shares - sold_shares, round(entry_px * 1.002, 4), sym),
                )
                log.info(f"{sym} kısmi satış ({sold_shares} adet) | Realized PnL: ${realized_pnl:+.2f} | Stop maliyete çekildi")
        conn.commit()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        list_positions()
    elif sys.argv[1] == "buy" and len(sys.argv) >= 5:
        buy_position(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
        list_positions()
    elif sys.argv[1] == "sell" and len(sys.argv) >= 3:
        qty = float(sys.argv[3]) if len(sys.argv) >= 4 else None
        px = float(sys.argv[4]) if len(sys.argv) >= 5 else None
        sell_position(sys.argv[2], qty, px)
        list_positions()
    else:
        print("Kullanım: python src/portfolio_cli.py [list | buy <SEMBOL> <ADET> <ALIŞ_FİYATI> | sell <SEMBOL> [ADET] [SATIŞ_FİYATI]]")
