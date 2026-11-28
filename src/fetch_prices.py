import time
from datetime import datetime, date, time as dt_time
from zoneinfo import ZoneInfo
import polars as pl
import yfinance as yf
from db import get_db_connection
from config import get_active_symbols, BENCHMARK_SYMBOL, PRICE_PERIOD
from logger import get_logger

log = get_logger("fetch_prices")
NY_TZ = ZoneInfo("America/New_York")


def _filter_incomplete_intraday_bar(df: pl.DataFrame, now_ny: datetime | None = None) -> pl.DataFrame:
    """NYSE seansi acikken (09:30 - 16:05 ET) olusan yarim gunluk bugunku bari eler."""
    if df.is_empty():
        return df
    now_ny = now_ny or datetime.now(NY_TZ)
    if now_ny.weekday() < 5:
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 5)
        if market_open <= now_ny.time() < market_close:
            today_ny = now_ny.date()
            before_len = df.height
            df = df.filter(pl.col("price_date") < today_ny)
            if df.height < before_len:
                log.info(f"Seans acik ({now_ny.strftime('%H:%M ET')}): Yarim gunluk bugunku bar elendi.")
    return df


def _check_trading_day_gaps(symbol: str, df: pl.DataFrame) -> None:
    if df.height < 2:
        return
    dates = df.sort("price_date")["price_date"].to_list()
    for prev_d, curr_d in zip(dates[:-1], dates[1:]):
        gap = (curr_d - prev_d).days
        if gap > 4:
            log.warning(f"{symbol}: {prev_d} ile {curr_d} arasinda {gap} takvim gunluk veri boslugu tespit edildi.")


def get_target_symbols() -> list[str]:
    symbols = set(get_active_symbols())
    symbols.add(BENCHMARK_SYMBOL)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM portfolio_positions WHERE shares > 0;")
                for (sym,) in cur.fetchall():
                    symbols.add(sym)
    except Exception as exc:
        log.error(f"Acik pozisyon sembolleri okunamadi: {exc}")
    return sorted(symbols)


def fetch_with_retry(symbol: str, period: str, max_retries: int = 3) -> pl.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                raise ValueError(f"{symbol} icin bos veri dondu.")

            hist = hist.reset_index()
            df = pl.from_pandas(hist).select([
                pl.lit(symbol).alias("symbol"),
                pl.col("Date").dt.date().alias("price_date"),
                pl.col("Open").cast(pl.Float64).alias("open_price"),
                pl.col("High").cast(pl.Float64).alias("high_price"),
                pl.col("Low").cast(pl.Float64).alias("low_price"),
                pl.col("Close").cast(pl.Float64).alias("close_price"),
                pl.col("Volume").cast(pl.Int64).alias("volume"),
            ]).drop_nulls()

            df = _filter_incomplete_intraday_bar(df)
            if not df.is_empty():
                _check_trading_day_gaps(symbol, df)
                last_date = df["price_date"].max()
                if (date.today() - last_date).days > 5:
                    log.warning(f"{symbol}: Son fiyat bari bayat ({last_date}).")
            return df

        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"{symbol} {max_retries} denemede de cekilemedi: {exc}") from exc
            wait_sec = 2 ** attempt
            print(f"[UYARI] {symbol} baglanti hatasi (Deneme {attempt}/{max_retries}). {wait_sec} sn bekleniyor...")
            time.sleep(wait_sec)

    return pl.DataFrame()


def upsert_prices(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    query = """
        INSERT INTO market_prices (symbol, price_date, open_price, high_price, low_price, close_price, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, price_date)
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume;
    """
    rows = df.rows()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    return len(rows)


def main() -> dict:
    log.info("Finansal Fiyat ETL Boru Hatti Baslatildi (Wilder + Seans Ici Bar Filtresi)")
    total_records = 0
    failed = []
    for sym in get_target_symbols():
        try:
            df = fetch_with_retry(sym, PRICE_PERIOD)
            inserted = upsert_prices(df)
            total_records += inserted
            print(f"[BASARILI] {sym}: {inserted} gunluk veri Polars ile islendi ve DB'ye yazildi.")
        except Exception as exc:
            failed.append(sym)
            log.error(f"[HATA] {sym} fiyat verisi cekilemedi: {exc}")

    if total_records == 0:
        raise RuntimeError("KRITIK HATA: Hicbir sembol icin fiyat verisi cekilemedi!")
    print(f"--- Tamamlandi: Toplam {total_records} kayit islendi | Hatali: {len(failed)} ---")
    return {"total_records": total_records, "failed": failed}


if __name__ == "__main__":
    main()
