import time
import polars as pl
import yfinance as yf
from db import get_db_connection
from logger import get_logger
from config import get_active_symbols, BENCHMARK_SYMBOL, PRICE_PERIOD

log = get_logger("fetch_prices")


def fetch_with_retry(symbol: str, period: str, max_retries: int = 3) -> pl.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                raise ValueError(f"{symbol} için boş veri döndü.")

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
            return df

        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"{symbol} {max_retries} denemede de çekilemedi: {exc}") from exc
            wait_sec = 2 ** attempt
            log.warning(f"{symbol} bağlantı hatası (Deneme {attempt}/{max_retries}). {wait_sec} sn bekleniyor...")
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


def main(symbols: list[str] | None = None) -> dict:
    log.info("Finansal Fiyat ETL Boru Hattı Başlatıldı (Dinamik Watchlist + Açık Pozisyonlar)")
    tickers = sorted(set((symbols or get_active_symbols()) + [BENCHMARK_SYMBOL]))
    total_records = 0
    failed = []

    for sym in tickers:
        try:
            df = fetch_with_retry(sym, PRICE_PERIOD)
            inserted = upsert_prices(df)
            total_records += inserted
            print(f"[BAŞARILI] {sym}: {inserted} günlük veri Polars ile işlendi ve DB'ye yazıldı.")
        except Exception as exc:
            failed.append(sym)
            log.error(f"[HATA] {sym} fiyat verisi atlandı: {exc}")

    print(f"--- Tamamlandı: Toplam {total_records} kayıt işlendi | Hatalı: {len(failed)} ---")
    return {"total_records": total_records, "failed": failed}


if __name__ == "__main__":
    main()
