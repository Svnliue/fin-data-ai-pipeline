import time
import polars as pl
import yfinance as yf
from db import get_db_connection

SYMBOLS = ["AAPL", "NVDA", "MSFT", "SAGT", "IREX"]

def extract_and_transform(symbol: str, period: str = "1mo", retries: int = 3) -> pl.DataFrame:
    """Fetches historical OHLCV data with retry logic and cleans it using Polars."""
    raw_pdf = None
    for attempt in range(1, retries + 1):
        try:
            raw_pdf = yf.download(
                symbol,
                period=period,
                progress=False,
                auto_adjust=True,
                timeout=30
            )
            if raw_pdf is not None and not raw_pdf.empty:
                if getattr(raw_pdf.columns, "nlevels", 1) > 1:
                    raw_pdf.columns = raw_pdf.columns.get_level_values(0)
                break
        except Exception as e:
            print(f"[DENEME {attempt}/{retries}] {symbol} ağ hatası: {e}")
        time.sleep(2)

    if raw_pdf is None or raw_pdf.empty:
        print(f"[UYARI] {symbol} için veri çekilemedi.")
        return pl.DataFrame()

    raw_pdf = raw_pdf.reset_index()
    df = pl.from_pandas(raw_pdf)

    clean_df = (
        df.select([
            pl.lit(symbol).alias("symbol"),
            pl.col("Date").dt.date().alias("price_date"),
            pl.col("Open").cast(pl.Float64).round(4).alias("open_price"),
            pl.col("High").cast(pl.Float64).round(4).alias("high_price"),
            pl.col("Low").cast(pl.Float64).round(4).alias("low_price"),
            pl.col("Close").cast(pl.Float64).round(4).alias("close_price"),
            pl.col("Volume").cast(pl.Int64).alias("volume"),
        ])
        .drop_nulls(subset=["price_date", "close_price"])
    )
    return clean_df

def load_to_postgres(df: pl.DataFrame):
    """Upserts transformed Polars DataFrame rows into PostgreSQL market_prices."""
    if df.is_empty():
        return

    upsert_query = """
        INSERT INTO market_prices (
            symbol, price_date, open_price, high_price, low_price, close_price, volume
        )
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
            cur.executemany(upsert_query, rows)
        conn.commit()

def run_pipeline():
    print("--- Finansal Fiyat ETL Boru Hattı Başlatıldı ---")
    total_rows = 0
    for sym in SYMBOLS:
        df = extract_and_transform(sym, period="1y")
        if not df.is_empty():
            load_to_postgres(df)
            total_rows += len(df)
            print(f"[BAŞARILI] {sym}: {len(df)} günlük veri Polars ile işlendi ve DB'ye yazıldı.")
        time.sleep(1)
    print(f"--- Tamamlandı: Toplam {total_rows} kayıt işlendi ---")

if __name__ == "__main__":
    run_pipeline()