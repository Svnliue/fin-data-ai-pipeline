import polars as pl
from db import get_db_connection

SYMBOLS = ["AAPL", "NVDA", "MSFT", "SAGT", "IREX"]
INITIAL_CAPITAL = 10000.0
COMMISSION_RATE = 0.001  # %0.1 komisyon + slippage payi


def run_backtest_for_symbol(df: pl.DataFrame, symbol: str) -> dict:
    sub = df.filter(pl.col("symbol") == symbol).sort("price_date")
    sub = sub.with_columns([
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_mean(50).alias("sma_50"),
        pl.col("volume").rolling_mean(20).alias("vol_sma_20"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]).alias("tr"),
    ]).with_columns([
        pl.col("tr").rolling_mean(14).alias("atr_14"),
    ])

    capital = INITIAL_CAPITAL
    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    shares = 0.0
    trades = []

    for row in sub.iter_rows(named=True):
        if row["sma_50"] is None or row["atr_14"] is None:
            continue

        close = row["close"]
        high = row["high"]
        low = row["low"]

        if in_position:
            exit_price = None
            if low <= stop_loss:
                exit_price = stop_loss
            elif high >= take_profit:
                exit_price = take_profit
            elif close < row["sma_20"]:
                exit_price = close

            if exit_price is not None:
                proceeds = shares * exit_price * (1.0 - COMMISSION_RATE)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                capital += proceeds
                trades.append(pnl_pct)
                in_position = False
                shares = 0.0

        else:
            trend_up = close > row["sma_20"] > row["sma_50"]
            vol_ok = row["volume"] >= (row["vol_sma_20"] or 0) * 1.1

            if trend_up and vol_ok:
                atr = row["atr_14"]
                entry_price = close
                stop_loss = max(0.01, close - 2.0 * atr)
                take_profit = close + 3.0 * atr
                shares = (capital * (1.0 - COMMISSION_RATE)) / entry_price
                capital = 0.0
                in_position = True

    if in_position:
        last_close = sub[-1, "close"]
        capital += shares * last_close * (1.0 - COMMISSION_RATE)
        trades.append(((last_close - entry_price) / entry_price) * 100.0)

    wins = sum(1 for t in trades if t > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "final_capital": round(capital, 2),
        "return_pct": round(total_return, 2),
    }


def main():
    query = """
        SELECT symbol, price_date,
               high_price::float AS high,
               low_price::float AS low,
               close_price::float AS close,
               volume::float AS volume
        FROM market_prices
        ORDER BY symbol, price_date ASC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    df = pl.DataFrame(rows, schema=["symbol", "price_date", "high", "low", "close", "volume"], orient="row")

    print("=" * 78)
    print("1 YILLIK WALK-FORWARD BACKTEST SİMÜLASYONU ($10,000 Başlangıç | %0.1 Komisyon)")
    print("=" * 78)

    for sym in SYMBOLS:
        res = run_backtest_for_symbol(df, sym)
        print(
            f"[{res['symbol']:4s}] İşlem: {res['trades']:2d} | "
            f"Win Rate: %{res['win_rate']:5.1f} | "
            f"Son Bakiye: ${res['final_capital']:9.2f} | "
            f"Net Getiri: {res['return_pct']:+6.2f}%"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
