import polars as pl
from db import get_db_connection
from config import SYMBOLS, BENCHMARK_SYMBOL, PORTFOLIO_CAPITAL, COMMISSION_RATE


def build_regime_map(df: pl.DataFrame) -> pl.DataFrame:
    qqq = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date")
    qqq = qqq.with_columns(
        pl.col("close").rolling_mean(200).alias("qqq_sma_200")
    ).with_columns(
        pl.when(pl.col("qqq_sma_200").is_null())
        .then(True)
        .otherwise(pl.col("close") >= pl.col("qqq_sma_200"))
        .alias("is_macro_bull")
    )
    return qqq.select(["price_date", "is_macro_bull"])


def run_backtest_for_symbol(df: pl.DataFrame, regime_df: pl.DataFrame, symbol: str) -> dict:
    sub = df.filter(pl.col("symbol") == symbol).sort("price_date")
    sub = sub.join(regime_df, on="price_date", how="left").with_columns(
        pl.col("is_macro_bull").fill_null(True)
    )

    sub = sub.with_columns([
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_mean(50).alias("sma_50"),
        pl.col("close").rolling_mean(200).alias("sma_200"),
        pl.col("volume").rolling_mean(20).alias("vol_sma_20"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]).alias("tr"),
    ]).with_columns([
        pl.col("tr").rolling_mean(14).alias("atr_14"),
    ])

    capital = PORTFOLIO_CAPITAL
    peak_equity = PORTFOLIO_CAPITAL
    max_drawdown_pct = 0.0

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    shares = 0.0

    trade_pnls_usd = []
    blocked_entries = 0

    for row in sub.iter_rows(named=True):
        if row["sma_50"] is None or row["atr_14"] is None:
            continue

        close = row["close"]
        high = row["high"]
        low = row["low"]

        current_equity = (shares * close) if in_position else capital
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

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
                cost_basis = shares * entry_price
                trade_pnls_usd.append(proceeds - cost_basis)
                capital += proceeds
                in_position = False
                shares = 0.0

        else:
            trend_up = close > row["sma_20"] > row["sma_50"]
            vol_ok = row["volume"] >= (row["vol_sma_20"] or 0) * 1.1

            if trend_up and vol_ok:
                stock_bull = (close >= row["sma_200"]) if row["sma_200"] is not None else True
                if not row["is_macro_bull"] or not stock_bull:
                    blocked_entries += 1
                    continue

                atr = row["atr_14"]
                entry_price = close
                stop_loss = max(0.01, close - 2.0 * atr)
                take_profit = close + 3.0 * atr
                shares = (capital * (1.0 - COMMISSION_RATE)) / entry_price
                capital = 0.0
                in_position = True

    if in_position:
        last_close = sub[-1, "close"]
        proceeds = shares * last_close * (1.0 - COMMISSION_RATE)
        trade_pnls_usd.append(proceeds - (shares * entry_price))
        capital += proceeds

    wins = [p for p in trade_pnls_usd if p > 0]
    losses = [abs(p) for p in trade_pnls_usd if p <= 0]

    win_rate = (len(wins) / len(trade_pnls_usd) * 100.0) if trade_pnls_usd else 0.0
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    total_return = ((capital - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0

    return {
        "symbol": symbol,
        "trades": len(trade_pnls_usd),
        "blocked": blocked_entries,
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_drawdown_pct, 2),
        "profit_factor": round(profit_factor, 2),
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
    regime_df = build_regime_map(df)

    print("=" * 98)
    print(f"KURUMSAL BACKTEST KARNESİ [KATMAN-1: {BENCHMARK_SYMBOL} & HİSSE SMA-200 KİLİTLİ] (${PORTFOLIO_CAPITAL:,.0f} | %{COMMISSION_RATE*100:.1f} Komisyon)")
    print("=" * 98)

    total_start = PORTFOLIO_CAPITAL * len(SYMBOLS)
    total_end = 0.0

    for sym in SYMBOLS:
        res = run_backtest_for_symbol(df, regime_df, sym)
        total_end += res["final_capital"]
        print(
            f"[{res['symbol']:4s}] İşlem: {res['trades']:2d} (Kilitlenen: {res['blocked']:2d}) | "
            f"Win: %{res['win_rate']:5.1f} | "
            f"MaxDD: %{res['max_dd']:5.2f} | "
            f"PF: {res['profit_factor']:4.2f} | "
            f"Bakiye: ${res['final_capital']:9.2f} ({res['return_pct']:+6.2f}%)"
        )

    total_ret = ((total_end - total_start) / total_start) * 100.0
    print("-" * 98)
    print(f"PORTFÖY TOPLAMI: ${total_start:,.2f} -> ${total_end:,.2f} (Net Değişim: {total_ret:+.2f}%)")
    print("=" * 98)


if __name__ == "__main__":
    main()
