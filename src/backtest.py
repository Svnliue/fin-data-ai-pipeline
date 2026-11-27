import polars as pl
from db import get_db_connection
from logger import get_logger
from config import (
    SYMBOLS,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    COMMISSION_RATE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
)

log = get_logger("backtest")


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


def prepare_features(df: pl.DataFrame, regime_df: pl.DataFrame) -> pl.DataFrame:
    sub = df.filter(pl.col("symbol").is_in(SYMBOLS)).sort(["symbol", "price_date"])
    sub = sub.join(regime_df, on="price_date", how="left").with_columns(
        pl.col("is_macro_bull").fill_null(True)
    )
    return sub.with_columns([
        pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(200).over("symbol").alias("sma_200"),
        pl.col("volume").rolling_mean(20).over("symbol").alias("vol_sma_20"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        ]).alias("tr"),
    ]).with_columns([
        pl.col("tr").rolling_mean(14).over("symbol").alias("atr_14"),
    ])


def run_backtest_for_symbol(feat_df: pl.DataFrame, symbol: str) -> dict:
    sub = feat_df.filter(pl.col("symbol") == symbol).sort("price_date")

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


def run_shared_portfolio_backtest(feat_df: pl.DataFrame) -> dict:
    total_start = PORTFOLIO_CAPITAL * len(SYMBOLS)
    cash = total_start
    peak_equity = total_start
    max_dd_pct = 0.0
    positions: dict[str, dict] = {}
    blocked_by_slot_limit = 0
    total_trades = 0

    dates = sorted(feat_df["price_date"].unique().to_list())
    rows_by_date: dict[object, list[dict]] = {d: [] for d in dates}
    for r in feat_df.iter_rows(named=True):
        rows_by_date[r["price_date"]].append(r)

    for d in dates:
        day_rows = rows_by_date[d]
        last_prices = {r["symbol"]: r["close"] for r in day_rows}

        # 1. Acik pozisyon cikislari
        for r in day_rows:
            sym = r["symbol"]
            if sym not in positions or r["sma_50"] is None:
                continue
            pos = positions[sym]
            exit_price = None
            if r["low"] <= pos["sl"]:
                exit_price = pos["sl"]
            elif r["high"] >= pos["tp"]:
                exit_price = pos["tp"]
            elif r["close"] < r["sma_20"]:
                exit_price = r["close"]

            if exit_price is not None:
                cash += pos["shares"] * exit_price * (1.0 - COMMISSION_RATE)
                del positions[sym]
                total_trades += 1

        # 2. Yeni girisler (Max 3 eszamanli pozisyon + %25 tavan)
        for r in day_rows:
            sym = r["symbol"]
            if sym in positions or r["sma_50"] is None or r["atr_14"] is None:
                continue
            trend_up = r["close"] > r["sma_20"] > r["sma_50"]
            vol_ok = r["volume"] >= (r["vol_sma_20"] or 0) * 1.1
            stock_bull = (r["close"] >= r["sma_200"]) if r["sma_200"] is not None else True

            if trend_up and vol_ok and r["is_macro_bull"] and stock_bull:
                if len(positions) >= MAX_OPEN_POSITIONS:
                    blocked_by_slot_limit += 1
                    continue

                equity_now = cash + sum(p["shares"] * last_prices.get(s, p["entry"]) for s, p in positions.items())
                alloc = min(cash, equity_now * MAX_POSITION_PCT)
                if alloc > 100.0:
                    shares = (alloc * (1.0 - COMMISSION_RATE)) / r["close"]
                    cash -= alloc
                    positions[sym] = {
                        "shares": shares,
                        "entry": r["close"],
                        "sl": max(0.01, r["close"] - 2.0 * r["atr_14"]),
                        "tp": r["close"] + 3.0 * r["atr_14"],
                    }

        equity_eod = cash + sum(p["shares"] * last_prices.get(s, p["entry"]) for s, p in positions.items())
        if equity_eod > peak_equity:
            peak_equity = equity_eod
        dd = ((peak_equity - equity_eod) / peak_equity) * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd

    final_equity = cash + sum(p["shares"] * last_prices.get(s, p["entry"]) for s, p in positions.items())
    return {
        "start": total_start,
        "end": round(final_equity, 2),
        "return_pct": round(((final_equity - total_start) / total_start) * 100.0, 2),
        "max_dd": round(max_dd_pct, 2),
        "trades": total_trades,
        "slot_blocked": blocked_by_slot_limit,
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
    feat_df = prepare_features(df, regime_df)

    print("=" * 98)
    print(f"KURUMSAL BACKTEST KARNESİ [2x SL / 3x TP + ÇİFT SMA-200 KİLİT] (${PORTFOLIO_CAPITAL:,.0f} | %{COMMISSION_RATE*100:.1f} Komisyon)")
    print("=" * 98)

    for sym in SYMBOLS:
        res = run_backtest_for_symbol(feat_df, sym)
        log.info(f"[{sym}] Return={res['return_pct']:+.2f}% | MaxDD={res['max_dd']:.2f}% | PF={res['profit_factor']:.2f}")
        print(
            f"[{res['symbol']:4s}] İşlem: {res['trades']:2d} (Kilitlenen: {res['blocked']:2d}) | "
            f"Win: %{res['win_rate']:5.1f} | "
            f"MaxDD: %{res['max_dd']:5.2f} | "
            f"PF: {res['profit_factor']:4.2f} | "
            f"Bakiye: ${res['final_capital']:9.2f} ({res['return_pct']:+6.2f}%)"
        )

    shared = run_shared_portfolio_backtest(feat_df)
    log.info(f"[SHARED_PORTFOLIO] End=${shared['end']} ({shared['return_pct']:+.2f}%) | MaxDD={shared['max_dd']:.2f}%")
    print("-" * 98)
    print(
        f"ORTAK PORTFÖY HAVUZU (Max {MAX_OPEN_POSITIONS} Pozisyon, %25 Tavan): "
        f"${shared['start']:,.2f} -> ${shared['end']:,.2f} ({shared['return_pct']:+.2f}%) | "
        f"MaxDD: %{shared['max_dd']:.2f} | Slot Engeli: {shared['slot_blocked']}"
    )
    print("=" * 98)


if __name__ == "__main__":
    main()
