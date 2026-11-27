import math
import polars as pl
from db import get_db_connection
from logger import get_logger
from config import (
    SYMBOLS,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    COMMISSION_RATE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
)

log = get_logger("backtest")


def build_benchmark_df(df: pl.DataFrame) -> pl.DataFrame:
    qqq = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date")
    qqq = qqq.with_columns([
        pl.col("close").rolling_mean(200).alias("qqq_sma_200"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("qqq_ret_20d"),
    ]).with_columns(
        pl.when(pl.col("qqq_sma_200").is_null())
        .then(True)
        .otherwise(pl.col("close") >= pl.col("qqq_sma_200"))
        .alias("is_macro_bull")
    )
    return qqq.select(["price_date", "is_macro_bull", "qqq_ret_20d"])


def prepare_features(df: pl.DataFrame, bench_df: pl.DataFrame) -> pl.DataFrame:
    sub = df.filter(pl.col("symbol").is_in(SYMBOLS)).sort(["symbol", "price_date"])
    sub = sub.join(bench_df, on="price_date", how="left").with_columns([
        pl.col("is_macro_bull").fill_null(True),
        pl.col("qqq_ret_20d").fill_null(0.0),
    ])

    sub = sub.with_columns([
        pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(200).over("symbol").alias("sma_200"),
        pl.col("volume").rolling_mean(20).over("symbol").alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("ret_20d"),
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("delta"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        ]).alias("tr"),
    ]).with_columns([
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
        pl.col("tr").rolling_mean(14).over("symbol").alias("atr_14"),
        (pl.col("volume") / (pl.col("vol_sma_20") + 1e-9)).alias("vol_ratio"),
        (((pl.col("ret_20d") - pl.col("qqq_ret_20d")) * 100.0)).alias("rs_vs_qqq_pct"),
    ]).with_columns([
        pl.col("gain").rolling_mean(14).over("symbol").alias("avg_gain"),
        pl.col("loss").rolling_mean(14).over("symbol").alias("avg_loss"),
    ]).with_columns(
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_14")
    )
    return sub


def compute_multifactor_score(row: dict) -> float:
    close = row["close"]
    sma20 = row["sma_20"]
    sma50 = row["sma_50"]
    rsi = row["rsi_14"]
    vol_ratio = row["vol_ratio"] or 1.0
    rs_pct = row["rs_vs_qqq_pct"] or 0.0

    score = 0.0
    if sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            score += 1.5
        elif close < sma20 < sma50:
            score -= 1.5

    if rs_pct >= 2.0:
        score += 0.5
    elif rs_pct <= -2.0:
        score -= 0.5

    if vol_ratio >= 1.2:
        score += 0.5 if score > 0 else -0.5

    if rsi is not None:
        if rsi < 35:
            score += 1.0
        elif rsi > 70:
            score -= 1.0

    return score


def calc_vol_targeted_allocation(cash: float, equity: float, close: float, atr: float) -> float:
    # %1 kasa riski + ATR volatilite hedeflemesi + %25 tavan
    stop_dist = max(0.01, 2.0 * atr)
    risk_budget = equity * RISK_PER_TRADE
    shares_by_risk = risk_budget / stop_dist
    usd_by_risk = shares_by_risk * close
    usd_by_cap = equity * MAX_POSITION_PCT
    return min(cash, usd_by_risk, usd_by_cap)


def run_backtest_for_symbol(feat_df: pl.DataFrame, symbol: str) -> dict:
    sub = feat_df.filter(pl.col("symbol") == symbol).sort("price_date")

    cash = PORTFOLIO_CAPITAL
    peak_equity = PORTFOLIO_CAPITAL
    max_drawdown_pct = 0.0
    daily_returns = []
    prev_equity = PORTFOLIO_CAPITAL

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    partial_tp = 0.0
    final_tp = 0.0
    shares = 0.0
    scaled_out = False
    current_trade_pnl = 0.0

    trade_pnls_usd = []
    blocked_entries = 0

    for row in sub.iter_rows(named=True):
        if row["sma_50"] is None or row["atr_14"] is None:
            continue

        close = row["close"]
        high = row["high"]
        low = row["low"]

        current_equity = cash + (shares * close if in_position else 0.0)
        if prev_equity > 0:
            daily_returns.append((current_equity - prev_equity) / prev_equity)
        prev_equity = current_equity

        if current_equity > peak_equity:
            peak_equity = current_equity
        dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        if in_position:
            # 1. Parcali Kar Alma (+1.5 ATR -> %50 pozisyon kapat + Stop Breakeven cek)
            if not scaled_out and high >= partial_tp and shares > 0:
                half_shares = shares * 0.5
                proceeds = half_shares * partial_tp * (1.0 - COMMISSION_RATE)
                current_trade_pnl += proceeds - (half_shares * entry_price)
                cash += proceeds
                shares -= half_shares
                stop_loss = max(stop_loss, entry_price * 1.002)  # Breakeven + komisyon kilidi
                scaled_out = True

            # 2. Tam Cikis Kontrolleri
            exit_price = None
            if low <= stop_loss:
                exit_price = stop_loss
            elif high >= final_tp:
                exit_price = final_tp
            elif close < row["sma_20"]:
                exit_price = close

            if exit_price is not None:
                proceeds = shares * exit_price * (1.0 - COMMISSION_RATE)
                current_trade_pnl += proceeds - (shares * entry_price)
                cash += proceeds
                trade_pnls_usd.append(current_trade_pnl)
                in_position = False
                shares = 0.0
                current_trade_pnl = 0.0

        else:
            score = compute_multifactor_score(row)
            if score >= 1.5:
                stock_bull = (close >= row["sma_200"]) if row["sma_200"] is not None else True
                if not row["is_macro_bull"] or not stock_bull:
                    blocked_entries += 1
                    continue

                atr = row["atr_14"]
                alloc = calc_vol_targeted_allocation(cash, current_equity, close, atr)
                if alloc > 50.0:
                    entry_price = close
                    stop_loss = max(0.01, close - 2.0 * atr)
                    partial_tp = close + 1.5 * atr
                    final_tp = close + 3.5 * atr
                    shares = (alloc * (1.0 - COMMISSION_RATE)) / entry_price
                    cash -= alloc
                    in_position = True
                    scaled_out = False
                    current_trade_pnl = -(alloc * COMMISSION_RATE)

    if in_position:
        last_close = sub[-1, "close"]
        proceeds = shares * last_close * (1.0 - COMMISSION_RATE)
        current_trade_pnl += proceeds - (shares * entry_price)
        cash += proceeds
        trade_pnls_usd.append(current_trade_pnl)

    wins = [p for p in trade_pnls_usd if p > 0]
    losses = [abs(p) for p in trade_pnls_usd if p <= 0]

    win_rate = (len(wins) / len(trade_pnls_usd) * 100.0) if trade_pnls_usd else 0.0
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    expectancy = (sum(trade_pnls_usd) / len(trade_pnls_usd)) if trade_pnls_usd else 0.0
    total_return = ((cash - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0

    # Yilliklandirilmis Sharpe Ratio
    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1))
        sharpe = ((mean_r / std_r) * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "symbol": symbol,
        "trades": len(trade_pnls_usd),
        "blocked": blocked_entries,
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_drawdown_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "expectancy": round(expectancy, 2),
        "final_capital": round(cash, 2),
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
    bench_df = build_benchmark_df(df)
    feat_df = prepare_features(df, bench_df)

    print("=" * 112)
    print("ALPHA ENGINE v4.4 BACKTEST [PARÇALI KÂR + BREAKEVEN LOCK + VOLATİLİTE BOYUTLANDIRMA + SHARPE]")
    print("=" * 112)

    total_start = PORTFOLIO_CAPITAL * len(SYMBOLS)
    total_end = 0.0

    for sym in SYMBOLS:
        res = run_backtest_for_symbol(feat_df, sym)
        total_end += res["final_capital"]
        log.info(f"[{sym}] Ret={res['return_pct']:+.2f}% | MaxDD={res['max_dd']:.2f}% | PF={res['profit_factor']:.2f} | Sharpe={res['sharpe']:.2f}")
        print(
            f"[{res['symbol']:4s}] İşlem:{res['trades']:2d} (Kilit:{res['blocked']:2d}) | "
            f"Win:%{res['win_rate']:5.1f} | "
            f"MaxDD:%{res['max_dd']:5.2f} | "
            f"PF:{res['profit_factor']:4.2f} | "
            f"Sharpe:{res['sharpe']:5.2f} | "
            f"Exp:${res['expectancy']:+6.1f} | "
            f"Bakiye:${res['final_capital']:9.2f} ({res['return_pct']:+6.2f}%)"
        )

    total_ret = ((total_end - total_start) / total_start) * 100.0
    print("-" * 112)
    print(f"PORTFÖY TOPLAMI: ${total_start:,.2f} -> ${total_end:,.2f} (Net Değişim: {total_ret:+.2f}%)")
    print("=" * 112)


if __name__ == "__main__":
    main()
