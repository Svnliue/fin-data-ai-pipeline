import math
import polars as pl
from db import get_db_connection
from logger import get_logger
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    COMMISSION_RATE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
)

log = get_logger("backtest")
SLIPPAGE_RATE = 0.0005
CORR_THRESHOLD = 0.75


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


def prepare_features(df: pl.DataFrame, symbols: list[str], bench_df: pl.DataFrame) -> pl.DataFrame:
    sub = df.filter(pl.col("symbol").is_in(symbols)).sort(["symbol", "price_date"])
    sub = sub.join(bench_df, on="price_date", how="left").with_columns([
        pl.col("is_macro_bull").fill_null(True),
        pl.col("qqq_ret_20d").fill_null(0.0),
    ])

    sub = sub.with_columns([
        pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(200).over("symbol").alias("sma_200"),
        pl.col("volume").rolling_mean(20).over("symbol").alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(1).over("symbol")) - 1.0).alias("daily_ret"),
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


def can_enter_by_profile(row: dict, profile: dict) -> bool:
    if profile["require_macro_bull"] and not row["is_macro_bull"]:
        return False
    if profile["require_stock_sma200"]:
        sma200 = row["sma_200"]
        if sma200 is not None and row["close"] < sma200:
            return False
    return True


def calc_vol_targeted_allocation(cash: float, equity: float, close: float, atr: float, sl_mult: float) -> float:
    stop_dist = max(0.01, sl_mult * atr)
    risk_budget = equity * RISK_PER_TRADE
    shares_by_risk = risk_budget / stop_dist
    usd_by_risk = shares_by_risk * close
    usd_by_cap = equity * MAX_POSITION_PCT
    return min(cash, usd_by_risk, usd_by_cap)


def evaluate_realistic_exit(row: dict, pos: dict) -> tuple[float | None, bool, float, float]:
    open_p = row["open"]
    high = row["high"]
    low = row["low"]
    close = row["close"]

    sl = pos["sl"]
    partial_tp = pos["partial_tp"]
    final_tp = pos["final_tp"]
    scaled_out = pos["scaled_out"]
    entry = pos["entry"]

    # 0. Kaldiracli ETF Zaman Bazli Cikis (Volatility Decay Korumasi)
    if pos["max_hold"] is not None and pos["days_held"] >= pos["max_hold"]:
        return close * (1.0 - SLIPPAGE_RATE), False, 0.0, sl

    # 1. Overnight Gap Down
    if open_p <= sl:
        return open_p * (1.0 - SLIPPAGE_RATE), False, 0.0, sl

    # 2. Gun ici Stop-Loss
    if low <= sl:
        return sl * (1.0 - SLIPPAGE_RATE), False, 0.0, sl

    # 3. Parcali Kar Alma + Breakeven Kilidi
    partial_triggered = False
    partial_exec = 0.0
    if not scaled_out:
        if open_p >= partial_tp:
            partial_triggered = True
            partial_exec = open_p * (1.0 - SLIPPAGE_RATE)
            sl = max(sl, entry * 1.003)
        elif high >= partial_tp:
            partial_triggered = True
            partial_exec = partial_tp * (1.0 - SLIPPAGE_RATE)
            sl = max(sl, entry * 1.003)

    # 4. Ana Hedef veya Trend Kirilimi
    if open_p >= final_tp:
        return open_p * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl
    if high >= final_tp:
        return final_tp * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl
    if close < row["sma_20"]:
        return close * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl

    return None, partial_triggered, partial_exec, sl


def run_backtest_for_symbol(feat_df: pl.DataFrame, symbol: str, asset_type: str) -> dict:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    sub = feat_df.filter(pl.col("symbol") == symbol).sort("price_date")

    cash = PORTFOLIO_CAPITAL
    peak_equity = PORTFOLIO_CAPITAL
    max_drawdown_pct = 0.0
    daily_returns = []
    prev_equity = PORTFOLIO_CAPITAL

    pos = None
    trade_pnls_usd = []

    for row in sub.iter_rows(named=True):
        if row["sma_50"] is None or row["atr_14"] is None:
            continue

        close = row["close"]
        current_equity = cash + (pos["shares"] * close if pos else 0.0)
        if prev_equity > 0:
            daily_returns.append((current_equity - prev_equity) / prev_equity)
        prev_equity = current_equity

        if current_equity > peak_equity:
            peak_equity = current_equity
        dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        if pos is not None:
            pos["days_held"] += 1
            exit_price, partial_hit, partial_px, new_sl = evaluate_realistic_exit(row, pos)
            pos["sl"] = new_sl

            if partial_hit and not pos["scaled_out"] and pos["shares"] > 0:
                half_shares = pos["shares"] * 0.5
                proceeds = half_shares * partial_px * (1.0 - COMMISSION_RATE)
                pos["pnl"] += proceeds - (half_shares * pos["entry"])
                cash += proceeds
                pos["shares"] -= half_shares
                pos["scaled_out"] = True

            if exit_price is not None:
                proceeds = pos["shares"] * exit_price * (1.0 - COMMISSION_RATE)
                pos["pnl"] += proceeds - (pos["shares"] * pos["entry"])
                cash += proceeds
                trade_pnls_usd.append(pos["pnl"])
                pos = None

        else:
            score = compute_multifactor_score(row)
            if score >= 1.5 and can_enter_by_profile(row, profile):
                atr = row["atr_14"]
                alloc = calc_vol_targeted_allocation(cash, current_equity, close, atr, profile["sl_atr"])
                if alloc > 50.0:
                    exec_entry = close * (1.0 + SLIPPAGE_RATE)
                    shares = (alloc * (1.0 - COMMISSION_RATE)) / exec_entry
                    cash -= alloc
                    pos = {
                        "entry": exec_entry,
                        "sl": max(0.01, exec_entry - profile["sl_atr"] * atr),
                        "partial_tp": exec_entry + profile["tp1_atr"] * atr,
                        "final_tp": exec_entry + profile["tp2_atr"] * atr,
                        "shares": shares,
                        "scaled_out": False,
                        "days_held": 0,
                        "max_hold": profile["max_hold_days"],
                        "pnl": -(alloc * COMMISSION_RATE),
                    }

    if pos is not None:
        last_close = sub[-1, "close"] * (1.0 - SLIPPAGE_RATE)
        proceeds = pos["shares"] * last_close * (1.0 - COMMISSION_RATE)
        pos["pnl"] += proceeds - (pos["shares"] * pos["entry"])
        cash += proceeds
        trade_pnls_usd.append(pos["pnl"])

    wins = [p for p in trade_pnls_usd if p > 0]
    losses = [abs(p) for p in trade_pnls_usd if p <= 0]

    win_rate = (len(wins) / len(trade_pnls_usd) * 100.0) if trade_pnls_usd else 0.0
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    expectancy = (sum(trade_pnls_usd) / len(trade_pnls_usd)) if trade_pnls_usd else 0.0
    total_return = ((cash - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0

    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1))
        sharpe = ((mean_r / std_r) * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "trades": len(trade_pnls_usd),
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_drawdown_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "expectancy": round(expectancy, 2),
        "final_capital": round(cash, 2),
        "return_pct": round(total_return, 2),
    }


def main():
    asset_map = get_active_asset_map()
    symbols = list(asset_map.keys())

    query = """
        SELECT symbol, price_date,
               open_price::float AS open,
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

    df = pl.DataFrame(rows, schema=["symbol", "price_date", "open", "high", "low", "close", "volume"], orient="row")
    bench_df = build_benchmark_df(df)
    feat_df = prepare_features(df, symbols, bench_df)

    print("=" * 122)
    print("ALPHA ENGINE v4.7 BACKTEST [VARLIK SINIFI PROFİLLERİ: STOCK vs COMMODITY vs SECTOR_ETF vs LEVERAGED]")
    print("=" * 122)

    total_start = PORTFOLIO_CAPITAL * len(symbols)
    total_end = 0.0

    for sym in symbols:
        atype = asset_map[sym]
        res = run_backtest_for_symbol(feat_df, sym, atype)
        total_end += res["final_capital"]
        print(
            f"[{res['symbol']:4s} | {res['asset_type']:10s}] İşlem:{res['trades']:2d} | "
            f"Win:%{res['win_rate']:5.1f} | "
            f"MaxDD:%{res['max_dd']:5.2f} | "
            f"PF:{res['profit_factor']:4.2f} | "
            f"Sharpe:{res['sharpe']:5.2f} | "
            f"Exp:${res['expectancy']:+6.1f} | "
            f"Bakiye:${res['final_capital']:9.2f} ({res['return_pct']:+6.2f}%)"
        )

    total_ret = ((total_end - total_start) / total_start) * 100.0
    print("-" * 122)
    print(f"PORTFÖY TOPLAMI ({len(symbols)} Varlık): ${total_start:,.2f} -> ${total_end:,.2f} (Net Değişim: {total_ret:+.2f}%)")
    print("=" * 122)


if __name__ == "__main__":
    main()
