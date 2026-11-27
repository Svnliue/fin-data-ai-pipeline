import math
import polars as pl
from db import get_db_connection
from logger import get_logger
from config import (
    get_active_symbols,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    COMMISSION_RATE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
)

log = get_logger("backtest")
SLIPPAGE_RATE = 0.0005       # %0.05 piyasa kaymasi (bid-ask spread + execution slip)
CORR_THRESHOLD = 0.75        # 30 gunluk maksimum korelasyon esigi


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
    sub = df.filter(pl.col("symbol").is_in(get_active_symbols())).sort(["symbol", "price_date"])
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


def calc_vol_targeted_allocation(cash: float, equity: float, close: float, atr: float) -> float:
    stop_dist = max(0.01, 2.0 * atr)
    risk_budget = equity * RISK_PER_TRADE
    shares_by_risk = risk_budget / stop_dist
    usd_by_risk = shares_by_risk * close
    usd_by_cap = equity * MAX_POSITION_PCT
    return min(cash, usd_by_risk, usd_by_cap)


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 10:
        return 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return (num / den) if den > 1e-12 else 0.0


def evaluate_realistic_exit(row: dict, pos: dict) -> tuple[float | None, bool, float, float]:
    # Overnight Gap + Slippage gercekci cikis motoru
    open_p = row["open"]
    high = row["high"]
    low = row["low"]
    close = row["close"]

    sl = pos["sl"]
    partial_tp = pos["partial_tp"]
    final_tp = pos["final_tp"]
    scaled_out = pos["scaled_out"]
    entry = pos["entry"]

    # 1. Sabah Acilis Boslugu (Overnight Gap Down -> Stop altinda acilis)
    if open_p <= sl:
        return open_p * (1.0 - SLIPPAGE_RATE), False, 0.0, sl

    # 2. Gun ici Stop-Loss tetiklenmesi
    if low <= sl:
        return sl * (1.0 - SLIPPAGE_RATE), False, 0.0, sl

    # 3. Parcali Kar Alma (+1.5 ATR)
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

    # 4. Ana Kar-Al (+3.5 ATR) veya Trend Kirilimi (Close < SMA20)
    if open_p >= final_tp:
        return open_p * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl
    if high >= final_tp:
        return final_tp * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl
    if close < row["sma_20"]:
        return close * (1.0 - SLIPPAGE_RATE), partial_triggered, partial_exec, sl

    return None, partial_triggered, partial_exec, sl


def run_backtest_for_symbol(feat_df: pl.DataFrame, symbol: str) -> dict:
    sub = feat_df.filter(pl.col("symbol") == symbol).sort("price_date")

    cash = PORTFOLIO_CAPITAL
    peak_equity = PORTFOLIO_CAPITAL
    max_drawdown_pct = 0.0
    daily_returns = []
    prev_equity = PORTFOLIO_CAPITAL

    pos = None
    trade_pnls_usd = []
    blocked_entries = 0
    gap_sl_hits = 0

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
            if row["open"] <= pos["sl"]:
                gap_sl_hits += 1

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
            if score >= 1.5:
                stock_bull = (close >= row["sma_200"]) if row["sma_200"] is not None else True
                if not row["is_macro_bull"] or not stock_bull:
                    blocked_entries += 1
                    continue

                atr = row["atr_14"]
                alloc = calc_vol_targeted_allocation(cash, current_equity, close, atr)
                if alloc > 50.0:
                    exec_entry = close * (1.0 + SLIPPAGE_RATE)
                    shares = (alloc * (1.0 - COMMISSION_RATE)) / exec_entry
                    cash -= alloc
                    pos = {
                        "entry": exec_entry,
                        "sl": max(0.01, exec_entry - 2.0 * atr),
                        "partial_tp": exec_entry + 1.5 * atr,
                        "final_tp": exec_entry + 3.5 * atr,
                        "shares": shares,
                        "scaled_out": False,
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
        "trades": len(trade_pnls_usd),
        "blocked": blocked_entries,
        "gap_sl": gap_sl_hits,
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_drawdown_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "expectancy": round(expectancy, 2),
        "final_capital": round(cash, 2),
        "return_pct": round(total_return, 2),
    }


def run_correlated_portfolio_stress_test(feat_df: pl.DataFrame) -> dict:
    total_start = PORTFOLIO_CAPITAL * len(get_active_symbols())
    cash = total_start
    peak_equity = total_start
    max_dd_pct = 0.0
    positions: dict[str, dict] = {}
    ret_history: dict[str, list[float]] = {s: [] for s in get_active_symbols()}
    corr_blocks = 0
    slot_blocks = 0
    total_trades = 0

    dates = sorted(feat_df["price_date"].unique().to_list())
    rows_by_date: dict[object, list[dict]] = {d: [] for d in dates}
    for r in feat_df.iter_rows(named=True):
        rows_by_date[r["price_date"]].append(r)

    last_prices: dict[str, float] = {}

    for d in dates:
        day_rows = rows_by_date[d]
        for r in day_rows:
            last_prices[r["symbol"]] = r["close"]
            if r["daily_ret"] is not None:
                ret_history[r["symbol"]].append(r["daily_ret"])

        # 1. Cikislar (Overnight Gap + Slippage)
        for r in day_rows:
            sym = r["symbol"]
            if sym not in positions or r["sma_50"] is None:
                continue
            pos = positions[sym]
            exit_price, partial_hit, partial_px, new_sl = evaluate_realistic_exit(r, pos)
            pos["sl"] = new_sl

            if partial_hit and not pos["scaled_out"] and pos["shares"] > 0:
                half_shares = pos["shares"] * 0.5
                cash += half_shares * partial_px * (1.0 - COMMISSION_RATE)
                pos["shares"] -= half_shares
                pos["scaled_out"] = True

            if exit_price is not None:
                cash += pos["shares"] * exit_price * (1.0 - COMMISSION_RATE)
                del positions[sym]
                total_trades += 1

        # 2. Girisler (Max 3 Pozisyon + 30 Gunluk Korelasyon Kilidi > 0.75)
        for r in day_rows:
            sym = r["symbol"]
            if sym in positions or r["sma_50"] is None or r["atr_14"] is None:
                continue

            score = compute_multifactor_score(r)
            stock_bull = (r["close"] >= r["sma_200"]) if r["sma_200"] is not None else True

            if score >= 1.5 and r["is_macro_bull"] and stock_bull:
                if len(positions) >= MAX_OPEN_POSITIONS:
                    slot_blocks += 1
                    continue

                high_corr = False
                for open_sym in positions:
                    c = pearson_corr(ret_history[sym][-30:], ret_history[open_sym][-30:])
                    if c > CORR_THRESHOLD:
                        high_corr = True
                        break

                if high_corr:
                    corr_blocks += 1
                    continue

                equity_now = cash + sum(p["shares"] * last_prices.get(s, p["entry"]) for s, p in positions.items())
                alloc = calc_vol_targeted_allocation(cash, equity_now, r["close"], r["atr_14"])
                if alloc > 50.0:
                    exec_entry = r["close"] * (1.0 + SLIPPAGE_RATE)
                    shares = (alloc * (1.0 - COMMISSION_RATE)) / exec_entry
                    cash -= alloc
                    positions[sym] = {
                        "entry": exec_entry,
                        "sl": max(0.01, exec_entry - 2.0 * r["atr_14"]),
                        "partial_tp": exec_entry + 1.5 * r["atr_14"],
                        "final_tp": exec_entry + 3.5 * r["atr_14"],
                        "shares": shares,
                        "scaled_out": False,
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
        "slot_blocked": slot_blocks,
        "corr_blocked": corr_blocks,
    }


def main():
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
    feat_df = prepare_features(df, bench_df)

    print("=" * 118)
    print(f"STRESS TEST v4.5 [OVERNIGHT GAP + %{SLIPPAGE_RATE*100:.2f} SLIPPAGE + %{COMMISSION_RATE*100:.1f} KOMİSYON + CORR>{CORR_THRESHOLD} KİLİT]")
    print("=" * 118)

    total_start = PORTFOLIO_CAPITAL * len(get_active_symbols())
    total_end = 0.0

    for sym in get_active_symbols():
        res = run_backtest_for_symbol(feat_df, sym)
        total_end += res["final_capital"]
        log.info(f"[{sym}] StressRet={res['return_pct']:+.2f}% | MaxDD={res['max_dd']:.2f}% | PF={res['profit_factor']:.2f} | GapSL={res['gap_sl']}")
        print(
            f"[{res['symbol']:4s}] İşlem:{res['trades']:2d} (GapSL:{res['gap_sl']}) | "
            f"Win:%{res['win_rate']:5.1f} | "
            f"MaxDD:%{res['max_dd']:5.2f} | "
            f"PF:{res['profit_factor']:4.2f} | "
            f"Sharpe:{res['sharpe']:5.2f} | "
            f"Exp:${res['expectancy']:+6.1f} | "
            f"Bakiye:${res['final_capital']:9.2f} ({res['return_pct']:+6.2f}%)"
        )

    total_ret = ((total_end - total_start) / total_start) * 100.0
    shared = run_correlated_portfolio_stress_test(feat_df)
    log.info(f"[CORR_PORTFOLIO] End=${shared['end']} ({shared['return_pct']:+.2f}%) | MaxDD={shared['max_dd']:.2f}% | CorrBlocked={shared['corr_blocked']}")

    print("-" * 118)
    print(f"BİREYSEL TOPLAM (Stres Altında): ${total_start:,.2f} -> ${total_end:,.2f} (Net Değişim: {total_ret:+.2f}%)")
    print(
        f"KORELASYON KORUMALI PORTFÖY (Max {MAX_OPEN_POSITIONS} Pozisyon, Corr>{CORR_THRESHOLD} Kilidi): "
        f"${shared['start']:,.2f} -> ${shared['end']:,.2f} ({shared['return_pct']:+.2f}%) | "
        f"MaxDD: %{shared['max_dd']:.2f} | Korelasyon Kilidi: {shared['corr_blocked']}"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
