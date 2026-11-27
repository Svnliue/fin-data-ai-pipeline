import math
import polars as pl
from signal_engine import load_prices_df, get_benchmark_metrics
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
    COMMISSION_RATE,
)

WARMUP_BARS = 200


def _prepare_symbol_df(df: pl.DataFrame, symbol: str, qqq_df: pl.DataFrame) -> pl.DataFrame:
    sub = df.filter(pl.col("symbol") == symbol).sort("price_date").join(qqq_df, on="price_date", how="left")
    sub = sub.with_columns([
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
        pl.col("volume").rolling_mean(window_size=20).alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("ret_20d"),
        (pl.col("close") - pl.col("close").shift(1)).alias("delta"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]).alias("tr"),
    ])
    sub = sub.with_columns([
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
        pl.col("tr").rolling_mean(window_size=14).alias("atr_14"),
        (pl.col("volume") / (pl.col("vol_sma_20") + 1e-9)).alias("vol_ratio"),
        ((pl.col("ret_20d") - pl.col("qqq_ret_20d")) * 100.0).alias("rs_vs_qqq_pct"),
    ])
    sub = sub.with_columns([
        pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14).alias("avg_loss"),
    ])
    return sub.with_columns(
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_14")
    )


def _simulate_slice(rows: list[dict], asset_type: str) -> dict:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    cash = PORTFOLIO_CAPITAL
    shares = 0.0
    entry_px = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    scaled_out = False
    bars_held = 0
    trades_pnl = []
    equity_curve = []

    for r in rows:
        close = r["close"]
        atr = r["atr_14"] or (close * 0.02)
        sma20 = r["sma_20"]
        sma50 = r["sma_50"]
        sma200 = r["sma_200"]
        rsi = r["rsi_14"]
        vol_ratio = r["vol_ratio"] or 1.0
        rs_pct = r["rs_vs_qqq_pct"] or 0.0
        is_macro_bull = bool(r["qqq_bull"]) if r["qqq_bull"] is not None else True

        if shares > 0:
            bars_held += 1
            # Gercek Trailing Stop: sadece yukari guncellenir
            stop_loss = max(stop_loss, close - (profile["sl_atr"] * atr))
            max_hold = profile["max_hold_days"]

            if close <= stop_loss or (max_hold is not None and bars_held >= max_hold) or close >= tp2:
                proceeds = shares * close * (1.0 - COMMISSION_RATE)
                trades_pnl.append(proceeds - (shares * entry_px))
                cash += proceeds
                shares = 0.0
                bars_held = 0
                scaled_out = False
            elif close >= tp1 and not scaled_out and shares >= 2:
                sell_qty = math.floor(shares / 2.0)
                proceeds = sell_qty * close * (1.0 - COMMISSION_RATE)
                trades_pnl.append(proceeds - (sell_qty * entry_px))
                cash += proceeds
                shares -= sell_qty
                scaled_out = True
                stop_loss = max(stop_loss, entry_px * 1.002)
        else:
            score = 0.0
            if sma20 and sma50:
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

            macro_ok = is_macro_bull if profile["require_macro_bull"] else True
            sma200_ok = (close >= sma200) if (profile["require_stock_sma200"] and sma200 is not None) else True

            if score >= 0.75 and macro_ok and sma200_ok:
                risk_per_share = max(0.01, profile["sl_atr"] * atr)
                qty = min(
                    int((cash * RISK_PER_TRADE) // risk_per_share),
                    int((cash * MAX_POSITION_PCT) // max(0.01, close)),
                )
                cost = qty * close * (1.0 + COMMISSION_RATE)
                if qty > 0 and cost <= cash:
                    cash -= cost
                    shares = float(qty)
                    entry_px = close
                    stop_loss = close - (profile["sl_atr"] * atr)
                    tp1 = close + (profile["tp1_atr"] * atr)
                    tp2 = close + (profile["tp2_atr"] * atr)
                    scaled_out = False
                    bars_held = 0

        equity_curve.append(cash + (shares * close))

    final_eq = equity_curve[-1] if equity_curve else PORTFOLIO_CAPITAL
    ret_pct = ((final_eq - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    bh_ret = (((rows[-1]["close"] - rows[0]["close"]) / rows[0]["close"]) * 100.0) if len(rows) > 1 else 0.0
    wins = sum(1 for p in trades_pnl if p > 0)
    win_rate = (wins / len(trades_pnl) * 100.0) if trades_pnl else 0.0

    peak = PORTFOLIO_CAPITAL
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = ((eq - peak) / peak) * 100.0
        max_dd = min(max_dd, dd)

    return {
        "return_pct": round(ret_pct, 2),
        "buy_hold_pct": round(bh_ret, 2),
        "alpha_vs_bh_pct": round(ret_pct - bh_ret, 2),
        "trades": len(trades_pnl),
        "win_rate_pct": round(win_rate, 1),
        "max_dd_pct": round(max_dd, 2),
    }


def run_backtest() -> list[dict]:
    df = load_prices_df()
    asset_map = get_active_asset_map()

    qqq = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date").with_columns([
        pl.col("close").rolling_mean(window_size=200).alias("qqq_sma_200"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("qqq_ret_20d"),
    ]).select([
        "price_date",
        (pl.col("close") >= pl.col("qqq_sma_200")).alias("qqq_bull"),
        "qqq_ret_20d",
    ])

    results = []
    for sym, atype in sorted(asset_map.items()):
        sub = _prepare_symbol_df(df, sym, qqq)
        all_rows = sub.to_dicts()
        if len(all_rows) <= WARMUP_BARS + 30:
            continue
        valid_rows = all_rows[WARMUP_BARS:]
        split_idx = int(len(valid_rows) * 0.7)
        in_sample = _simulate_slice(valid_rows[:split_idx], atype)
        out_sample = _simulate_slice(valid_rows[split_idx:], atype)
        results.append({
            "symbol": sym,
            "asset_type": atype,
            "bars_tested": len(valid_rows),
            "in_sample": in_sample,
            "out_of_sample": out_sample,
        })
    return results


def main():
    results = run_backtest()
    print("\n" + "=" * 100)
    print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RAPORU (200-BAR WARM-UP + %70 IN-SAMPLE / %30 OUT-OF-SAMPLE)")
    print("=" * 100)
    for r in results:
        ins = r["in_sample"]
        oos = r["out_of_sample"]
        print(
            f"[{r['symbol']:<6} | {r['asset_type']:<10}] Bar: {r['bars_tested']} | "
            f"IS Getiri: {ins['return_pct']:+6.2f}% (Alpha: {ins['alpha_vs_bh_pct']:+6.2f}%) | "
            f"OOS Getiri: {oos['return_pct']:+6.2f}% (B&H: {oos['buy_hold_pct']:+6.2f}%, Alpha: {oos['alpha_vs_bh_pct']:+6.2f}%) | "
            f"OOS İşlem: {oos['trades']} | Win: %{oos['win_rate_pct']:.1f} | MaxDD: {oos['max_dd_pct']:.2f}%"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
