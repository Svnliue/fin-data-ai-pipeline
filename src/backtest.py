import math
import polars as pl
from indicators import add_technical_indicators
from signal_engine import load_prices_df
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
)

WARMUP_BARS = 200


def _simulate_slice(rows: list[dict], asset_type: str) -> dict:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    cash = PORTFOLIO_CAPITAL
    shares = 0.0
    entry_px = 0.0
    position_cost = 0.0
    partial_realized = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    scaled_out = False
    bars_held = 0
    pending_entry = False

    closed_trade_pnls = []
    equity_curve = []

    for i, r in enumerate(rows):
        open_px = r["open"]
        high_px = r["high"]
        low_px = r["low"]
        close = r["close"]
        atr = r["atr_14"] or (close * 0.02)
        sma20 = r["sma_20"]
        sma50 = r["sma_50"]
        sma200 = r["sma_200"]
        rsi = r["rsi_14"]
        vol_ratio = r["vol_ratio"] or 1.0
        rs_pct = r["rs_vs_qqq_pct"] or 0.0
        is_macro_bull = bool(r["qqq_bull"]) if r["qqq_bull"] is not None else True

        # Bir onceki bar kapanisinda uretilen sinyale ertesi gun acilista (next_open + slippage) giris
        if pending_entry and shares == 0:
            exec_entry = open_px * (1.0 + SLIPPAGE_RATE)
            risk_per_share = max(0.01, profile["sl_atr"] * atr)
            qty = min(
                int((cash * RISK_PER_TRADE) // risk_per_share),
                int((cash * MAX_POSITION_PCT) // max(0.01, exec_entry)),
            )
            total_cost = qty * exec_entry * (1.0 + COMMISSION_RATE)
            if qty > 0 and total_cost <= cash:
                cash -= total_cost
                shares = float(qty)
                entry_px = exec_entry
                position_cost = total_cost
                partial_realized = 0.0
                stop_loss = exec_entry - (profile["sl_atr"] * atr)
                tp1 = exec_entry + (profile["tp1_atr"] * atr)
                tp2 = exec_entry + (profile["tp2_atr"] * atr)
                scaled_out = False
                bars_held = 0
            pending_entry = False

        # Teknik skor (canli _score_factors ile birebir hizali)
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

        if shares > 0:
            bars_held += 1
            # Gercek Trailing Stop: sadece yukari gider
            stop_loss = max(stop_loss, close - (profile["sl_atr"] * atr))
            max_hold = profile["max_hold_days"]

            # Gun ici stop (gap dahil) ve trend bozulmasi cikisi (canli kuralla ayni)
            hit_stop = low_px <= stop_loss
            hit_tp2 = high_px >= tp2
            trend_broken = (sma20 is not None and close < sma20 and score < 0)
            time_expired = (max_hold is not None and bars_held >= max_hold)

            if hit_stop or hit_tp2 or trend_broken or time_expired:
                if hit_stop:
                    raw_exit = min(open_px, stop_loss) * (1.0 - SLIPPAGE_RATE)
                elif hit_tp2:
                    raw_exit = max(open_px, tp2) * (1.0 - SLIPPAGE_RATE)
                else:
                    raw_exit = close * (1.0 - SLIPPAGE_RATE)

                proceeds = shares * raw_exit * (1.0 - COMMISSION_RATE)
                total_trade_pnl = (partial_realized + proceeds) - position_cost
                closed_trade_pnls.append(total_trade_pnl)
                cash += proceeds
                shares = 0.0
                bars_held = 0
                scaled_out = False
            elif high_px >= tp1 and not scaled_out and shares >= 2:
                sell_qty = math.floor(shares / 2.0)
                raw_tp1 = max(open_px, tp1) * (1.0 - SLIPPAGE_RATE)
                proceeds = sell_qty * raw_tp1 * (1.0 - COMMISSION_RATE)
                partial_realized += proceeds
                cash += proceeds
                shares -= sell_qty
                scaled_out = True
                stop_loss = max(stop_loss, entry_px * 1.002)
        else:
            macro_ok = is_macro_bull if profile["require_macro_bull"] else True
            sma200_ok = (close >= sma200) if (profile["require_stock_sma200"] and sma200 is not None) else True
            if score >= 0.75 and macro_ok and sma200_ok and i < len(rows) - 1:
                pending_entry = True

        equity_curve.append(cash + (shares * close))

    final_eq = equity_curve[-1] if equity_curve else PORTFOLIO_CAPITAL
    ret_pct = ((final_eq - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    bh_ret = (((rows[-1]["close"] - rows[0]["close"]) / rows[0]["close"]) * 100.0) if len(rows) > 1 else 0.0

    wins = [p for p in closed_trade_pnls if p > 0]
    losses = [abs(p) for p in closed_trade_pnls if p < 0]
    win_rate = (len(wins) / len(closed_trade_pnls) * 100.0) if closed_trade_pnls else 0.0
    profit_factor = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
    expectancy = (sum(closed_trade_pnls) / len(closed_trade_pnls)) if closed_trade_pnls else 0.0

    # Gunluk getiri uzerinden yillik Sharpe
    daily_rets = []
    for idx in range(1, len(equity_curve)):
        prev_e = equity_curve[idx - 1]
        if prev_e > 0:
            daily_rets.append((equity_curve[idx] - prev_e) / prev_e)
    if len(daily_rets) > 1:
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in daily_rets) / (len(daily_rets) - 1))
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

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
        "trades": len(closed_trade_pnls),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "expectancy_usd": round(expectancy, 2),
        "sharpe": round(sharpe, 2),
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

    enriched = add_technical_indicators(df, qqq.select(["price_date", "qqq_ret_20d"])).join(
        qqq.select(["price_date", "qqq_bull"]), on="price_date", how="left"
    )

    results = []
    for sym, atype in sorted(asset_map.items()):
        all_rows = enriched.filter(pl.col("symbol") == sym).sort("price_date").to_dicts()
        if len(all_rows) <= WARMUP_BARS + 30:
            continue
        valid_rows = all_rows[WARMUP_BARS:]
        split_idx = int(len(valid_rows) * 0.7)
        regime_a = _simulate_slice(valid_rows[:split_idx], atype)
        regime_b = _simulate_slice(valid_rows[split_idx:], atype)
        results.append({
            "symbol": sym,
            "asset_type": atype,
            "bars_tested": len(valid_rows),
            "period_1": regime_a,
            "period_2": regime_b,
        })
    return results


def main():
    results = run_backtest()
    print("\n" + "=" * 112)
    print("İKİ DÖNEMLİ REJİM DOĞRULAMA RAPORU (NEXT-OPEN + SLIPPAGE + CANLI ÇIKIŞ KURALLARI | 200-BAR WARM-UP)")
    print("=" * 112)
    for r in results:
        p1 = r["period_1"]
        p2 = r["period_2"]
        print(
            f"[{r['symbol']:<6} | {r['asset_type']:<10}] Bar: {r['bars_tested']} | "
            f"Dönem-1: {p1['return_pct']:+6.2f}% (PF: {p1['profit_factor']:.2f}) | "
            f"Dönem-2: {p2['return_pct']:+6.2f}% (B&H: {p2['buy_hold_pct']:+6.2f}%) | "
            f"İşlem: {p2['trades']} | Win: %{p2['win_rate_pct']:.1f} | PF: {p2['profit_factor']:.2f} | "
            f"Sharpe: {p2['sharpe']:.2f} | Exp: ${p2['expectancy_usd']:+.2f} | MaxDD: {p2['max_dd_pct']:.2f}%"
        )
    print("=" * 112)


if __name__ == "__main__":
    main()
